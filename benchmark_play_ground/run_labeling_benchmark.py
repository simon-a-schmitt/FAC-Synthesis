#!/usr/bin/env python3
"""How well do the API models answer the benchmarks in the labeling format?

For every benchmark of experiments/config/experiments.yaml (test_tsv, max_prompts) and every API
model (deepseek, llama, gpt; benchmark_play_ground/api_prediction.py API_MODEL_PRESETS) the test
set is sent once to the model, with the labeling prompt of data_synthesis/prompts/<domain>/labeling.py
(SYSTEM_PROMPT + USER_PROMPT_PREFIX) - the plain arm, no few-shot examples. Temperature 0,
max_tokens 64 (= run_labeling.py LABELING_MODEL_PARAMS) for all benchmarks, logprobs off. No retries:
one call per example; a call that still fails after the retry budget of call_openrouter_chat aborts
the run (rerun to continue - finished calls are kept, see below).

Every answer is parsed in two ways:
  exact        the labeling parser (data_synthesis/labeling/run_labeling.py build_answer_regex +
               normalize_label_line): one line must match the full answer template. All slots or none.
  recoverable  the lenient parser of the run_*_benchmark.py --mode api paths, slot by slot
               (CVSS_METRIC_RE after cut_at_stop_strings, CLAUDETTE_SLOT_RE, TOXICITY_ANSWER_RE).
               A slot that cannot be parsed is invalid, the parsed slots of the same answer still
               count. An answer that parses exactly takes its exact values, so every exact answer
               is also recoverable.

Metrics, once per parse group (only exact answers valid / recoverable slots valid too):
an invalid slot prediction counts as a false negative for its gold class and never as a false
positive (it is a prediction that matches no class).
  toxicity_detection  precision / recall / F1 of "toxic" and of "safe"
  claudette_tos       8-class scenario of run_claudette_benchmark.py: per clause type F1 with "Y" as
                      positive, micro-F1 (TP/FP/FN pooled over the 8) and macro-F1 (mean of the 8)
  cti_vsp             per metric and class F1 over the gold classes observed in the test set,
                      macro-F1 = mean of the 8 per-metric macro-F1s, micro-F1 pooled over every
                      (metric, class) pair (as run_cti_vsp_benchmark.py)

Outputs under --output-dir (default benchmark_play_ground/labeling_benchmark/):
  <bench>/<model>/raw_outputs.jsonl  one line per API call (prompt, gt, raw text, OpenRouter
                                     metadata); append-only, a rerun only queries missing indices
  <bench>/<model>/predictions.jsonl  parsed values per example (exact + recoverable), rewritten
                                     from raw_outputs.jsonl on every run
  <bench>/<model>/summary.json       metrics of both parse groups, parse counts, API parameters
  summary.json / summary.csv         all benchmarks x models x parse groups (CSV in long format:
                                     one row per metric)

Run with the benchmark env (fac_env) from the repo root, e.g.
  python benchmark_play_ground/run_labeling_benchmark.py --benchmarks cti_vsp --models gpt
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_SYNTHESIS_DIR = ROOT_DIR / "data_synthesis"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(DATA_SYNTHESIS_DIR))
sys.path.insert(0, str(DATA_SYNTHESIS_DIR / "labeling"))

from benchmark_play_ground.api_prediction import (  # noqa: E402
    API_MODEL_PRESETS, ApiPredictor, add_api_cli_args, first_messages_dump,
)
from benchmark_play_ground.data_loader import load_claudette_tsv, load_cti_vsp_tsv, load_toxicity_tsv  # noqa: E402
from prompts import LabelingPrompt, load_labeling_prompt  # noqa: E402
from run_labeling import build_answer_regex, normalize_label_line  # noqa: E402

DEFAULT_CONFIG = ROOT_DIR / "experiments" / "config" / "experiments.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "labeling_benchmark"

MAX_TOKENS = 64
PARSE_GROUPS = ("exact", "recoverable")

LOADERS = {
    "toxicity_detection": load_toxicity_tsv,
    "claudette_tos": load_claudette_tsv,
    "cti_vsp": load_cti_vsp_tsv,
}

# ---------------------------------------------------------------------------
# Lenient benchmark parsers, copied from the --mode api paths of the run_*_benchmark.py scripts
# (not imported: those modules import torch at module level).
# ---------------------------------------------------------------------------

TOXICITY_SLOT = "label"
TOXICITY_CLASSES = ["toxic", "safe"]
TOXICITY_ANSWER_RE = re.compile(r"Answer:\s*(toxic|safe)", re.IGNORECASE)  # run_toxicity_benchmark.py

CLAUDETTE_METRICS = ["LTD", "TER", "CH", "CR", "USE", "LAW", "J", "ARB"]
CLAUDETTE_SLOT_RE = {m: re.compile(rf"\b{m}:\s*([YN])\b") for m in CLAUDETTE_METRICS}  # run_claudette_benchmark.py

CVSS_METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
CVSS_METRIC_VALUE_CHARS = {  # run_cti_vsp_benchmark.py
    "AV": "NALP", "AC": "LH", "PR": "NLH", "UI": "NR", "S": "UC", "C": "NLH", "I": "NLH", "A": "NLH",
}
CVSS_METRIC_RE = {m: re.compile(rf"\b{m}:\s*([{chars}])") for m, chars in CVSS_METRIC_VALUE_CHARS.items()}
CTI_VSP_STOP_STRINGS = ["\nCVE Description:", "\n\n"]


def cut_at_stop_strings(text: str) -> str:
    """run_cti_vsp_benchmark.cut_at_stop_strings: the text up to the first stop string."""
    text = text.lstrip()
    cut = min((i for i in (text.find(stop) for stop in CTI_VSP_STOP_STRINGS) if i >= 0), default=len(text))
    return text[:cut]


def recover_toxicity(text: str) -> dict:
    match = TOXICITY_ANSWER_RE.search(text)
    return {TOXICITY_SLOT: match.group(1).lower() if match else None}


def recover_claudette(text: str) -> dict:
    out = {}
    for m in CLAUDETTE_METRICS:
        match = CLAUDETTE_SLOT_RE[m].search(text)
        out[m] = match.group(1) if match else None
    return out


def recover_cti_vsp(text: str) -> dict:
    text = cut_at_stop_strings(text)
    out = {}
    for m in CVSS_METRICS:
        match = CVSS_METRIC_RE[m].search(text)
        out[m] = match.group(1) if match else None
    return out


RECOVER = {
    "toxicity_detection": recover_toxicity,
    "claudette_tos": recover_claudette,
    "cti_vsp": recover_cti_vsp,
}

SLOTS = {
    "toxicity_detection": [TOXICITY_SLOT],
    "claudette_tos": CLAUDETTE_METRICS,
    "cti_vsp": CVSS_METRICS,
}


# ---------------------------------------------------------------------------
# Exact labeling parser (run_labeling.py)
# ---------------------------------------------------------------------------

class ExactParser:
    """normalize_label_line of run_labeling.py, with the slot values of the normalized line."""

    def __init__(self, domain: str, prompt: LabelingPrompt):
        self.fragments = prompt.fragments
        self.regex = build_answer_regex(self.fragments)
        self.slots = SLOTS[domain]
        if self.fragments.get("kind") == "vector" and list(self.fragments["field_order"]) != self.slots:
            raise SystemExit(f"{domain}: labeling fields {self.fragments['field_order']} != benchmark slots {self.slots}")

    def parse(self, text: str) -> tuple[str | None, dict | None]:
        """(normalized label line, {slot: value}) or (None, None) if no line matches the template."""
        line = normalize_label_line(text, self.fragments, self.regex)
        if line is None:
            return None, None
        match = self.regex.match(line)
        if self.fragments.get("kind") == "vector":
            return line, {f: match.group(f).upper() for f in self.slots}
        return line, {self.slots[0]: match.group(1).lower()}


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def build_chat_messages(args, query_prompt: str) -> list[dict]:
    """Plain arm of the run_*_benchmark.py build_chat_messages(), with the labeling prompt of the
    domain (args.labeling_prompt): system turn SYSTEM_PROMPT, user turn USER_PROMPT_PREFIX + query
    (cti_vsp: "CVE Description: <description>", as in run_cti_vsp_benchmark.py)."""
    prompt: LabelingPrompt = args.labeling_prompt
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user_content(query_prompt.strip())},
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def class_stats(pairs: list[tuple[str, str | None]], cls: str) -> dict:
    """pairs = (gold, predicted); predicted None = invalid -> FN for its gold class, never FP."""
    tp = sum(1 for g, p in pairs if p == cls and g == cls)
    fp = sum(1 for g, p in pairs if p == cls and g != cls)
    fn = sum(1 for g, p in pairs if p != cls and g == cls)
    stats = prf(tp, fp, fn)
    stats["support"] = sum(1 for g, _ in pairs if g == cls)
    return stats


def slot_pairs(rows: list[dict], group: str, slot: str) -> list[tuple[str, str | None]]:
    return [(r["gt_values"][slot], r[group]["values"][slot]) for r in rows]


def metrics_toxicity(rows: list[dict], group: str) -> dict:
    pairs = slot_pairs(rows, group, TOXICITY_SLOT)
    per_class = {cls: class_stats(pairs, cls) for cls in TOXICITY_CLASSES}
    return {
        "f1_toxic": per_class["toxic"]["f1"],
        "f1_safe": per_class["safe"]["f1"],
        "per_class": per_class,
    }


def metrics_claudette(rows: list[dict], group: str) -> dict:
    """8-class scenario of run_claudette_benchmark.py: class m is "slot m answered Y"."""
    per_class = {}
    for m in CLAUDETTE_METRICS:
        pairs = [("Y" if g == "Y" else "N", p) for g, p in slot_pairs(rows, group, m)]
        per_class[m] = class_stats(pairs, "Y")
    micro = prf(*(sum(per_class[m][k] for m in CLAUDETTE_METRICS) for k in ("tp", "fp", "fn")))
    return {
        "micro_f1": micro["f1"],
        "macro_f1": sum(per_class[m]["f1"] for m in CLAUDETTE_METRICS) / len(CLAUDETTE_METRICS),
        "per_class": per_class,
    }


def metrics_cti_vsp(rows: list[dict], group: str, metric_classes: dict[str, list[str]]) -> dict:
    """As run_cti_vsp_benchmark.evaluate_cti_vsp_predictions, but over all examples: classes are
    the gold classes observed in the test set, invalid slots count as FN."""
    per_metric = {}
    tp = fp = fn = 0
    for m in CVSS_METRICS:
        pairs = slot_pairs(rows, group, m)
        classes = {cls: class_stats(pairs, cls) for cls in metric_classes[m]}
        supported = [s["f1"] for s in classes.values() if s["support"] > 0]
        per_metric[m] = {"macro_f1": sum(supported) / len(supported) if supported else 0.0, "classes": classes}
        tp += sum(s["tp"] for s in classes.values())
        fp += sum(s["fp"] for s in classes.values())
        fn += sum(s["fn"] for s in classes.values())
    return {
        "micro_f1": prf(tp, fp, fn)["f1"],
        "macro_f1": sum(per_metric[m]["macro_f1"] for m in CVSS_METRICS) / len(CVSS_METRICS),
        "per_metric": per_metric,
    }


def parse_counts(rows: list[dict], slots: list[str]) -> dict:
    n_invalid_slots = {
        group: {s: sum(1 for r in rows if r[group]["values"][s] is None) for s in slots} for group in PARSE_GROUPS
    }
    return {
        "n_examples": len(rows),
        "n_exact": sum(1 for r in rows if r["parse_status"] == "exact"),
        "n_recoverable_only": sum(1 for r in rows if r["parse_status"] == "recoverable"),
        "n_partially_recoverable": sum(1 for r in rows if r["parse_status"] == "partial"),
        "n_invalid": sum(1 for r in rows if r["parse_status"] == "invalid"),
        "n_invalid_slots": {g: sum(v.values()) for g, v in n_invalid_slots.items()},
        "n_invalid_slots_by_slot": n_invalid_slots,
    }


def evaluate(domain: str, rows: list[dict], metric_classes: dict) -> dict:
    out = {}
    for group in PARSE_GROUPS:
        if domain == "toxicity_detection":
            out[group] = metrics_toxicity(rows, group)
        elif domain == "claudette_tos":
            out[group] = metrics_claudette(rows, group)
        else:
            out[group] = metrics_cti_vsp(rows, group, metric_classes)
    return out


def flat_metrics(domain: str, metrics: dict) -> dict[str, float]:
    """Metric name -> value of one parse group, for summary.csv."""
    if domain == "toxicity_detection":
        flat = {f"{cls}/{k}": s[k] for cls, s in metrics["per_class"].items() for k in ("precision", "recall", "f1")}
        return {"f1_toxic": metrics["f1_toxic"], "f1_safe": metrics["f1_safe"], **flat}
    flat = {"micro_f1": metrics["micro_f1"], "macro_f1": metrics["macro_f1"]}
    if domain == "claudette_tos":
        flat.update({f"f1/{m}": s["f1"] for m, s in metrics["per_class"].items()})
    else:
        for m, pm in metrics["per_metric"].items():
            flat[f"macro_f1/{m}"] = pm["macro_f1"]
            flat.update({f"f1/{m}/{cls}": s["f1"] for cls, s in pm["classes"].items()})
    return flat


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--benchmarks", nargs="+", choices=sorted(LOADERS), default=None,
                   help="Default: every benchmark of --config.")
    p.add_argument("--models", nargs="+", choices=sorted(API_MODEL_PRESETS), default=None,
                   help="API models. Default: --api-model if given, else api_models of --config.")
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Override max_prompts of --config (e.g. for a smoke test; 0 = all).")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Requests per chunk sent in parallel and appended to raw_outputs.jsonl together.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    add_api_cli_args(p)
    return p.parse_args()


def load_raw(path: Path, records: list[dict], api_model: str) -> dict[int, dict]:
    """index -> raw call record of an earlier run; refuses a file from another model or test set."""
    done: dict[int, dict] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by a killed run; its index is queried again
            i = rec["index"]
            if rec["api_model"] != api_model or i >= len(records) or rec["prompt"] != records[i]["prompt"]:
                raise SystemExit(f"{path}: line for index {i} does not belong to model {api_model} / this test set.")
            done[i] = rec
    return done


def query_missing(run_args, records: list[dict], raw_path: Path) -> dict[int, dict]:
    done = load_raw(raw_path, records, run_args.api_model)
    todo = [i for i in range(len(records)) if i not in done]
    print(f"[{run_args.domain} / {run_args.api_model}] {len(done)} answers found, {len(todo)} to query", flush=True)
    if not todo:
        return done
    predictor = ApiPredictor(run_args, max_tokens=MAX_TOKENS, logprobs=False)
    first_messages_dump(build_chat_messages(run_args, records[todo[0]]["prompt"]))
    batch_size = max(1, run_args.batch_size)
    with raw_path.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            responses = predictor.predict_many([build_chat_messages(run_args, records[i]["prompt"]) for i in batch])
            for i, resp in zip(batch, responses):
                rec = {
                    "index": i,
                    "api_model": predictor.api_model,
                    "model_id": predictor.model_id,
                    "prompt": records[i]["prompt"],
                    "gt": records[i]["gt"],
                    "raw_output": resp["text"],
                    "openrouter": resp["openrouter"],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[i] = rec
            fh.flush()
            print(f"  [{min(start + batch_size, len(todo))}/{len(todo)}]", flush=True)
    run_args.api_describe = predictor.describe()
    return done


def parse_row(domain: str, exact: ExactParser, rec: dict, gt_values: dict) -> dict:
    slots = SLOTS[domain]
    text = rec["raw_output"]
    label_line, exact_values = exact.parse(text)
    benchmark_values = RECOVER[domain](text)
    # An exact answer keeps its exact values in the recoverable group (exact is a subset of it).
    recoverable_values = exact_values if exact_values is not None else benchmark_values
    n_recovered = sum(1 for s in slots if recoverable_values[s] is not None)
    if exact_values is not None:
        status = "exact"
    elif n_recovered == len(slots):
        status = "recoverable"
    elif n_recovered:
        status = "partial"
    else:
        status = "invalid"
    return {
        "index": rec["index"],
        "api_model": rec["api_model"],
        "prompt": rec["prompt"],
        "gt": rec["gt"],
        "gt_values": gt_values,
        "raw_output": text,
        "parse_status": status,
        "exact": {"valid": exact_values is not None, "label_line": label_line,
                  "values": exact_values or {s: None for s in slots}},
        "recoverable": {"valid_slots": n_recovered, "values": recoverable_values,
                        "benchmark_parser_values": benchmark_values},
    }


def run_one(cli, domain: str, model: str, records: list[dict], gt_values: list[dict],
            prompt: LabelingPrompt, metric_classes: dict, data_info: dict) -> None:
    out_dir = cli.output_dir / domain / model
    out_dir.mkdir(parents=True, exist_ok=True)
    run_args = argparse.Namespace(**vars(cli))
    run_args.api_model, run_args.domain, run_args.labeling_prompt, run_args.api_describe = model, domain, prompt, None

    raw = query_missing(run_args, records, out_dir / "raw_outputs.jsonl")
    exact = ExactParser(domain, prompt)
    rows = [parse_row(domain, exact, raw[i], gt_values[i]) for i in range(len(records))]
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "benchmark": domain,
        "api_model": model,
        "model_id": raw[0]["model_id"],
        "max_tokens": MAX_TOKENS,
        "api": run_args.api_describe,  # None if every answer came from an earlier run
        **data_info,
        "parse_counts": parse_counts(rows, SLOTS[domain]),
        "metrics": evaluate(domain, rows, metric_classes),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print_summary(summary)


def print_summary(s: dict) -> None:
    c = s["parse_counts"]
    print(f"== {s['benchmark']} / {s['api_model']}: n={c['n_examples']}  exact={c['n_exact']}  "
          f"recoverable_only={c['n_recoverable_only']}  partial={c['n_partially_recoverable']}  "
          f"invalid={c['n_invalid']}  invalid_slots={c['n_invalid_slots']}")
    for group in PARSE_GROUPS:
        m = s["metrics"][group]
        if s["benchmark"] == "toxicity_detection":
            print(f"   {group:<11} f1_toxic={m['f1_toxic']:.4f}  f1_safe={m['f1_safe']:.4f}")
        else:
            print(f"   {group:<11} micro_f1={m['micro_f1']:.4f}  macro_f1={m['macro_f1']:.4f}")


def write_overall_summary(output_dir: Path) -> None:
    """summary.json / summary.csv over every <bench>/<model>/summary.json in output_dir, so runs of
    single benchmarks or models add up instead of replacing each other."""
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(output_dir.glob("*/*/summary.json"))]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["benchmark", "api_model", "parse_group", "n_examples", "n_invalid_slots", "metric", "value"])
        for s in summaries:
            for group in PARSE_GROUPS:
                for name, value in flat_metrics(s["benchmark"], s["metrics"][group]).items():
                    writer.writerow([s["benchmark"], s["api_model"], group, s["parse_counts"]["n_examples"],
                                     s["parse_counts"]["n_invalid_slots"][group], name, f"{value:.6f}"])


def main():
    cli = parse_args()
    config = yaml.safe_load(cli.config.read_text(encoding="utf-8"))
    benchmarks = cli.benchmarks or list(config["benchmarks"])
    models = cli.models or ([cli.api_model] if cli.api_model else list(config["api_models"]))

    for domain in benchmarks:
        bench_cfg = config["benchmarks"][domain]
        test_tsv = ROOT_DIR / bench_cfg["test_tsv"]
        max_prompts = bench_cfg["max_prompts"] if cli.max_prompts is None else cli.max_prompts
        records = LOADERS[domain](str(test_tsv))
        if max_prompts > 0:
            records = records[:max_prompts]
        if not records:
            raise SystemExit(f"No records loaded from {test_tsv}")

        prompt = load_labeling_prompt(domain)
        exact = ExactParser(domain, prompt)
        gt_values = []
        for i, rec in enumerate(records):
            _, values = exact.parse(rec["gt"])
            if values is None:
                raise SystemExit(f"{test_tsv}: gt of record {i} is not in the labeling format: {rec['gt']!r}")
            gt_values.append(values)
        # cti_vsp: classes per metric = gold classes observed in the test set (run_cti_vsp_benchmark.py)
        metric_classes = {s: sorted({v[s] for v in gt_values}) for s in SLOTS[domain]}
        data_info = {"test_tsv": str(test_tsv.relative_to(ROOT_DIR)), "max_prompts": max_prompts,
                     "n_records": len(records)}

        for model in models:
            run_one(cli, domain, model, records, gt_values, prompt, metric_classes, data_info)

    write_overall_summary(cli.output_dir)
    print(f"Overall summary written to {cli.output_dir / 'summary.json'} and summary.csv")


if __name__ == "__main__":
    main()
