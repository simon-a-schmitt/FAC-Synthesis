#!/usr/bin/env python3
"""How well do the API models answer the benchmarks in the labeling format?

For every benchmark of experiments/config/experiments.yaml (test_tsv, max_prompts) and every API
model (deepseek, llama, gpt; benchmark_play_ground/api_prediction.py API_MODEL_PRESETS) the test
set is sent once to the model, with the labeling prompt of data_synthesis/prompts/<domain>/labeling.py
(SYSTEM_PROMPT + USER_PROMPT_PREFIX) - the plain arm, no few-shot examples. Temperature 0,
max_tokens 64 (= run_labeling.py LABELING_MODEL_PARAMS) for all benchmarks, logprobs off. No retries:
one call per example; a call that still fails after the retry budget of call_openrouter_chat aborts
the run (rerun to continue - finished calls are kept, see below).

Every line of raw_outputs.jsonl carries a request fingerprint: a hash over the SHA-256 of the system
prompt, the user prompt prefix, max_tokens, the model id and the full request parameters. A rerun
reuses only lines with the current fingerprint and refuses to continue a file that holds others,
so a changed prompt, max_tokens or model id behind the same preset name never mixes old and new
answers (move the file away or pass another --output-dir).

Every answer is parsed in two ways:
  exact        the labeling parser (data_synthesis/labeling/run_labeling.py build_answer_regex +
               normalize_label_line): one line must match the full answer template. All slots or none.
  recoverable  a lenient parser over the whole answer, case-insensitive, Markdown emphasis/code
               characters (* _ ` # ~) removed first:
               toxicity_detection  exactly one of the class labels "toxic" / "safe" occurs as a word
                                   (not inside "non-toxic", "unsafe", "toxicity"); both or neither
                                   -> invalid
               claudette_tos,      per slot: every "<key> : <value>" / "<key> = <value>" occurrence,
               cti_vsp             key = abbreviation or metric name ("AV", "Attack Vector",
                                   "Attack Vector (AV)"), value = letter or its spelled-out name
                                   ("N" / "Network", "Y" / "Yes"); exactly one distinct value ->
                                   that value, none or conflicting values -> the slot is invalid
               The other slots of the same answer still count. An answer that parses exactly
               takes its exact values, so every exact answer is also recoverable.

Metrics, once per parse group (only exact answers valid / recoverable slots valid too):
an invalid slot prediction counts as a false negative for its gold class and never as a false
positive (it is a prediction that matches no class).
  toxicity_detection  precision / recall / F1 of "toxic" and of "safe", macro-F1 (mean of the two)
  claudette_tos       8-class scenario of run_claudette_benchmark.py: per clause type F1 with "Y" as
                      positive, micro-F1 (TP/FP/FN pooled over the 8) and macro-F1 (mean of the 8)
  cti_vsp             per metric and class F1 over the gold classes observed in the test set,
                      macro-F1 = mean of the 8 per-metric macro-F1s, micro-F1 pooled over every
                      (metric, class) pair (as run_cti_vsp_benchmark.py)

Outputs under --output-dir (default benchmark_play_ground/labeling_benchmark/):
  <bench>/<model>/raw_outputs.jsonl  one line per API call (prompt, gt, raw text, OpenRouter
                                     metadata, fingerprint); append-only, a rerun only queries
                                     missing indices
  <bench>/<model>/predictions.jsonl  parsed values per example (exact + recoverable), rewritten
                                     from raw_outputs.jsonl on every run
  <bench>/<model>/summary.json       metrics of both parse groups, parse counts, fingerprint and
                                     its components
  summary.json / summary.csv         all benchmarks x models x parse groups (CSV in long format:
                                     one row per metric)

Run with the benchmark env (fac_env) from the repo root, e.g.
  python benchmark_play_ground/run_labeling_benchmark.py --benchmarks cti_vsp --models gpt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
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
    API_BASE_PARAMS, API_MODEL_PRESETS, ApiPredictor, add_api_cli_args, first_messages_dump,
)
from benchmark_play_ground.data_loader import load_claudette_tsv, load_cti_vsp_tsv, load_toxicity_tsv  # noqa: E402
from prompts import LabelingPrompt, load_labeling_prompt  # noqa: E402
from run_labeling import build_answer_regex, normalize_label_line  # noqa: E402
from shared.openrouter import build_model_params  # noqa: E402

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
# Lenient parsers (recoverable group)
# ---------------------------------------------------------------------------

TOXICITY_SLOT = "label"
TOXICITY_CLASSES = ["toxic", "safe"]

CLAUDETTE_METRICS = ["LTD", "TER", "CH", "CR", "USE", "LAW", "J", "ARB"]
CVSS_METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

# Markdown emphasis / code / heading characters, removed before lenient parsing ("**AV:** `N`").
MARKDOWN_RE = re.compile(r"[*_`#~]")

# Spelled-out values per slot (CVSS v3.1 specification names); the letters themselves always count.
_CVSS_IMPACT = {"N": ["None"], "L": ["Low"], "H": ["High"]}
VALUE_NAMES = {
    "cti_vsp": {
        "AV": {"N": ["Network"], "A": ["Adjacent Network", "Adjacent"], "L": ["Local"], "P": ["Physical"]},
        "AC": {"L": ["Low"], "H": ["High"]},
        "PR": _CVSS_IMPACT,
        "UI": {"N": ["None"], "R": ["Required"]},
        "S": {"U": ["Unchanged"], "C": ["Changed"]},
        "C": _CVSS_IMPACT,
        "I": _CVSS_IMPACT,
        "A": _CVSS_IMPACT,
    },
    "claudette_tos": {m: {"Y": ["Yes"], "N": ["No"]} for m in CLAUDETTE_METRICS},
}
# Spelled-out slot keys: the metric names of the labeling prompts ("Attack Vector", "arbitration").
METRIC_NAMES = {
    "cti_vsp": importlib.import_module("prompts.cti_vsp.labeling").CVSS_METRIC_NAMES,
    "claudette_tos": importlib.import_module("prompts.claudette_tos.labeling").CLAUDETTE_METRIC_NAMES,
}


def _alternation(spellings) -> str:
    """Longest first, so "None" wins over "N"; inner spaces match any whitespace."""
    return "|".join(re.escape(x).replace(r"\ ", r"\s+") for x in sorted(spellings, key=len, reverse=True))


class LenientParser:
    """The recoverable parser of one domain, see the module docstring."""

    def __init__(self, domain: str, prompt: LabelingPrompt):
        self.domain = domain
        if domain == "toxicity_detection":
            if sorted(prompt.fragments["labels"]) != sorted(TOXICITY_CLASSES):
                raise SystemExit(f"labels {prompt.fragments['labels']} != {TOXICITY_CLASSES}")
            # (?<![\w-]) / (?![\w-]): "non-toxic", "unsafe" and "toxicity" are not a label.
            self.label_re = {c: re.compile(rf"(?<![\w-]){c}(?![\w-])", re.IGNORECASE) for c in TOXICITY_CLASSES}
            return
        self.slot_re, self.value_of = {}, {}
        for slot in SLOTS[domain]:
            names = VALUE_NAMES[domain][slot]
            if set(names) != set(prompt.fragments["fields"][slot]):
                raise SystemExit(f"{domain} {slot}: spelled-out values {sorted(names)} != "
                                 f"labeling values {sorted(prompt.fragments['fields'][slot])}")
            self.value_of[slot] = {" ".join(x.split()).lower(): v for v, xs in names.items() for x in [v, *xs]}
            key = _alternation([slot, METRIC_NAMES[domain][slot]])
            value = _alternation(self.value_of[slot])
            self.slot_re[slot] = re.compile(
                rf"(?<![A-Za-z])(?:{key})(?:\s*\(\s*{re.escape(slot)}\s*\))?\s*[:=]\s*({value})(?![A-Za-z])",
                re.IGNORECASE,
            )

    def parse(self, text: str) -> dict:
        """{slot: value or None}."""
        text = MARKDOWN_RE.sub("", text)
        if self.domain == "toxicity_detection":
            found = [c for c in TOXICITY_CLASSES if self.label_re[c].search(text)]
            return {TOXICITY_SLOT: found[0] if len(found) == 1 else None}
        out = {}
        for slot, regex in self.slot_re.items():
            values = {self.value_of[slot][" ".join(m.group(1).split()).lower()] for m in regex.finditer(text)}
            out[slot] = values.pop() if len(values) == 1 else None
        return out


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
        "macro_f1": sum(per_class[c]["f1"] for c in TOXICITY_CLASSES) / len(TOXICITY_CLASSES),
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
        return {"macro_f1": metrics["macro_f1"], "f1_toxic": metrics["f1_toxic"], "f1_safe": metrics["f1_safe"], **flat}
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


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_fingerprint(run_args) -> tuple[str, dict]:
    """(fingerprint, its components): everything besides the example that shapes an answer.
    The request parameters are built exactly as ApiPredictor.__init__ builds them (checked in
    query_missing), but without needing an API key, so a rerun without new calls can verify too."""
    model_id, provider = API_MODEL_PRESETS[run_args.api_model]
    no_overrides = argparse.Namespace(temperature=None, top_p=None, max_tokens=None, extra_params=None)
    model_params = build_model_params(model_id, provider, dict(API_BASE_PARAMS, max_tokens=MAX_TOKENS), no_overrides)
    model_params["provider"]["allow_fallbacks"] = True
    prompt: LabelingPrompt = run_args.labeling_prompt
    components = {
        "system_prompt_sha256": sha256(prompt.system),
        "user_prompt_prefix": prompt.user_prefix,
        "max_tokens": MAX_TOKENS,
        "model_id": model_id,
        "model_params": model_params,
    }
    return sha256(json.dumps(components, sort_keys=True))[:16], components


def load_raw(path: Path, records: list[dict], api_model: str, fingerprint: str) -> dict[int, dict]:
    """index -> raw call record of an earlier run; refuses a file from another model, test set or
    request fingerprint."""
    done: dict[int, dict] = {}
    if not path.exists():
        return done
    stale: dict[str | None, int] = {}
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
            if rec.get("fingerprint") != fingerprint:
                stale[rec.get("fingerprint")] = stale.get(rec.get("fingerprint"), 0) + 1
                continue
            done[i] = rec
    if stale:
        found = ", ".join(f"{fp}: {n} line(s)" for fp, n in stale.items())
        raise SystemExit(
            f"{path} holds answers of another request setup (fingerprint {found}; current: {fingerprint}). "
            "The system prompt, max_tokens, model id or request parameters changed - move the file away "
            "or use another --output-dir."
        )
    return done


def query_missing(run_args, records: list[dict], raw_path: Path, fingerprint: str) -> dict[int, dict]:
    done = load_raw(raw_path, records, run_args.api_model, fingerprint)
    todo = [i for i in range(len(records)) if i not in done]
    print(f"[{run_args.domain} / {run_args.api_model}] {len(done)} answers found, {len(todo)} to query", flush=True)
    if not todo:
        return done
    predictor = ApiPredictor(run_args, max_tokens=MAX_TOKENS, logprobs=False)
    if predictor.model_params != request_fingerprint(run_args)[1]["model_params"]:
        raise SystemExit("request_fingerprint() no longer builds the request parameters ApiPredictor sends.")
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
                    "fingerprint": fingerprint,
                    "prompt": records[i]["prompt"],
                    "gt": records[i]["gt"],
                    "raw_output": resp["text"],
                    "openrouter": resp["openrouter"],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done[i] = rec
            fh.flush()
            print(f"  [{min(start + batch_size, len(todo))}/{len(todo)}]", flush=True)
    return done


def parse_row(domain: str, exact: ExactParser, lenient: LenientParser, rec: dict, gt_values: dict) -> dict:
    slots = SLOTS[domain]
    text = rec["raw_output"]
    label_line, exact_values = exact.parse(text)
    lenient_values = lenient.parse(text)
    # An exact answer keeps its exact values in the recoverable group (exact is a subset of it).
    recoverable_values = exact_values if exact_values is not None else lenient_values
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
                        "lenient_parser_values": lenient_values},
    }


def run_one(cli, domain: str, model: str, records: list[dict], gt_values: list[dict],
            prompt: LabelingPrompt, metric_classes: dict, data_info: dict) -> None:
    out_dir = cli.output_dir / domain / model
    out_dir.mkdir(parents=True, exist_ok=True)
    run_args = argparse.Namespace(**vars(cli))
    run_args.api_model, run_args.domain, run_args.labeling_prompt = model, domain, prompt
    fingerprint, fingerprint_components = request_fingerprint(run_args)

    raw = query_missing(run_args, records, out_dir / "raw_outputs.jsonl", fingerprint)
    exact, lenient = ExactParser(domain, prompt), LenientParser(domain, prompt)
    rows = [parse_row(domain, exact, lenient, raw[i], gt_values[i]) for i in range(len(records))]
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "benchmark": domain,
        "api_model": model,
        "model_id": fingerprint_components["model_id"],
        "max_tokens": MAX_TOKENS,
        "fingerprint": fingerprint,
        "fingerprint_components": fingerprint_components,
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
            print(f"   {group:<11} macro_f1={m['macro_f1']:.4f}  f1_toxic={m['f1_toxic']:.4f}  f1_safe={m['f1_safe']:.4f}")
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
