"""Collects one row per run whose bench stage is done into experiments/results/runs.csv, and a pivot
(arm variant x benchmark, mean +- SD over the seed groups present, n per cell) into
experiments/results/summary.md.

Columns (empty where a value does not apply or is not available):
  identity     run_id, bench, arm, setup, seed_set, group, n_bb, n_fg, rouge_threshold,
               activation_threshold (gen arms: the thresholds the run used, see runs.py threshold tag)
  metrics      bench.summary.json - toxicity: auprc_toxic, macro_f1; claudette_tos: weighted_auprc_8,
               macro_auprc_8; cti_vsp: mad, macro_f1 (the first metric is the
               primary one in summary.md).
  data         FT arms, training TSV (gold: GOLD_TSV, synthetic: LABEL_TSV): n_train, label_dist
               (toxicity: toxic rate; claudette: Y rate per slot; cti: mode per metric), TV to the test
               set via helper_scripts/distribution_analysis: tv_marginal (mean over slots), tv_slots,
               tv_joint (claudette, cti_vsp)
  length       FT arms, text column of the same training TSV via helper_scripts/length_analysis:
               len_chars_mean, len_words_mean, len_words_p95 (whitespace split), len_tokens_mean,
               len_tokens_p50, len_tokens_p95, len_tokens_max (base-model tokenizer, raw text without chat
               template or special tokens; empty if transformers is not importable),
               len_words_ratio_test, len_tokens_ratio_test (training mean / test-set mean),
               distance of the token-length distribution to the test set's: len_tokens_tv_test (TV
               distance over the test set's decile bins plus one under-/overflow bin each, in [0, 1]),
               len_tokens_w1_test (Wasserstein-1 / earth mover's distance in tokens, divided by the test
               mean token length)
  generation   gen/label logs: n_accepted, n_rejected, n_rouge_duplicate, n_feature_inactive and the rates
               acceptance_rate, rejection_rate, rouge_rate, inactive_rate (each / checked candidates =
               accepted + rejected; a candidate is checked for feature activation first, then ROUGE-L, so
               the three reasons partition it; discarded candidates were never checked) - unprefixed: whole
               run (hybrid: both phases summed), bb_* / fg_*: per phase (bb arm: bb_, fg arm: fg_, hybrid:
               both); n_discarded, n_majority_fallback,
               gen_temperature, gen_top_p (logged model_params), gen/label prompt + completion tokens and
               cost_usd (summed usage.cost of the logged calls)
  compute      wall_<stage> (run_meta.json), sae_gpu_seconds (generation log)
  provenance   git_<stage> (run_meta.json), ft_seed, gen_fingerprint (generation log)
  fac          fac_relevant (fraction of the task-relevant SAE features covered), fac_test (fraction of
               the test-set features covered) from experiments/runs/<run_id>/fac.json - only if its
               input_sha256 matches the current training TSV

Runs: those of the current config (filtered); with --all-on-disk also every other run under
experiments/runs/ (e.g. other threshold variants), with the same filters.

Usage:
    python experiments/collect_results.py [--bench B] [--arm A] [--setup S] [--seed-set K] [--group G] [--all-on-disk]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from runs import ARMS, EXPERIMENTS_DIR, PROJECT_DIR, RUNS_DIR, file_sha256, load_config, parse_run_id, resolve
from stages import bench_state
from status import add_filter_args, filters_of, run_order_key, select_specs

RESULTS_DIR = EXPERIMENTS_DIR / "results"
METRICS = {
    "toxicity_detection": ("auprc_toxic", "macro_f1"),
    "claudette_tos": ("weighted_auprc_8", "macro_auprc_8"),
    "cti_vsp": ("mad", "macro_f1"),
}
PRIMARY = {bench: metrics[0] for bench, metrics in METRICS.items()}  # first metric per benchmark
LOWER_IS_BETTER = {"mad"}
STAGES_META = ("gen", "label_build", "ft", "bench")
GEN_PHASES = ("bb", "fg")
GEN_COUNTS = ("n_accepted", "n_rejected", "n_rouge_duplicate", "n_feature_inactive")
GEN_RATES = ("acceptance_rate", "rejection_rate", "rouge_rate", "inactive_rate")
COLUMNS = [
    "run_id", "bench", "arm", "setup", "seed_set", "group", "n_bb", "n_fg",
    "rouge_threshold", "activation_threshold",
    "auprc_toxic", "macro_f1", "weighted_auprc_8", "macro_auprc_8", "mad",
    "n_train", "label_dist", "tv_marginal", "tv_slots", "tv_joint",
    *GEN_COUNTS, *GEN_RATES, *[f"{phase}_{k}" for phase in GEN_PHASES for k in (*GEN_COUNTS, *GEN_RATES)],
    "n_discarded",
    "n_majority_fallback", "gen_temperature", "gen_top_p",
    "gen_prompt_tokens", "gen_completion_tokens", "gen_cost_usd",
    "label_prompt_tokens", "label_completion_tokens", "label_cost_usd",
    *[f"wall_{s}" for s in STAGES_META], "sae_gpu_seconds",
    *[f"git_{s}" for s in STAGES_META], "ft_seed", "gen_fingerprint",
    "fac_relevant", "fac_test",
    "len_chars_mean", "len_words_mean", "len_words_p95",
    "len_tokens_mean", "len_tokens_p50", "len_tokens_p95", "len_tokens_max",
    "len_words_ratio_test", "len_tokens_ratio_test", "len_tokens_tv_test", "len_tokens_w1_test",
]


def load_distribution_analysis():
    path = PROJECT_DIR / "helper_scripts" / "distribution_analysis" / "distribution_analysis.py"
    spec = importlib.util.spec_from_file_location("distribution_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    specs = {s["domain_dir"]: s for s in module.build_domain_specs().values()}
    return module, specs


def load_length_analysis():
    path = PROJECT_DIR / "helper_scripts" / "length_analysis" / "length_analysis.py"
    spec = importlib.util.spec_from_file_location("length_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizers:
    """Base-model tokenizers, loaded on first use; None (token columns left empty) without transformers."""

    def __init__(self):
        self.cache: dict = {}
        self.available = True

    def get(self, model_path):
        if not self.available:
            return None
        if model_path not in self.cache:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                print("warning: transformers not importable - len_tokens_* columns left empty "
                      "(run in the project env, e.g. after sourcing start_llama.sh)")
                self.available = False
                return None
            self.cache[model_path] = AutoTokenizer.from_pretrained(str(model_path))
        return self.cache[model_path]


def read_json(path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def last_run(log_path) -> dict | None:
    log = read_json(log_path)
    return (log.get("runs") or [None])[-1] if log else None


def summed_cost(run: dict | None) -> float | None:
    if not run or not run.get("calls"):
        return None
    return round(sum(((c.get("openrouter_response") or {}).get("usage") or {}).get("cost") or 0.0
                     for c in run["calls"]), 6)


# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

def metric_columns(r: dict) -> dict:
    s = read_json(Path(r["bench_jsonl"]).with_suffix(".summary.json")) or {}
    return {m: s.get(m) for m in METRICS[r["bench"]]}


def data_columns(r: dict, da, da_specs: dict, ref_cache: dict) -> dict:
    tsv = Path(r["gold_tsv"] if r["arm"] == "gold" else r.get("label_tsv", ""))
    if r["arm"] not in ("gold", "bb", "fg", "hybrid") or not tsv.is_file():
        return {}
    spec = da_specs[r["bench"]]
    result = da.analyze(tsv, spec)
    if r["bench"] not in ref_cache:
        ref_cache[r["bench"]] = da.analyze(Path(r["test_tsv"]), spec)
    tv = da.compute_tv(result, ref_cache[r["bench"]], spec)
    dist = result["distribution"]
    if spec["kind"] == "binary":
        return {"n_train": result["n_total"], "label_dist": f"{spec['positive']}={dist['positive_rate']:.3f}",
                "tv_marginal": round(tv["tv"], 4)}
    if spec["positive"] is not None:  # claudette: Y rate per slot
        label_dist = " ".join(f"{f}={m['positive_rate']:.3f}" for f, m in dist["marginal"].items())
    else:  # cti_vsp: most frequent value per metric
        label_dist = " ".join(f"{f}={max(m['proportions'], key=m['proportions'].get)}"
                              f"({max(m['proportions'].values()):.2f})" for f, m in dist["marginal"].items())
    return {"n_train": result["n_total"], "label_dist": label_dist,
            "tv_marginal": round(tv["marginal"]["mean"], 4),
            "tv_slots": " ".join(f"{f}={v:.3f}" for f, v in tv["marginal"]["per_slot"].items()),
            "tv_joint": round(tv["joint"], 4)}


def length_columns(r: dict, la, tokenizers: Tokenizers, ref_cache: dict) -> dict:
    tsv = Path(r["gold_tsv"] if r["arm"] == "gold" else r.get("label_tsv", ""))
    if r["arm"] not in ("gold", "bb", "fg", "hybrid") or not tsv.is_file():
        return {}
    tok = tokenizers.get(r["base_model"])
    lengths = la.measure_lengths(la.read_texts(tsv), tok)
    result = la.analyze_lengths(lengths)
    key = (r["bench"], r["base_model"] if tok else None)
    if key not in ref_cache:
        ref_lengths = la.measure_lengths(la.read_texts(Path(r["test_tsv"])), tok)
        ref_cache[key] = (la.analyze_lengths(ref_lengths), ref_lengths)
    ref, ref_lengths = ref_cache[key]
    row = {"len_chars_mean": round(result["chars"]["mean"], 1),
           "len_words_mean": round(result["words"]["mean"], 1),
           "len_words_p95": round(result["words"]["p95"], 1),
           "len_words_ratio_test": round(result["words"]["mean"] / ref["words"]["mean"], 3)}
    if "tokens" in result:
        t = result["tokens"]
        row.update(len_tokens_mean=round(t["mean"], 1), len_tokens_p50=round(t["p50"], 1),
                   len_tokens_p95=round(t["p95"], 1), len_tokens_max=int(t["max"]),
                   len_tokens_ratio_test=round(t["mean"] / ref["tokens"]["mean"], 3))
        d = la.distance_to_reference(lengths["tokens"], ref_lengths["tokens"])
        row.update(len_tokens_tv_test=round(d["tv"], 4), len_tokens_w1_test=round(d["w1_norm"], 4))
    return row


def acceptance_stats(runs: list[dict], prefix: str = "") -> dict:
    """GEN_COUNTS summed over the given generation log runs, plus GEN_RATES over the checked candidates."""
    counts = {k: sum(x.get(k) or 0 for x in runs) for k in GEN_COUNTS}
    checked = counts["n_accepted"] + counts["n_rejected"]

    def rate(n):
        return round(n / checked, 4) if checked else None

    stats = {**counts, "acceptance_rate": rate(counts["n_accepted"]), "rejection_rate": rate(counts["n_rejected"]),
             "rouge_rate": rate(counts["n_rouge_duplicate"]), "inactive_rate": rate(counts["n_feature_inactive"])}
    return {f"{prefix}{k}": v for k, v in stats.items()}


def generation_columns(r: dict) -> dict:
    if r["arm"] not in ("bb", "fg", "hybrid"):
        return {}
    main = last_run(r["gen_log_json"])
    if not main:
        return {}
    if r["arm"] == "hybrid":
        by_phase = {tag: last_run(r[f"gen_{tag}_log_json"]) for tag in GEN_PHASES}
    else:
        by_phase = {r["arm"]: main}
    by_phase = {tag: p for tag, p in by_phase.items() if p}
    phases = list(by_phase.values())

    def total(key):
        return sum(p.get(key) or 0 for p in phases)

    stats = acceptance_stats(phases)
    for tag, p in by_phase.items():
        stats.update(acceptance_stats([p], f"{tag}_"))
    params = main.get("model_params") or {}
    label = last_run(r["label_log_json"])
    sae = main.get("sae_total") or {}
    costs = [summed_cost(p) for p in phases]
    return {
        **stats, "n_discarded": total("n_discarded"),
        "gen_temperature": params.get("temperature"), "gen_top_p": params.get("top_p"),
        "gen_prompt_tokens": main.get("prompt_tokens"), "gen_completion_tokens": main.get("completion_tokens"),
        "gen_cost_usd": round(sum(c for c in costs if c is not None), 6) if any(c is not None for c in costs) else None,
        "label_prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in (read_json(r["label_log_json"]) or {}).get("runs", [])) or None,
        "label_completion_tokens": sum(x.get("completion_tokens") or 0 for x in (read_json(r["label_log_json"]) or {}).get("runs", [])) or None,
        "label_cost_usd": round(sum(summed_cost(x) or 0 for x in (read_json(r["label_log_json"]) or {}).get("runs", [])), 6) if label else None,
        "n_majority_fallback": sum(x.get("n_majority_fallback") or 0 for x in (read_json(r["label_log_json"]) or {}).get("runs", [])) if label else None,
        "sae_gpu_seconds": sae.get("gpu_seconds"),
        "gen_fingerprint": main.get("gen_fingerprint"),
    }


def meta_columns(r: dict) -> dict:
    meta = read_json(r["run_meta_json"]) or {}
    row = {}
    for stage, rec in (meta.get("stages") or {}).items():
        if stage in STAGES_META:
            row[f"wall_{stage}"] = rec.get("wall_seconds")
            row[f"git_{stage}"] = (rec.get("git_commit") or "")[:10] + ("+dirty" if rec.get("git_dirty") else "")
    ft_log = read_json(r["ft_log_json"]) if "ft_log_json" in r else None
    if ft_log:
        row["ft_seed"] = ft_log.get("ft_seed")
    return row


def fac_columns(r: dict) -> dict:
    fac = read_json(Path(r["run_dir"]) / "fac.json")
    if not fac:
        return {}
    tsv = r["gold_tsv"] if r["arm"] == "gold" else r.get("label_tsv")
    if fac.get("input_sha256") != file_sha256(tsv):
        return {"fac_relevant": "stale", "fac_test": "stale"}
    key = fac["threshold"]
    m = fac["metrics"]
    return {"fac_relevant": round(m[f"feature_coverage_at_t{key}"]["fraction"], 4),
            "fac_test": round(m[f"feature_coverage_test_at_t{key}"]["fraction"], 4)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def variant(row: dict) -> str:
    parts = [row["arm"]]
    if row.get("setup"):
        parts.append(row["setup"])
    if row.get("n_fg"):
        parts.append(f"r{row['n_bb']}-{row['n_fg']}")
    if row.get("seed_set"):
        parts.append(row["seed_set"])
    if row.get("threshold_tag"):
        parts.append(row["threshold_tag"])
    return "/".join(parts)


def write_summary(rows: list[dict], config: dict, path: Path) -> None:
    benches = list(config["benchmarks"])
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        v = variant(row)
        if v not in order:
            order.append(v)
        value = row.get(PRIMARY[row["bench"]])
        if isinstance(value, (int, float)):
            cells[(v, row["bench"])].append(value)

    def fmt(values: list[float]) -> str:
        if not values:
            return ""
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{statistics.mean(values):.3f} ± {sd:.3f} (n={len(values)})"

    lines = ["# Results", "",
             "Primary metric per benchmark (mean ± SD over the seed groups present, n = runs): "
             + ", ".join(f"{b}: `{PRIMARY[b]}`" + (" (lower is better)" if PRIMARY[b] in LOWER_IS_BETTER else "")
                        for b in benches), "",
             "| arm variant | " + " | ".join(benches) + " |",
             "|---|" + "---|" * len(benches)]
    for v in order:
        lines.append(f"| {v} | " + " | ".join(fmt(cells.get((v, b), [])) for b in benches) + " |")
    lines += ["", f"{len(rows)} run(s) with bench done; per-run values in runs.csv."]
    lines += acceptance_section(rows, benches)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# (table title, arm, phases as (label, column prefix), show the per-reason columns)
ACCEPTANCE_TABLES = (
    ("Black-box (bb)", "bb", (("", "bb_"),), False),
    ("Feature-guided (fg)", "fg", (("", "fg_"),), True),
    ("Hybrid", "hybrid", (("bb", "bb_"), ("fg", "fg_"), ("total", "")), True),
)


def acceptance_section(rows: list[dict], benches: list[str]) -> list[str]:
    """One table per generation arm: candidate counts (mean over the runs of a cell) and rates
    (mean +- SD over the runs; each / checked candidates = accepted + rejected)."""

    def values(group: list[dict], key: str) -> list[float]:
        return [row[key] for row in group if isinstance(row.get(key), (int, float))]

    def count(group, key):
        v = values(group, key)
        return f"{statistics.mean(v):.1f}".removesuffix(".0") if v else ""

    def rate(group, key):
        v = values(group, key)
        if not v:
            return ""
        sd = f" ± {100 * statistics.stdev(v):.1f}" if len(v) > 1 else ""
        return f"{100 * statistics.mean(v):.1f}{sd}"

    lines = ["", "## Generation acceptance", "",
             "Per arm variant x benchmark: candidate counts (mean over the n runs) and rates in % (mean ± SD over "
             "the runs) of the checked candidates (accepted + rejected; candidates discarded after the target was "
             "reached are not counted). Rejection reasons: activation = target feature not active (checked first), "
             "ROUGE-L = too similar to the pool (ROUGE-L > rouge_threshold)."]
    for title, arm, phases, reasons in ACCEPTANCE_TABLES:
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            if row["arm"] == arm and row.get("n_accepted") is not None:
                groups[(variant(row), row["bench"])].append(row)
        if not groups:
            continue
        keys = sorted(groups, key=lambda vb: (benches.index(vb[1]), list(dict.fromkeys(v for v, _ in groups)).index(vb[0])))
        if reasons:
            head = ["accepted", "rej. ROUGE-L", "rej. activation", "rejected", "acceptance %", "ROUGE-L %",
                    "activation %", "rejection %"]
            cols = [(count, "n_accepted"), (count, "n_rouge_duplicate"), (count, "n_feature_inactive"),
                    (count, "n_rejected"), (rate, "acceptance_rate"), (rate, "rouge_rate"),
                    (rate, "inactive_rate"), (rate, "rejection_rate")]
        else:
            head = ["accepted", "rejected", "acceptance %", "rejection %"]
            cols = [(count, "n_accepted"), (count, "n_rejected"), (rate, "acceptance_rate"), (rate, "rejection_rate")]
        has_phase = len(phases) > 1
        head = ["arm variant", "benchmark", *(["phase"] if has_phase else []), "n", *head]
        lines += ["", f"### {title}", "", "| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
        for v, bench in keys:
            group = groups[(v, bench)]
            for label, prefix in phases:
                cells = [v, bench, *([label] if has_phase else []), str(len(group)),
                         *(fmt(group, prefix + key) for fmt, key in cols)]
                lines.append("| " + " | ".join(cells) + " |")
    return lines


def on_disk_specs(config: dict, filters: dict) -> list:
    """RunSpecs of the run directories under RUNS_DIR that resolve under the current config (other
    threshold variants etc.); names that are no run_id are skipped."""
    specs = []
    for path in sorted(RUNS_DIR.iterdir()):
        try:
            spec = parse_run_id(path.name)
            resolve(spec, config)
        except (ValueError, KeyError):
            continue
        if path.is_dir() and all(v is None or getattr(spec, k) == v for k, v in filters.items()):
            specs.append(spec)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_filter_args(parser)
    parser.add_argument("--all-on-disk", action="store_true",
                        help="Also every run under experiments/runs/ that the current config does not expand to.")
    args = parser.parse_args()

    config = load_config()
    da, da_specs = load_distribution_analysis()
    la, tokenizers = load_length_analysis(), Tokenizers()
    ref_cache: dict = {}
    len_ref_cache: dict = {}
    specs = select_specs(config, args.run_ids, **filters_of(args))
    if args.all_on_disk:
        specs = list(dict.fromkeys(specs + on_disk_specs(config, filters_of(args))))
    specs = sorted(specs, key=lambda s: run_order_key(config, s))
    rows = []
    for spec in specs:
        r = resolve(spec, config)
        if bench_state(r)[0] != "done":
            continue
        row = {"run_id": r["run_id"], "bench": spec.bench, "arm": spec.arm, "setup": spec.setup,
               "seed_set": spec.seed_set, "group": spec.group,
               "n_bb": spec.n_bb, "n_fg": spec.n_fg, "rouge_threshold": r.get("rouge_threshold"),
               "activation_threshold": r.get("activation_threshold"), "threshold_tag": r.get("threshold_tag")}
        for part in (metric_columns(r), data_columns(r, da, da_specs, ref_cache), generation_columns(r),
                     meta_columns(r), fac_columns(r), length_columns(r, la, tokenizers, len_ref_cache)):
            row.update(part)
        rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "runs.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})
    write_summary(rows, config, RESULTS_DIR / "summary.md")
    print(f"{len(rows)} run(s) with bench done -> {csv_path}, {RESULTS_DIR / 'summary.md'}")
    print("arms:", dict(Counter(row["arm"] for row in rows)))


if __name__ == "__main__":
    main()
