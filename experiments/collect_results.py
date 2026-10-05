"""Collects one row per run whose bench stage is done into experiments/results/runs.csv, and a pivot
(arm variant x benchmark, mean +- SD over the seed groups present, n per cell) into
experiments/results/summary.md.

Columns (empty where a value does not apply or is not available):
  identity     run_id, bench, arm, setup, api_model, seed_set, group, n_bb, n_fg
  metrics      bench.summary.json - toxicity: auprc_toxic, macro_f1; claudette_tos: weighted_auprc_8,
               macro_auprc_8; cti_vsp: macro_f1, mad. api runs: n_no_logprobs (examples scored from the
               hard label because the response had no usable logprobs; cti_vsp parses free text and
               uses no logprobs), n_unparsable_pred.
  data         FT arms, training TSV (gold: GOLD_TSV, synthetic: LABEL_TSV): n_train, label_dist
               (toxicity: toxic rate; claudette: Y rate per slot; cti: mode per metric), TV to the test
               set via helper_scripts/distribution_analysis: tv_marginal (mean over slots), tv_slots,
               tv_joint (claudette, cti_vsp)
  generation   gen/label logs: n_accepted, n_rejected, acceptance_rate = accepted/(accepted+rejected),
               n_rouge_duplicate, n_feature_inactive, n_discarded (hybrid: both phases), n_majority_fallback,
               gen_temperature, gen_top_p (logged model_params), gen/label prompt + completion tokens and
               cost_usd (summed usage.cost of the logged calls)
  compute      wall_<stage> (run_meta.json), sae_gpu_seconds (generation log)
  provenance   git_<stage> (run_meta.json), ft_seed, gen_fingerprint (generation log)
  fac          fac_relevant (fraction of the task-relevant SAE features covered), fac_test (fraction of
               the test-set features covered) from experiments/runs/<run_id>/fac.json - only if its
               input_sha256 matches the current training TSV

Usage:
    python experiments/collect_results.py [--bench B] [--arm A] [--setup S] [--seed-set K] [--group G]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from runs import ARMS, EXPERIMENTS_DIR, PROJECT_DIR, file_sha256, load_config, resolve
from stages import bench_state
from status import add_filter_args, filters_of, run_order_key, select_specs

RESULTS_DIR = EXPERIMENTS_DIR / "results"
METRICS = {
    "toxicity_detection": ("auprc_toxic", "macro_f1"),
    "claudette_tos": ("weighted_auprc_8", "macro_auprc_8"),
    "cti_vsp": ("macro_f1", "mad"),
}
PRIMARY = {bench: metrics[0] for bench, metrics in METRICS.items()}
STAGES_META = ("gen", "label_build", "ft", "bench")
COLUMNS = [
    "run_id", "bench", "arm", "setup", "api_model", "seed_set", "group", "n_bb", "n_fg",
    "auprc_toxic", "macro_f1", "weighted_auprc_8", "macro_auprc_8", "mad", "n_no_logprobs", "n_unparsable_pred",
    "n_train", "label_dist", "tv_marginal", "tv_slots", "tv_joint",
    "n_accepted", "n_rejected", "acceptance_rate", "n_rouge_duplicate", "n_feature_inactive", "n_discarded",
    "n_majority_fallback", "gen_temperature", "gen_top_p",
    "gen_prompt_tokens", "gen_completion_tokens", "gen_cost_usd",
    "label_prompt_tokens", "label_completion_tokens", "label_cost_usd",
    *[f"wall_{s}" for s in STAGES_META], "sae_gpu_seconds",
    *[f"git_{s}" for s in STAGES_META], "ft_seed", "gen_fingerprint",
    "fac_relevant", "fac_test",
]


def load_distribution_analysis():
    path = PROJECT_DIR / "helper_scripts" / "distribution_analysis" / "distribution_analysis.py"
    spec = importlib.util.spec_from_file_location("distribution_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    specs = {s["domain_dir"]: s for s in module.build_domain_specs().values()}
    return module, specs


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
    row = {m: s.get(m) for m in METRICS[r["bench"]]}
    if r["arm"] == "api":
        row["n_unparsable_pred"] = s.get("n_unparsable_pred", s.get("n_excluded_unparseable"))
        if r["bench"] == "cti_vsp":
            row["n_no_logprobs"] = "n/a (free text)"
        else:
            row["n_no_logprobs"] = n_no_logprobs(r)
    return row


def n_no_logprobs(r: dict) -> int | None:
    """Examples with at least one score taken from the hard label (no usable logprobs)."""
    n = 0
    with open(r["bench_jsonl"], encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if r["bench"] == "toxicity_detection":
                n += ex.get("log_odds") is None
            else:
                n += any(v is None for v in (ex.get("slot_log_odds") or {}).values())
    return n


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


def generation_columns(r: dict) -> dict:
    if r["arm"] not in ("bb", "fg", "hybrid"):
        return {}
    main = last_run(r["gen_log_json"])
    if not main:
        return {}
    phases = [last_run(r["gen_bb_log_json"]), last_run(r["gen_fg_log_json"])] if r["arm"] == "hybrid" else [main]
    phases = [p for p in phases if p]

    def total(key):
        return sum(p.get(key) or 0 for p in phases)

    accepted, rejected = total("n_accepted"), total("n_rejected")
    params = main.get("model_params") or {}
    label = last_run(r["label_log_json"])
    sae = main.get("sae_total") or {}
    costs = [summed_cost(p) for p in phases]
    return {
        "n_accepted": accepted, "n_rejected": rejected,
        "acceptance_rate": round(accepted / (accepted + rejected), 4) if accepted + rejected else None,
        "n_rouge_duplicate": total("n_rouge_duplicate"), "n_feature_inactive": total("n_feature_inactive"),
        "n_discarded": total("n_discarded"),
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
    if row.get("api_model"):
        parts.append(row["api_model"])
    if row.get("n_fg"):
        parts.append(f"r{row['n_bb']}-{row['n_fg']}")
    if row.get("seed_set"):
        parts.append(row["seed_set"])
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
             + ", ".join(f"{b}: `{PRIMARY[b]}`" for b in benches), "",
             "| arm variant | " + " | ".join(benches) + " |",
             "|---|" + "---|" * len(benches)]
    for v in order:
        lines.append(f"| {v} | " + " | ".join(fmt(cells.get((v, b), [])) for b in benches) + " |")
    lines += ["", f"{len(rows)} run(s) with bench done; per-run values in runs.csv."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_filter_args(parser)
    args = parser.parse_args()

    config = load_config()
    da, da_specs = load_distribution_analysis()
    ref_cache: dict = {}
    specs = sorted(select_specs(config, args.run_ids, **filters_of(args)), key=lambda s: run_order_key(config, s))
    rows = []
    for spec in specs:
        r = resolve(spec, config)
        if bench_state(r)[0] != "done":
            continue
        row = {"run_id": r["run_id"], "bench": spec.bench, "arm": spec.arm, "setup": spec.setup,
               "api_model": spec.api_model, "seed_set": spec.seed_set, "group": spec.group,
               "n_bb": spec.n_bb, "n_fg": spec.n_fg}
        for part in (metric_columns(r), data_columns(r, da, da_specs, ref_cache), generation_columns(r),
                     meta_columns(r), fac_columns(r)):
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
