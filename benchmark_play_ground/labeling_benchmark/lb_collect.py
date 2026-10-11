"""Collects the labeling-benchmark results into labeling_benchmark/results/runs.csv (one row per run x
source) and a pivot labeling_benchmark/results/summary.md (variant x source per benchmark, mean +- SD
over the seed groups present, n per cell).

Sources:
  lbench       free generation, labeling parser    runs/<run_id>/summary.json (lbench done)
  experiments     experiments/runs/<run_id>/bench.jsonl (experiments bench done; not for api),
                  same metrics via lb_experiments_eval.py
Metric per benchmark (lb_metrics.py): toxicity f1_toxic, claudette weighted_f1, cti_vsp macro_f1;
plus valid_example_rate / valid_slot_rate.

Usage:
    python benchmark_play_ground/labeling_benchmark/lb_collect.py [--bench B] [--arm A] [--setup S] [--seed-set K] [--group G]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from lb_experiments_eval import evaluate_experiments_run
from lb_metrics import PRIMARY_METRIC
from lb_runs import (LB_DIR, add_filter_args, experiments_bench_done, filters_of, lb_state, load_configs,
                     resolve_full, select)

RESULTS_DIR = LB_DIR / "results"
SOURCES = ("lbench", "experiments")
COLUMNS = ["run_id", "bench", "arm", "variant", "setup", "api_model", "seed_set", "group", "n_bb", "n_fg",
           "rouge_threshold", "activation_threshold", "source", "metric", "value", "valid_example_rate",
           "valid_slot_rate", "n_examples", "n_hit_max_new_tokens"]


def variant(spec) -> str:
    parts = [spec.arm]
    for value in (spec.setup, spec.api_model, f"r{spec.n_bb}-{spec.n_fg}" if spec.n_fg else None, spec.seed_set):
        if value:
            parts.append(value)
    if spec.arm != "api" and spec.rouge_threshold is not None:
        parts.append(spec.exp.run_id.rsplit("__", 1)[1])  # threshold tag
    return "/".join(parts)


def result_row(spec, r: dict, source: str, res: dict, extra: dict | None = None) -> dict:
    v = res["validity"]
    return {"run_id": r["run_id"], "bench": spec.bench, "arm": spec.arm, "variant": variant(spec),
            "setup": spec.setup, "api_model": spec.api_model, "seed_set": spec.seed_set, "group": spec.group,
            "n_bb": spec.n_bb, "n_fg": spec.n_fg, "rouge_threshold": r.get("rouge_threshold"),
            "activation_threshold": r.get("activation_threshold"), "source": source, "metric": res["metric"],
            "value": round(res["value"], 6), "valid_example_rate": round(v["valid_example_rate"], 6),
            "valid_slot_rate": round(v["valid_slot_rate"], 6), "n_examples": v["n_examples"], **(extra or {})}


def fmt(values: list[float]) -> str:
    if not values:
        return ""
    sd = f" ± {statistics.stdev(values):.3f}" if len(values) > 1 else ""
    return f"{statistics.mean(values):.3f}{sd} (n={len(values)})"


def write_summary(rows: list[dict], exp_config: dict, path: Path) -> None:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    order: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        cells[(row["bench"], row["variant"], row["source"])].append(row)
        if row["variant"] not in order[row["bench"]]:
            order[row["bench"]].append(row["variant"])
    lines = ["# Labeling benchmark", "",
             "Metric: mean ± SD over the seed groups present (n). valid = mean valid_example_rate. "
             "Sources: lbench = free generation parsed with the labeling parser (an answer it cannot parse "
             "is invalid as a whole); experiments = outputs of the experiments benchmark (constrained for toxicity/claudette). "
             "Invalid predictions count as FN of the gold class, never as FP.", ""]
    head = ["variant", *[f"{s}" for s in SOURCES], *[f"valid {s}" for s in SOURCES]]
    for bench in exp_config["benchmarks"]:
        if bench not in order:
            continue
        lines += [f"## {bench} ({PRIMARY_METRIC[bench]})", "", "| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
        for v in order[bench]:
            vals = [fmt([c["value"] for c in cells.get((bench, v, s), [])]) for s in SOURCES]
            valid = [f"{statistics.mean(c['valid_example_rate'] for c in cells[(bench, v, s)]):.3f}"
                     if cells.get((bench, v, s)) else "" for s in SOURCES]
            lines.append("| " + " | ".join([v, *vals, *valid]) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_filter_args(parser)
    args = parser.parse_args()
    exp_config, lb_config = load_configs()
    rows = []
    n_lb = n_exp = 0
    for spec in select(exp_config, lb_config, args.run_ids, **filters_of(args)):
        r, exp_r = resolve_full(spec, exp_config, lb_config)
        if lb_state(r, exp_r) == "done":
            summary = json.loads(Path(r["summary_json"]).read_text(encoding="utf-8"))
            extra = {"n_hit_max_new_tokens": summary["counts"]["n_hit_max_new_tokens"]}
            rows.append(result_row(spec, r, "lbench", summary["metrics"], extra))
            n_lb += 1
        if experiments_bench_done(exp_r):
            rows.append(result_row(spec, r, "experiments", evaluate_experiments_run(r)))
            n_exp += 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "runs.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})
    write_summary(rows, exp_config, RESULTS_DIR / "summary.md")
    print(f"{n_lb} run(s) with lbench done, {n_exp} with experiments bench done -> {csv_path}, "
          f"{RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
