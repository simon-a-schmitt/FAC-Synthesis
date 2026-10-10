"""The labeling-benchmark metrics (lb_metrics.py) on the outputs of the experiments benchmark
(experiments/runs/<run_id>/bench.jsonl, constrained decoding for toxicity/claudette, free generation
for cti_vsp) - read only, nothing under experiments/ is written.

Predictions as the experiments benchmark scripts determined them:
  toxicity_detection  "predicted" (constrained, always valid)
  claudette_tos       "predicted_metrics" (constrained, always valid)
  cti_vsp             "predicted_metrics"; a None slot is invalid -> FN of its gold class (the
                      experiments script itself excludes such examples; here they stay in, as in
                      the labeling benchmark)
Gold: the test TSV parsed with the exact parser; every line's gt must equal the TSV's.

Usage:
    python benchmark_play_ground/labeling_benchmark/lb_experiments_eval.py <run_id> [<run_id> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lb_metrics import evaluate
from lb_parsers import SLOTS, TOXICITY_SLOT, gold_values
from lb_runs import experiments_bench_done, load_configs, parse_run_id, resolve_full

_TEST_CACHE: dict[tuple[str, int], tuple[list[dict], list[dict]]] = {}


def load_test(domain: str, test_tsv: str, max_prompts: int) -> tuple[list[dict], list[dict]]:
    """(records, gold values) of the test set, cached per (test_tsv, max_prompts)."""
    key = (test_tsv, max_prompts)
    if key not in _TEST_CACHE:
        from benchmark_play_ground.data_loader import load_claudette_tsv, load_cti_vsp_tsv, load_toxicity_tsv
        loader = {"toxicity_detection": load_toxicity_tsv, "claudette_tos": load_claudette_tsv,
                  "cti_vsp": load_cti_vsp_tsv}[domain]
        records = loader(test_tsv)
        if max_prompts > 0:
            records = records[:max_prompts]
        _TEST_CACHE[key] = (records, gold_values(domain, records, test_tsv))
    return _TEST_CACHE[key]


def predicted_values(domain: str, line: dict) -> dict:
    if domain == "toxicity_detection":
        return {TOXICITY_SLOT: line.get("predicted")}
    metrics = line.get("predicted_metrics") or {}
    return {s: metrics.get(s) for s in SLOTS[domain]}


def evaluate_experiments_run(r: dict) -> dict:
    """Metrics of the experiments output of run r (lb_runs.resolve); raises if it does not match the test set."""
    domain = r["bench"]
    records, gold = load_test(domain, r["test_tsv"], r["max_prompts"])
    lines = [json.loads(l) for l in Path(r["exp_bench_jsonl"]).read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) != len(records):
        raise ValueError(f"{r['exp_bench_jsonl']}: {len(lines)} lines, test set has {len(records)}.")
    pred = []
    for i, (line, rec) in enumerate(zip(lines, records)):
        if line.get("index", i) != i or line.get("gt") != rec["gt"]:
            raise ValueError(f"{r['exp_bench_jsonl']}: line {i} does not belong to example {i} of {r['test_tsv']}.")
        pred.append(predicted_values(domain, line))
    return evaluate(domain, gold, pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    exp_config, lb_config = load_configs()
    for run_id in args.run_ids:
        r, exp_r = resolve_full(parse_run_id(run_id), exp_config, lb_config)
        if r["arm"] == "api":
            print(f"{run_id}: api runs have no experiments output")
            continue
        if not experiments_bench_done(exp_r):
            print(f"{run_id}: experiments bench not done")
            continue
        res = evaluate_experiments_run(r)
        v = res["validity"]
        print(f"{run_id}: {res['metric']}={res['value']:.4f}  valid_examples={v['valid_example_rate']:.4f}  "
              f"valid_slots={v['valid_slot_rate']:.4f}  (n={v['n_examples']})")


if __name__ == "__main__":
    main()
