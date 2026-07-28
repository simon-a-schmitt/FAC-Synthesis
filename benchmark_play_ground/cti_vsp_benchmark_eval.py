#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.evaluator import CVSS_METRICS, evaluate_cti_vsp_predictions


def load_cti_vsp_metric_classes(path: str):
    """Same logic as data_loader.load_cti_vsp_metric_classes, inlined here to
    avoid pulling in the pandas dependency (used by data_loader for TSV loading,
    which this eval script doesn't need)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {metric: list(classes.keys()) for metric, classes in data["metric_distribution"].items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--predictions-jsonl",
        default=str(Path(__file__).parent / "cti_vsp_bench_plain_archive.jsonl"),
        help="JSONL file with per-example predictions (as produced by run_cti_vsp_benchmark.py)",
    )
    p.add_argument(
        "--metric-distribution-json",
        default=str(ROOT_DIR / "benchmarks" / "cti_vsp" / "cti_vsp_gt_metric_distribution.json"),
        help="JSON file listing the ground-truth classes per CVSS metric (for macro-F1 labels)",
    )
    return p.parse_args()


def load_predictions(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    args = parse_args()

    records = load_predictions(args.predictions_jsonl)
    metric_classes = load_cti_vsp_metric_classes(args.metric_distribution_json)

    summary = evaluate_cti_vsp_predictions(records, metric_classes)

    print("Summary:")
    print(f"  total: {summary['total']}")
    print(f"  exact_match: {summary['exact_match']}")
    print(f"  exact_match_accuracy: {summary['exact_match_accuracy']:.4f}")
    print(f"  overall_macro_f1: {summary['overall_macro_f1']:.4f}")
    print("  per_metric_macro_f1:")
    for metric in CVSS_METRICS:
        print(f"    {metric}: {summary['per_metric_macro_f1'][metric]:.4f}")


if __name__ == "__main__":
    main()
