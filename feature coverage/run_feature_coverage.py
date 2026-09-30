"""Feature coverage of a dataset (--input-tsv), per threshold t in --thresholds (feature activity
measured exactly as in coverage_common.py / run_feature_coverage_reference.py):

  features_active_at_t<t>       number of SAE features active on the input TSV
  feature_coverage_at_t<t>      how many of the task-relevant features (label in --feature-labels in
                                data_synthesis/data/feature_scores/<path>_feature_relevance_scores.jsonl)
                                are active on the input TSV - absolute and as fraction, plus per label
  feature_coverage_test_at_t<t> how many of the features active on the reference (test) data at t
                                (run_feature_coverage_reference.py output) are also active on the input
                                TSV - absolute and as fraction

Output: feature coverage/<path>/output/<input-tsv-stem>_feature_coverage.json (metrics, metadata and
active feature ids per threshold) and ..._feature_coverage.tsv (one row per threshold).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import coverage_common as cc

DEFAULT_FEATURE_LABELS = ["Yes", "Probably", "Maybe"]


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def load_relevant_features(feature_scores: Path, labels: list[str]) -> dict[int, str]:
    """{feature_id: label} for every feature whose label is in `labels`."""
    if not feature_scores.is_file():
        raise SystemExit(f"Feature scores file not found: {feature_scores}")
    relevant = {}
    with open(feature_scores, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("label") in labels:
                relevant[int(row["feature_id"])] = row["label"]
    if not relevant:
        raise SystemExit(f"No features with a label in {labels!r} found in {feature_scores}.")
    return relevant


def find_reference_json(path: str) -> Path:
    domain_dir = cc.domain_output_dir(path)
    matches = sorted(domain_dir.glob("*_reference_active_features.json"))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one *_reference_active_features.json in {domain_dir}, found {len(matches)}"
            f"{': ' + ', '.join(m.name for m in matches) if matches else ''}. Pass --reference-json explicitly "
            "(or run run_feature_coverage_reference.py first)."
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature coverage of a TSV relative to the task-relevant features and to the reference (test) features."
    )
    cc.add_common_args(parser, multiple_inputs=True)
    parser.add_argument(
        "--reference-json", type=Path, default=None,
        help="Output of run_feature_coverage_reference.py (default: the only *_reference_active_features.json "
        "in feature coverage/<path>/).",
    )
    parser.add_argument(
        "--feature-scores", type=Path, default=None,
        help="Feature relevance JSONL (default: data_synthesis/data/feature_scores/<path>_feature_relevance_scores.jsonl).",
    )
    parser.add_argument(
        "--feature-labels", type=str, nargs="+", default=DEFAULT_FEATURE_LABELS,
        choices=["Yes", "Probably", "Maybe", "No"],
        help="Task-relevant feature labels (default: %(default)s).",
    )
    args = parser.parse_args()

    # Validate all inputs before loading the model.
    feature_scores = args.feature_scores or cc.FEATURE_SCORES_DIR / f"{args.path}_feature_relevance_scores.jsonl"
    relevant = load_relevant_features(feature_scores, args.feature_labels)
    reference_path = args.reference_json or find_reference_json(args.path)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("path") != args.path:
        raise SystemExit(f"{reference_path} was computed for path {reference.get('path')!r}, not {args.path!r}.")
    missing = [cc.threshold_key(t) for t in args.thresholds if cc.threshold_key(t) not in reference["thresholds"]]
    if missing:
        raise SystemExit(
            f"Threshold(s) {missing} not in {reference_path} (available: {sorted(reference['thresholds'])}). "
            "Re-run run_feature_coverage_reference.py with these thresholds."
        )
    print(f"Relevant features: {len(relevant)} (labels {args.feature_labels}) from {feature_scores}")
    print(f"Reference: {reference_path} ({reference['n_samples']} samples of {reference['input_tsv']})")
    for input_tsv in args.input_tsv:
        if not input_tsv.is_file():
            raise SystemExit(f"Input TSV not found: {input_tsv}")
    # Output files are named after the TSV stem, so equal stems would overwrite each other.
    stems = [input_tsv.stem for input_tsv in args.input_tsv]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise SystemExit(f"Several --input-tsv files share the name(s) {duplicates}; their outputs would collide.")

    runner = cc.load_runner(args)
    for i, input_tsv in enumerate(args.input_tsv, start=1):
        print(f"\n=== [{i}/{len(args.input_tsv)}] {input_tsv} ===")
        process_tsv(args, input_tsv, runner, relevant, reference, reference_path, feature_scores)


def process_tsv(
    args: argparse.Namespace,
    input_tsv: Path,
    runner: cc.Runner,
    relevant: dict[int, str],
    reference: dict,
    reference_path: Path,
    feature_scores: Path,
) -> None:
    """Computes and writes the coverage metrics of one input TSV, independently of all others."""
    _, active, metadata = cc.run_sae_on_tsv(args, input_tsv, runner)

    relevant_ids = set(relevant)
    metrics = {}
    tsv_rows = []
    for key, ids in active.items():
        active_set = set(ids)
        covered_relevant = active_set & relevant_ids
        test_ids = set(reference["thresholds"][key]["feature_ids"])
        covered_test = active_set & test_ids
        per_label = {}
        for label in args.feature_labels:
            label_ids = {fid for fid, lab in relevant.items() if lab == label}
            n_cov = len(active_set & label_ids)
            per_label[label] = {"n_covered": n_cov, "n_relevant": len(label_ids), "fraction": fraction(n_cov, len(label_ids))}

        metrics[f"features_active_at_t{key}"] = len(active_set)
        metrics[f"feature_coverage_at_t{key}"] = {
            "n_covered": len(covered_relevant),
            "n_relevant": len(relevant_ids),
            "fraction": fraction(len(covered_relevant), len(relevant_ids)),
            "per_label": per_label,
        }
        metrics[f"feature_coverage_test_at_t{key}"] = {
            "n_covered": len(covered_test),
            "n_test_active": len(test_ids),
            "fraction": fraction(len(covered_test), len(test_ids)),
        }
        tsv_rows.append({
            "threshold": key,
            "features_active": len(active_set),
            "feature_coverage_n": len(covered_relevant),
            "feature_coverage_n_relevant": len(relevant_ids),
            "feature_coverage_frac": fraction(len(covered_relevant), len(relevant_ids)),
            "feature_coverage_test_n": len(covered_test),
            "feature_coverage_test_n_test_active": len(test_ids),
            "feature_coverage_test_frac": fraction(len(covered_test), len(test_ids)),
        })

    out_dir = cc.domain_output_dir(args.path) / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_tsv.stem
    json_path = out_dir / f"{stem}_feature_coverage.json"
    tsv_path = out_dir / f"{stem}_feature_coverage.tsv"

    result = {
        **metadata,
        "reference_json": str(reference_path.resolve()),
        "reference_input_tsv": reference["input_tsv"],
        "feature_scores": str(feature_scores.resolve()),
        "feature_labels": list(args.feature_labels),
        "metrics": metrics,
        "active_feature_ids": active,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tsv_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(tsv_rows)

    def fmt(x: float | None) -> str:
        return "   n/a" if x is None else f"{x:6.3f}"

    print(f"{'t':>8} {'active':>7} {'relevant covered':>22} {'test covered':>22}")
    for row in tsv_rows:
        print(
            f"{row['threshold']:>8} {row['features_active']:>7} "
            f"{row['feature_coverage_n']:>6}/{row['feature_coverage_n_relevant']:<6} ({fmt(row['feature_coverage_frac'])}) "
            f"{row['feature_coverage_test_n']:>6}/{row['feature_coverage_test_n_test_active']:<6} ({fmt(row['feature_coverage_test_frac'])})"
        )
    print(f"Wrote {json_path} and {tsv_path}")


if __name__ == "__main__":
    main()
