"""
Compare per-feature max-magnitude coverage across three corpora.

Input
-----
Three TSVs in feature_coverage/output, each produced by
run_feature_coverage.py (columns: feature_id, max_normalized_magnitude,
n_prompts_active).

Processing
----------
For every file, a feature counts as "active" if its max_normalized_magnitude
is >= --threshold. This yields one set of feature ids per file.

Output
------
A single JSON file in feature_coverage/output containing, per input file,
how many features are active and how much its active set overlaps with each
of the other two files, plus the full 3-way Venn decomposition (only-in-one,
pairwise-only, and all-three intersections) with the concrete feature ids for
every region.
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Set

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_max_magnitude_tsv(tsv_path: str) -> Dict[int, float]:
    """{feature_id: max_normalized_magnitude} from a run_feature_coverage.py output TSV."""
    features: Dict[int, float] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                fid = int(row["feature_id"])
                mag = float(row["max_normalized_magnitude"])
            except (KeyError, ValueError):
                continue
            features[fid] = mag
    if not features:
        raise ValueError(f"No features found in {tsv_path}")
    return features


def label_for(tsv_path: str) -> str:
    stem = os.path.splitext(os.path.basename(tsv_path))[0]
    suffix = "_max_magnitude"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def active_feature_set(features: Dict[int, float], threshold: float) -> Set[int]:
    return {fid for fid, mag in features.items() if mag >= threshold}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_report(
    labels: List[str],
    paths: List[str],
    all_features: List[Dict[int, float]],
    threshold: float,
) -> Dict:
    active_sets = [active_feature_set(feat, threshold) for feat in all_features]

    files_report = []
    for i, label in enumerate(labels):
        overlap_with = {}
        for j, other_label in enumerate(labels):
            if i == j:
                continue
            overlap_with[other_label] = len(active_sets[i] & active_sets[j])
        files_report.append({
            "label": label,
            "path": os.path.abspath(paths[i]),
            "n_features_total": len(all_features[i]),
            "n_features_active": len(active_sets[i]),
            "overlap_with": overlap_with,
        })

    a, b, c = active_sets
    la, lb, lc = labels

    only_a = a - b - c
    only_b = b - a - c
    only_c = c - a - b
    ab_only = (a & b) - c
    ac_only = (a & c) - b
    bc_only = (b & c) - a
    all_three = a & b & c
    union = a | b | c

    def region(ids: Set[int]) -> Dict:
        return {"n": len(ids), "feature_ids": sorted(ids)}

    venn = {
        f"only_{la}": region(only_a),
        f"only_{lb}": region(only_b),
        f"only_{lc}": region(only_c),
        f"{la}_and_{lb}_only": region(ab_only),
        f"{la}_and_{lc}_only": region(ac_only),
        f"{lb}_and_{lc}_only": region(bc_only),
        f"{la}_and_{lb}_and_{lc}": region(all_three),
    }

    return {
        "threshold": threshold,
        "threshold_rule": "max_normalized_magnitude >= threshold",
        "files": files_report,
        "n_union_active_features": len(union),
        "venn": venn,
    }


def write_report(report: Dict, output_json: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare three run_feature_coverage.py output TSVs: filter each "
            "by a max-normalized-magnitude threshold and report per-file "
            "active-feature counts, pairwise overlaps, and the full 3-way "
            "Venn decomposition (with feature ids)."
        )
    )
    parser.add_argument("--input-tsv-a", type=str, required=True)
    parser.add_argument("--input-tsv-b", type=str, required=True)
    parser.add_argument("--input-tsv-c", type=str, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Minimum max_normalized_magnitude for a feature to count as active.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Destination JSON. Defaults to "
        "feature_coverage/output/coverage_overlap_threshold_<threshold>.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = [args.input_tsv_a, args.input_tsv_b, args.input_tsv_c]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input TSV not found: {path}")

    labels = [label_for(path) for path in paths]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Input files must have distinct labels/stems, got: {labels}")

    all_features = [load_max_magnitude_tsv(path) for path in paths]
    for label, feat in zip(labels, all_features):
        print(f"Loaded {len(feat)} features from {label}")

    report = build_report(labels, paths, all_features, args.threshold)

    output_json = args.output_json
    if not output_json:
        threshold_str = str(args.threshold).replace(".", "p")
        output_json = os.path.join(
            _DEFAULT_OUTPUT_DIR, f"coverage_overlap_threshold_{threshold_str}.json"
        )

    write_report(report, output_json)

    for file_report in report["files"]:
        print(
            f"{file_report['label']}: {file_report['n_features_active']} / "
            f"{file_report['n_features_total']} active; overlap "
            f"{file_report['overlap_with']}"
        )
    print(f"Union of active features: {report['n_union_active_features']}")
    print(f"Wrote overlap report to {output_json}")


if __name__ == "__main__":
    main()
