"""
Compare two active-feature-id JSON files (each a JSON list of ints) and
compute their set difference (both directions) and intersection, along
with the cardinality of each resulting set.

Input paths and output path are hardcoded below.
"""

import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_JSON_A = os.path.join(
    _SCRIPT_DIR, "output", "cti_vsp_gold_487_active_feature_ids.json"
)
INPUT_JSON_B = os.path.join(
    _SCRIPT_DIR, "output", "cti_vsp_gold_test_500_active_feature_ids.json"
)
OUTPUT_JSON = os.path.join(
    _SCRIPT_DIR, "output", "cti_vsp_gold_487_vs_test_500_comparison.json"
)


def main() -> None:
    with open(INPUT_JSON_A, "r", encoding="utf-8") as f:
        set_a = set(json.load(f))
    with open(INPUT_JSON_B, "r", encoding="utf-8") as f:
        set_b = set(json.load(f))

    only_in_a = sorted(set_a - set_b)
    only_in_b = sorted(set_b - set_a)
    intersection = sorted(set_a & set_b)

    result = {
        "input_a": INPUT_JSON_A,
        "input_b": INPUT_JSON_B,
        "count_a": len(set_a),
        "count_b": len(set_b),
        "only_in_a": only_in_a,
        "count_only_in_a": len(only_in_a),
        "only_in_b": only_in_b,
        "count_only_in_b": len(only_in_b),
        "intersection": intersection,
        "count_intersection": len(intersection),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"|A| = {len(set_a)}, |B| = {len(set_b)}")
    print(f"only in A: {len(only_in_a)}")
    print(f"only in B: {len(only_in_b)}")
    print(f"intersection: {len(intersection)}")
    print(f"written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
