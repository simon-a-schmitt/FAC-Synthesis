"""
Extract the ids of all features whose max_normalized_magnitude exceeds a
threshold from a single run_feature_coverage.py output TSV.

Input, threshold and output path are hardcoded below.
"""

import csv
import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_TSV = os.path.join(
    _SCRIPT_DIR, "output", "cti_vsp_gold_487_fc_content.tsv"
)
THRESHOLD = 1.5
OUTPUT_JSON = os.path.join(
    _SCRIPT_DIR, "output", "cti_vsp_gold_487_active_feature_ids.json"
)


def main() -> None:
    active_feature_ids = []
    with open(INPUT_TSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if float(row["max_raw_magnitude"]) > THRESHOLD:
                active_feature_ids.append(int(row["feature_id"]))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(active_feature_ids, f, indent=2)

    print(
        f"{len(active_feature_ids)} features with max_raw_magnitude > "
        f"{THRESHOLD} written to {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()
