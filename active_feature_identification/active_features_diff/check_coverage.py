#!/usr/bin/env python3
"""
Prueft fuer alle Eintraege in der feature-relevance-scores JSONL-Datei mit
Label "Yes", "Probably" oder "Maybe", ob die zugehoerige feature_id bereits
in der Liste der aktiven Features enthalten ist.

Gibt eine Verteilung pro Label aus: wie viele Eintraege abgedeckt (in der
Liste enthalten) und wie viele nicht abgedeckt sind.
"""

import json
from pathlib import Path

SCORES_PATH = Path(
    "/pfs/work9/workspace/scratch/ka_ai3967-master_thesis_exp/code/FAC-Synthesis/"
    "data_synthesis/data/feature_scores/toxicity_detection_feature_relevance_scores.jsonl"
)
ACTIVE_FEATURES_PATH = Path(
    "/pfs/work9/workspace/scratch/ka_ai3967-master_thesis_exp/code/FAC-Synthesis/"
    "active_feature_identification/get_active_features/output/"
    "toxicity_seed_group_01_threshold_0_active_features.json"
)

RELEVANT_LABELS = {"Yes", "Probably", "Maybe"}


def load_active_features(path: Path) -> set[str]:
    with path.open() as f:
        active = json.load(f)
    return {str(fid) for fid in active}


def main() -> None:
    active_features = load_active_features(ACTIVE_FEATURES_PATH)

    counts = {label: {"covered": 0, "not_covered": 0} for label in RELEVANT_LABELS}
    not_covered_ids = {label: [] for label in RELEVANT_LABELS}

    with SCORES_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            label = entry.get("label")
            if label not in RELEVANT_LABELS:
                continue

            feature_id = str(entry["feature_id"])
            if feature_id in active_features:
                counts[label]["covered"] += 1
            else:
                counts[label]["not_covered"] += 1
                not_covered_ids[label].append(feature_id)

    print("Verteilung nach Label (abgedeckt = feature_id in active_features):\n")
    header = f"{'Label':<10} {'Abgedeckt':>10} {'Nicht abgedeckt':>17} {'Gesamt':>8}"
    print(header)
    print("-" * len(header))

    total_covered = 0
    total_not_covered = 0
    for label in ["Yes", "Probably", "Maybe"]:
        covered = counts[label]["covered"]
        not_covered = counts[label]["not_covered"]
        total = covered + not_covered
        total_covered += covered
        total_not_covered += not_covered
        print(f"{label:<10} {covered:>10} {not_covered:>17} {total:>8}")

    print("-" * len(header))
    grand_total = total_covered + total_not_covered
    print(f"{'Gesamt':<10} {total_covered:>10} {total_not_covered:>17} {grand_total:>8}")

    out_path = Path(__file__).parent / "not_covered_feature_ids.json"
    with out_path.open("w") as f:
        json.dump(not_covered_ids, f, indent=2)
    print(f"\nNicht abgedeckte feature_ids je Label gespeichert in: {out_path}")


if __name__ == "__main__":
    main()
