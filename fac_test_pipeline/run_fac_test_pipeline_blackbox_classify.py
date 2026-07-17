"""
Classify the top-20 (by log_odds) features of every synthetic blackbox example in
output/features_gen_blackbox.jsonl against the seed-derived feature classification
in output/feature_classification_casehold.jsonl.

From feature_classification_casehold.jsonl:
  - template_features (single entry): top 10 (by min_log_odds, already sorted) are
    the "seed template" feature IDs.
  - subdomain_features (one entry per seed example): top 10 (by log_odds_on_example,
    already sorted) per seed are unioned across all seeds into the "seed subdomain"
    feature ID set.

Each feature on each blackbox example is then labelled:
  seed_template    - feature_id is among the seed template IDs
  seed_subdomain   - else feature_id is among the (unioned) seed subdomain IDs
  new_concept      - else

The labelled records are written to output/features_gen_blackbox_classified.jsonl
(identical structure to features_gen_blackbox.jsonl, each feature dict gains a
"category" key). A short per-example summary of feature IDs per category is printed.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Set, Tuple


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CLASSIFICATION_JSONL = os.path.join(_SCRIPT_DIR, "output", "feature_classification_casehold.jsonl")
_DEFAULT_BLACKBOX_JSONL = os.path.join(_SCRIPT_DIR, "output", "features_gen_blackbox.jsonl")
_DEFAULT_OUTPUT_JSONL = os.path.join(_SCRIPT_DIR, "output", "features_gen_blackbox_classified.jsonl")

SEED_TEMPLATE = "seed_template"
SEED_SUBDOMAIN = "seed_subdomain"
NEW_CONCEPT = "new_concept"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_seed_feature_ids(classification_jsonl: str, top_n: int) -> Tuple[Set[int], Set[int]]:
    """Return (template_ids, subdomain_ids): top-N feature IDs (by the existing
    sort order) from the template_features entry, and the union of top-N feature
    IDs across all subdomain_features entries (one per seed example)."""
    template_ids: Set[int] = set()
    subdomain_ids: Set[int] = set()

    for rec in load_jsonl(classification_jsonl):
        rec_type = rec.get("type")
        top_features = rec.get("features", [])[:top_n]
        if rec_type == "template_features":
            template_ids.update(int(f["feature_id"]) for f in top_features)
        elif rec_type == "subdomain_features":
            subdomain_ids.update(int(f["feature_id"]) for f in top_features)

    return template_ids, subdomain_ids


def classify_feature(feature_id: int, template_ids: Set[int], subdomain_ids: Set[int]) -> str:
    if feature_id in template_ids:
        return SEED_TEMPLATE
    if feature_id in subdomain_ids:
        return SEED_SUBDOMAIN
    return NEW_CONCEPT


def classify_blackbox_examples(
    blackbox_records: List[Dict[str, Any]],
    template_ids: Set[int],
    subdomain_ids: Set[int],
) -> List[Dict[str, Any]]:
    """Return new example records with a "category" key added to each feature dict."""
    classified: List[Dict[str, Any]] = []
    for rec in blackbox_records:
        new_rec = dict(rec)
        new_features = []
        for feat in rec.get("features", []):
            new_feat = dict(feat)
            new_feat["category"] = classify_feature(int(feat["feature_id"]), template_ids, subdomain_ids)
            new_features.append(new_feat)
        new_rec["features"] = new_features
        classified.append(new_rec)
    return classified


def write_jsonl(records: List[Dict[str, Any]], output_jsonl: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_summary(classified_records: List[Dict[str, Any]]) -> None:
    for rec in classified_records:
        by_category: Dict[str, List[int]] = {SEED_TEMPLATE: [], SEED_SUBDOMAIN: [], NEW_CONCEPT: []}
        for feat in rec.get("features", []):
            by_category[feat["category"]].append(int(feat["feature_id"]))
        print(f"Example {rec.get('example_index')}")
        print(
            f"template_features: {by_category[SEED_TEMPLATE]}, "
            f"subdomain_features: {by_category[SEED_SUBDOMAIN]}, "
            f"new_concept: {by_category[NEW_CONCEPT]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify blackbox example features (features_gen_blackbox.jsonl) as "
            "seed_template / seed_subdomain / new_concept against the top-N seed "
            "feature classification (feature_classification_casehold.jsonl)."
        )
    )
    parser.add_argument("--classification-jsonl", type=str, default=_DEFAULT_CLASSIFICATION_JSONL)
    parser.add_argument("--blackbox-jsonl", type=str, default=_DEFAULT_BLACKBOX_JSONL)
    parser.add_argument("--output-jsonl", type=str, default=_DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--top-n", type=int, default=10, help="Top-N (per list) seed features to match against.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.classification_jsonl):
        raise FileNotFoundError(f"Classification JSONL not found: {args.classification_jsonl}")
    if not os.path.isfile(args.blackbox_jsonl):
        raise FileNotFoundError(f"Blackbox JSONL not found: {args.blackbox_jsonl}")

    template_ids, subdomain_ids = load_seed_feature_ids(args.classification_jsonl, top_n=args.top_n)
    print(
        f"Loaded {len(template_ids)} seed template feature IDs and "
        f"{len(subdomain_ids)} seed subdomain feature IDs (top {args.top_n} each)."
    )

    blackbox_records = load_jsonl(args.blackbox_jsonl)
    classified_records = classify_blackbox_examples(blackbox_records, template_ids, subdomain_ids)

    write_jsonl(classified_records, args.output_jsonl)
    print(f"Wrote {len(classified_records)} classified records to {args.output_jsonl}")
    print()

    print_summary(classified_records)


if __name__ == "__main__":
    main()
