"""
Classify and aggregate the per-example "new_concept_features" found by
run_seed_gen_feature_comparison.py (output/*_feature_evolution.jsonl).

Sorting
-------
Records are sorted ascending by batch_id (primary), seed_example_id
(secondary), example_index (tertiary, for a stable order among records that
share the same batch_id/seed_example_id). Records from the original seed
examples (type == "seed") carry no batch_id; they sort first (treated as
coming before batch 0), since they represent the origin example each call is
derived from.

Classification
--------------
Every feature in a record's "new_concept_features" is judged against all
records that come *before* it in the sorted order, along two independent
axes:

  - global axis: "new_unique_concept" if no earlier record (regardless of
    which call it came from) already listed this feature_id as a new
    concept; otherwise "concept_duplicate".
  - call axis: "new_unique_call_concept" if no earlier record from the same
    call (identical batch_id AND seed_example_id) already listed this
    feature_id; otherwise "concept_call_duplicate".

A feature can therefore be e.g. a concept_duplicate (seen in an earlier
call) while still being new_unique_call_concept (first time in *this* call).

Outputs
-------
1. --output-jsonl: one line per input record, sorted, carrying batch_id,
   seed_example_id, text, subdomain_features_found, and new_concept_features
   annotated with global_status/call_status.
2. --output-aggregation-json: mean per-example counts (template features
   found, subdomain features found, and the four concept classification
   buckets), both overall and broken down per batch_id.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_INPUT_JSONL = os.path.join(_SCRIPT_DIR, "output", "cti_vsp_bb_deepseek_feature_evolution.jsonl")
_DEFAULT_OUTPUT_JSONL = os.path.join(_SCRIPT_DIR, "output", "cti_vsp_bb_deepseek_feature_evolution_classified.jsonl")
_DEFAULT_OUTPUT_AGGREGATION_JSON = os.path.join(
    _SCRIPT_DIR, "output", "cti_vsp_bb_deepseek_feature_evolution_aggregation.json"
)

CountKey = Tuple[Optional[int], int]  # (batch_id, seed_example_id) call identity


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def sort_key(rec: Dict[str, Any]) -> Tuple[int, int, int]:
    batch_id = rec.get("batch_id")
    seed_example_id = rec.get("seed_example_id")
    example_index = rec.get("example_index")
    return (
        batch_id if batch_id is not None else -1,
        seed_example_id if seed_example_id is not None else -1,
        example_index if example_index is not None else -1,
    )


def classify_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort records and annotate each new_concept_feature with its
    global_status/call_status, based only on records earlier in the sort
    order (never on records from the same example or later)."""
    ordered = sorted(records, key=sort_key)

    seen_global: set = set()
    seen_by_call: Dict[CountKey, set] = {}

    classified: List[Dict[str, Any]] = []
    for rec in ordered:
        batch_id = rec.get("batch_id")
        seed_example_id = rec.get("seed_example_id")
        call_key: CountKey = (batch_id, seed_example_id)
        call_seen = seen_by_call.setdefault(call_key, set())

        annotated_features = []
        for feat in rec.get("new_concept_features", []):
            fid = feat["feature_id"]
            annotated_features.append(
                {
                    "feature_id": fid,
                    "activation_density": feat.get("activation_density"),
                    "peak_magnitude": feat.get("peak_magnitude"),
                    "log_odds": feat.get("log_odds"),
                    "global_status": "concept_duplicate" if fid in seen_global else "new_unique_concept",
                    "call_status": "concept_call_duplicate" if fid in call_seen else "new_unique_call_concept",
                }
            )

        # Only mark features as "seen" after classifying the whole example,
        # so features within the same example never collide with each other.
        for feat in annotated_features:
            seen_global.add(feat["feature_id"])
            call_seen.add(feat["feature_id"])

        classified.append(
            {
                "batch_id": batch_id,
                "seed_example_id": seed_example_id,
                "text": rec.get("text"),
                "template_features_found": rec.get("template_features_found", []),
                "subdomain_features_found": rec.get("subdomain_features_found", []),
                "new_concept_features": annotated_features,
            }
        )

    return classified


def _example_counts(rec: Dict[str, Any]) -> Dict[str, int]:
    features = rec["new_concept_features"]
    return {
        "n_template_features_found": len(rec.get("template_features_found", [])),
        "n_subdomain_features_found": len(rec["subdomain_features_found"]),
        "n_new_unique_concept": sum(1 for f in features if f["global_status"] == "new_unique_concept"),
        "n_concept_duplicate": sum(1 for f in features if f["global_status"] == "concept_duplicate"),
        "n_new_unique_call_concept": sum(1 for f in features if f["call_status"] == "new_unique_call_concept"),
        "n_concept_call_duplicate": sum(1 for f in features if f["call_status"] == "concept_call_duplicate"),
    }


def _mean_counts(rows: List[Dict[str, int]]) -> Dict[str, float]:
    n = len(rows)
    keys = [
        "n_template_features_found",
        "n_subdomain_features_found",
        "n_new_unique_concept",
        "n_concept_duplicate",
        "n_new_unique_call_concept",
        "n_concept_call_duplicate",
    ]
    means = {f"avg_{k[2:]}": (sum(r[k] for r in rows) / n if n else 0.0) for k in keys}
    return {"n_examples": n, **means}


def aggregate(classified: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    rows_by_batch: Dict[Any, List[Dict[str, int]]] = {}
    for rec in classified:
        counts = _example_counts(rec)
        rows.append(counts)
        batch_label = rec["batch_id"] if rec["batch_id"] is not None else "seed"
        rows_by_batch.setdefault(batch_label, []).append(counts)

    per_batch = {
        str(batch_label): _mean_counts(batch_rows)
        for batch_label, batch_rows in sorted(rows_by_batch.items(), key=lambda kv: (kv[0] == "seed", kv[0]))
    }

    return {
        "overall": _mean_counts(rows),
        "per_batch": per_batch,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sort a feature-evolution JSONL by (batch_id, seed_example_id), classify each "
            "new_concept_feature as new/duplicate on a global and a per-call axis, and write "
            "the classified per-example records plus an overall/per-batch aggregation."
        )
    )
    parser.add_argument("--input-jsonl", type=str, default=_DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-jsonl", type=str, default=_DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-aggregation-json", type=str, default=_DEFAULT_OUTPUT_AGGREGATION_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = load_jsonl(args.input_jsonl)
    print(f"Loaded {len(records)} records from {args.input_jsonl}")

    classified = classify_records(records)

    out_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for rec in classified:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(classified)} classified records to {args.output_jsonl}")

    aggregation = aggregate(classified)
    agg_dir = os.path.dirname(os.path.abspath(args.output_aggregation_json))
    if agg_dir:
        os.makedirs(agg_dir, exist_ok=True)
    with open(args.output_aggregation_json, "w", encoding="utf-8") as f:
        json.dump(aggregation, f, indent=2)
    print(f"Wrote aggregation to {args.output_aggregation_json}")

    print("Overall:")
    for k, v in aggregation["overall"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
