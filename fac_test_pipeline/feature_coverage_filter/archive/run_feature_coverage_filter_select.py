"""
Greedily select a subset of pool prompts that together maximise SAE-feature
coverage, then extract the corresponding records from the original pool
input file.

Input
-----
- --pool-features-jsonl: output of run_feature_coverage_filter.py, one JSON
  object per line: {"pool_id": ..., "n_content_tokens": ..., "features":
  {feature_id: peak_magnitude, ...}}.
- --input-json: the original pool input file (feature_coverage_filter/input),
  a JSON list of pool items keyed by "pool_id".

Processing
----------
1. Per prompt, only features with peak_magnitude >= --threshold (default
   1.5) count as "active".
2. Greedy maximum-coverage selection: repeatedly pick the not-yet-selected
   prompt whose active features add the most *still-countable* coverage,
   where an individual feature only counts up to --cap (default 3) times
   across all selected prompts — once a feature has been counted --cap
   times, further occurrences of it contribute zero additional gain. Ties
   are broken by the smaller pool_id. Repeat until --n prompts are chosen
   (or the pool is exhausted).

Output
------
The original records (same schema as --input-json) for the selected
pool_ids, in their original input-file order, written as a JSON list to
feature_coverage_filter/output.

The console reports how many individual (unique) features were covered and
the total capped feature-activation count achieved by the selected subset.
"""

import argparse
import collections
import json
import os
from typing import Any, Dict, List, Set, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_POOL_FEATURES_JSONL = os.path.join(
    _SCRIPT_DIR, "output", "cti_vsp_bb_pool_deepseek_1025_feature_magnitudes.jsonl"
)
_DEFAULT_INPUT_JSON = os.path.join(
    _SCRIPT_DIR, "input", "cti_vsp_bb_pool_deepseek_1025.json"
)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_pool_features(pool_features_jsonl: str) -> Dict[Any, Dict[str, float]]:
    """{pool_id: {feature_id_str: peak_magnitude}} from run_feature_coverage_filter.py output."""
    pool_features: Dict[Any, Dict[str, float]] = {}
    with open(pool_features_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pool_id = rec.get("pool_id")
            if pool_id is None:
                continue
            pool_features[pool_id] = rec.get("features", {})
    if not pool_features:
        raise ValueError(f"No pool feature records found in {pool_features_jsonl}")
    return pool_features


def load_input_records(input_json: str) -> List[Dict[str, Any]]:
    with open(input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of pool items in {input_json}")
    return payload


def active_features_by_prompt(
    pool_features: Dict[Any, Dict[str, float]],
    threshold: float,
) -> Dict[Any, Set[int]]:
    """{pool_id: {feature_id, ...}} restricted to peak_magnitude >= threshold."""
    return {
        pool_id: {int(fid) for fid, mag in features.items() if mag >= threshold}
        for pool_id, features in pool_features.items()
    }


# ---------------------------------------------------------------------------
# Greedy max-coverage selection with a per-feature multiplicity cap
# ---------------------------------------------------------------------------

def greedy_select_prompts(
    active_features: Dict[Any, Set[int]],
    n: int,
    cap: int = 3,
) -> Tuple[List[Any], Dict[int, int], Dict[int, int]]:
    """Greedily pick up to n pool_ids maximising capped feature coverage.

    A feature only contributes to the selection gain the first `cap` times
    it appears across selected prompts; further occurrences are free but do
    not increase the score. Ties in gain are broken by the smaller pool_id.

    Returns (selected_pool_ids, capped_counts, raw_counts):
      - capped_counts[fid] = min(occurrences, cap) among selected prompts
      - raw_counts[fid]    = actual occurrences among selected prompts
    """
    capped_counts: Dict[int, int] = collections.defaultdict(int)
    raw_counts: Dict[int, int] = collections.defaultdict(int)
    remaining = set(active_features.keys())
    selected: List[Any] = []

    n_steps = min(n, len(remaining))
    for _ in range(n_steps):
        best_pid = None
        best_gain = -1
        for pid in sorted(remaining):
            feats = active_features[pid]
            # Use .get() (not capped_counts[fid]) so probing a candidate's
            # gain never auto-vivifies a defaultdict entry for features that
            # end up not being selected.
            gain = sum(1 for fid in feats if capped_counts.get(fid, 0) < cap)
            if gain > best_gain:
                best_gain = gain
                best_pid = pid

        selected.append(best_pid)
        remaining.remove(best_pid)
        for fid in active_features[best_pid]:
            raw_counts[fid] += 1
            if capped_counts[fid] < cap:
                capped_counts[fid] += 1

    return selected, dict(capped_counts), dict(raw_counts)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def extract_selected_records(
    input_records: List[Dict[str, Any]],
    selected_pool_ids: Set[Any],
) -> List[Dict[str, Any]]:
    """Records from input_records whose pool_id is selected, original order kept."""
    return [rec for rec in input_records if rec.get("pool_id") in selected_pool_ids]


def write_records_json(records: List[Dict[str, Any]], output_json: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Greedily select n pool prompts maximising SAE-feature coverage "
            "(features counted only above --threshold, each feature capped "
            "at --cap occurrences across the selection), then extract the "
            "corresponding records from the original pool input file."
        )
    )
    parser.add_argument("--pool-features-jsonl", type=str, default=_DEFAULT_POOL_FEATURES_JSONL)
    parser.add_argument("--input-json", type=str, default=_DEFAULT_INPUT_JSON)

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Minimum peak_magnitude for a feature to count as active.",
    )
    parser.add_argument("--n", type=int, required=True, help="Number of prompts to select.")
    parser.add_argument(
        "--cap",
        type=int,
        default=3,
        help="Max number of times an individual feature counts towards the "
        "selection gain (across selected prompts).",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Destination JSON. Defaults to "
        "feature_coverage_filter/output/<input-json-stem>_greedy_top<n>_t<threshold>.json.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.pool_features_jsonl):
        raise FileNotFoundError(f"Pool-features JSONL not found: {args.pool_features_jsonl}")
    if not os.path.isfile(args.input_json):
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    pool_features = load_pool_features(args.pool_features_jsonl)
    input_records = load_input_records(args.input_json)
    print(
        f"Loaded {len(pool_features)} pool feature records from "
        f"{args.pool_features_jsonl} and {len(input_records)} input records "
        f"from {args.input_json}"
    )

    active_features = active_features_by_prompt(pool_features, args.threshold)

    selected_pool_ids, capped_counts, raw_counts = greedy_select_prompts(
        active_features, n=args.n, cap=args.cap
    )

    output_json = args.output_json
    if not output_json:
        stem = os.path.splitext(os.path.basename(args.input_json))[0]
        threshold_str = str(args.threshold).replace(".", "p")
        output_json = os.path.join(
            _SCRIPT_DIR, "output", f"{stem}_greedy_top{args.n}_t{threshold_str}.json"
        )

    selected_records = extract_selected_records(input_records, set(selected_pool_ids))
    write_records_json(selected_records, output_json)

    n_individual_features = len(capped_counts)
    n_total_capped = sum(capped_counts.values())
    n_total_raw = sum(raw_counts.values())
    print(
        f"Selected {len(selected_pool_ids)} / {len(active_features)} prompts "
        f"(threshold={args.threshold}, cap={args.cap})."
    )
    print(f"Aktivierte (unique) features mit Aktivierung > {args.threshold}: {n_individual_features}")
    print(
        f"Aktivierte features mit Aktivierung > {args.threshold} "
        f"(cap {args.cap} pro unique feature): {n_total_capped}"
    )
    print(f"Features insgesamt aktiv: {n_total_raw}")
    print(f"Wrote {len(selected_records)} records to {output_json}")


if __name__ == "__main__":
    main()
