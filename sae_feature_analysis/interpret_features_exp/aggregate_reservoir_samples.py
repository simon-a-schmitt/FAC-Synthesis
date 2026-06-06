"""
Aggregates reservoir-sampled activation values across all shards and writes
one TSV row per feature:

    feature_id  \t  activation_count  \t  activation_values  \t  <stats...>

Only features with activation_count >= 100 are kept.
Statistics written per feature: min, p10, p20, ..., p80, p85, p90, p95, max.
"""

import argparse
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_shard(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_shard_dirs(base_dir: str) -> list[str]:
    entries = sorted(
        e for e in os.listdir(base_dir)
        if e.startswith("reservoir_samples_shard") and "_of_" in e
        and os.path.isdir(os.path.join(base_dir, e))
    )
    return [os.path.join(base_dir, e) for e in entries]


def aggregate(shard_dirs: list[str], pkl_filename: str) -> dict[int, dict]:
    merged: dict[int, dict] = {}

    for shard_dir in shard_dirs:
        pkl_path = os.path.join(shard_dir, pkl_filename)
        if not os.path.exists(pkl_path):
            print(f"[WARN] Not found, skipping: {pkl_path}", file=sys.stderr)
            continue

        print(f"[INFO] Loading {pkl_path}", file=sys.stderr)
        data = load_shard(pkl_path)
        feature_samplers = data.get("feature_samplers", {})
        print(
            f"       {len(feature_samplers)} features, "
            f"processed_docs={data.get('processed_docs')}, "
            f"completed={data.get('completed')}",
            file=sys.stderr,
        )

        for feature_id, state in feature_samplers.items():
            fid = int(feature_id)
            items = list(state["items"])
            seen = int(state["seen"])

            if fid not in merged:
                merged[fid] = {"items": [], "seen": 0}

            merged[fid]["items"].extend(items)
            merged[fid]["seen"] += seen

    return merged


PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95]
MIN_ACTIVATION_COUNT = 100


def compute_stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float32)
    stats = {"min": float(arr.min()), "max": float(arr.max())}
    for p in PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(arr, p))
    return stats


def write_tsv(merged: dict[int, dict], out_path: str) -> None:
    stat_cols = ["min"] + [f"p{p}" for p in PERCENTILES] + ["max"]
    header = "\t".join(["feature_id", "activation_count"] + stat_cols)

    kept = {fid: e for fid, e in merged.items() if e["seen"] >= MIN_ACTIVATION_COUNT}
    skipped = len(merged) - len(kept)
    print(
        f"[INFO] Keeping {len(kept)} features (dropped {skipped} with < {MIN_ACTIVATION_COUNT} activations)",
        file=sys.stderr,
    )
    print(f"[INFO] Writing to {out_path}", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for fid in sorted(kept.keys()):
            entry = kept[fid]
            stats = compute_stats(entry["items"])
            stat_vals = "\t".join(f"{stats[c]:.6g}" for c in stat_cols)
            f.write(f"{fid}\t{entry['seen']}\t{stat_vals}\n")

    print("[INFO] Done.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate reservoir-sample shards into a feature-level TSV.",
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "xxx"),
        help="Directory that contains the reservoir_samples_shard*_of_* subdirectories "
             "(default: <script_dir>/xxx)",
    )
    parser.add_argument(
        "--pkl-filename",
        type=str,
        default="reservoir_samples.checkpoint.pkl",
        help="Pickle filename inside each shard directory (default: reservoir_samples.checkpoint.pkl)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(SCRIPT_DIR, "xxx", "feature_activation_baseline_agg.tsv"),
        help="Output TSV path (default: <shard-dir>/feature_activation_baseline_agg.tsv)",
    )
    args = parser.parse_args()

    shard_dirs = find_shard_dirs(args.shard_dir)
    if not shard_dirs:
        print(f"[ERROR] No shard directories found in {args.shard_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(shard_dirs)} shard(s):", file=sys.stderr)
    for d in shard_dirs:
        print(f"       {d}", file=sys.stderr)

    merged = aggregate(shard_dirs, args.pkl_filename)
    write_tsv(merged, args.out)


if __name__ == "__main__":
    main()
