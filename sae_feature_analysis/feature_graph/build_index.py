"""Step 3a: build the transposed (feature -> windows) index for a
graph/w{w}_s{stride}_tau{tau}/ directory produced by build_window_graph.py.

Step 2 already gives us win_ptr/win_feats, a CSR matrix over
(window -> its ON features). Step 3's NPMI/co-occurrence computation needs
the transpose: for a given feature, the sorted list of windows it's ON in
(its "postings list"). That's exactly a CSR -> CSC conversion of the same
matrix, done via scipy.sparse (an O(nnz) counting-sort transpose, not a
comparison sort) rather than hand-rolling one.

Output (same directory): feat_ptr.npy (int64 [n_features+1]) and
feat_wins.npy (int32 [nnz], ascending window ids within each feature).

Consistency check: feat_ptr[f+1] - feat_ptr[f] must equal df[f] for every
feature f (df.npy was written by Step 2 from the same win_ptr/win_feats, so
any mismatch means either file is stale/corrupt) - abort loudly rather than
writing a silently-inconsistent index.
"""

import argparse
import datetime
import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from merge_shards import git_commit  # noqa: E402
from _common import sha256_of_file  # noqa: E402

REQUIRED_INPUTS = ("win_ptr.npy", "win_feats.npy", "df.npy")
INDEX_OUTPUTS = ("feat_ptr.npy", "feat_wins.npy")


def build_index_for_dir(graph_dir, n_features_override=None, force=False):
    missing = [f for f in REQUIRED_INPUTS if not os.path.exists(os.path.join(graph_dir, f))]
    if missing:
        raise FileNotFoundError(f"{graph_dir}: missing required input(s) {missing}; run build_window_graph.py first.")

    if not force and all(os.path.exists(os.path.join(graph_dir, f)) for f in INDEX_OUTPUTS):
        print(f"[INFO] {graph_dir}: index already present, skipping (pass --force to rebuild)")
        return

    t0 = time.perf_counter()

    win_ptr = np.load(os.path.join(graph_dir, "win_ptr.npy"))
    win_feats = np.load(os.path.join(graph_dir, "win_feats.npy"))
    df = np.load(os.path.join(graph_dir, "df.npy"))

    n_windows = len(win_ptr) - 1
    nnz = len(win_feats)
    if n_windows < 0:
        raise RuntimeError(f"{graph_dir}: win_ptr has length {len(win_ptr)} (< 1)")
    if int(win_ptr[-1]) != nnz:
        raise RuntimeError(f"{graph_dir}: win_ptr[-1] ({win_ptr[-1]}) != len(win_feats) ({nnz})")
    if n_windows and np.any(np.diff(win_ptr) < 0):
        raise RuntimeError(f"{graph_dir}: win_ptr is not monotonically non-decreasing")

    n_features = n_features_override if n_features_override is not None else len(df)
    if nnz and int(win_feats.max()) >= n_features:
        raise RuntimeError(f"{graph_dir}: win_feats contains an id >= n_features ({n_features})")

    # CSR(window, feature) -> CSC == transpose == the per-feature postings
    # lists we want. Values don't matter (presence only), so use the
    # smallest workable dtype for the throwaway `data` array.
    data = np.ones(nnz, dtype=np.uint8)
    csr = sp.csr_matrix((data, win_feats.astype(np.int32, copy=False), win_ptr), shape=(n_windows, n_features))
    csc = csr.tocsc()
    csc.sort_indices()  # guarantee ascending window ids per feature; don't rely on tocsc()'s internal ordering as a contract

    feat_ptr = csc.indptr.astype(np.int64)
    feat_wins = csc.indices.astype(np.int32)

    if len(feat_ptr) != n_features + 1:
        raise RuntimeError(f"{graph_dir}: len(feat_ptr) {len(feat_ptr)} != n_features+1 {n_features + 1}")
    if len(feat_wins) != nnz:
        raise RuntimeError(f"{graph_dir}: len(feat_wins) {len(feat_wins)} != nnz {nnz}")

    counts = np.diff(feat_ptr)
    df_arr = df[:n_features].astype(np.int64)
    if not np.array_equal(counts, df_arr):
        bad = np.nonzero(counts != df_arr)[0]
        raise RuntimeError(
            f"{graph_dir}: feat_ptr/df mismatch at {len(bad)} feature(s) "
            f"(first few ids: {bad[:10].tolist()}, feat_ptr counts vs df: "
            f"{list(zip(counts[bad[:10]].tolist(), df_arr[bad[:10]].tolist()))}); aborting."
        )

    np.save(os.path.join(graph_dir, "feat_ptr.npy"), feat_ptr)
    np.save(os.path.join(graph_dir, "feat_wins.npy"), feat_wins)

    wall_clock = time.perf_counter() - t0
    repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    index_meta = {
        "n_features": n_features,
        "n_windows": n_windows,
        "nnz": nnz,
        "method": "scipy.sparse csr_matrix(window,feature).tocsc() transpose, sort_indices() enforced",
        "consistency_check": "feat_ptr[f+1]-feat_ptr[f] == df[f] for all f: passed",
        "input_hashes": {f: sha256_of_file(os.path.join(graph_dir, f)) for f in REQUIRED_INPUTS},
        "wall_clock_seconds": wall_clock,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit(repo_root),
    }
    with open(os.path.join(graph_dir, "index_meta.json"), "w", encoding="utf8") as f:
        json.dump(index_meta, f, indent=2, sort_keys=True)

    print(f"[INFO] {graph_dir}: feat_ptr/feat_wins built ({nnz} nnz, {n_features} features, "
          f"{n_windows} windows) in {wall_clock:.1f}s; consistency check vs df.npy passed")


def discover_graph_dirs(graph_root):
    pattern = os.path.join(graph_root, "w*_s*_tau*")
    return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def main():
    parser = argparse.ArgumentParser(
        description="Step 3a: build the transposed feature->windows CSC index for a graph/w{w}_s{stride}_tau{tau}/ dir.",
    )
    parser.add_argument("--graph-dir", type=str, default=None,
                         help="A single graph/w..._s..._tau.../ directory to index")
    parser.add_argument("--graph-root", type=str, default=os.path.join(SCRIPT_DIR, "graph"),
                         help="Root containing w*_s*_tau*/ subdirs (used with --all)")
    parser.add_argument("--all", action="store_true",
                         help="Build the index for every w*_s*_tau*/ subdir under --graph-root")
    parser.add_argument("--force", action="store_true", help="Rebuild even if feat_ptr.npy/feat_wins.npy already exist")
    parser.add_argument("--n-features", type=int, default=None, help="Override feature count (default: len(df.npy))")
    args = parser.parse_args()

    if not args.graph_dir and not args.all:
        parser.error("pass either --graph-dir <dir> or --all (with --graph-root)")

    targets = [args.graph_dir] if args.graph_dir else discover_graph_dirs(args.graph_root)
    if not targets:
        raise RuntimeError(f"No w*_s*_tau*/ directories found under {args.graph_root}")

    for graph_dir in targets:
        build_index_for_dir(graph_dir, n_features_override=args.n_features, force=args.force)


if __name__ == "__main__":
    main()
