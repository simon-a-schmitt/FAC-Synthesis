"""Step 3b/3c: query the feature co-occurrence graph built by Step 2
(win_ptr/win_feats/df) and Step 3a (feat_ptr/feat_wins).

3b. neighbors(f): raw co-occurrence counts of feature f against every other
    feature, via a vectorized two-hop gather - feat_wins[f] (hop 1: which
    windows f is ON in, from the Step 3a transposed index) -> win_ptr
    slices (hop 2: which features are ON in each of those windows, from
    Step 2's CSR) -> bincount. No Python loop over the (potentially many)
    windows: the ragged per-window slices are flattened via a single
    vectorized gather (see _ragged_gather) before the one bincount call.
    f's self-count is zeroed (every window containing f trivially "co-
    occurs" with f itself, which isn't a real neighbor relationship).

    neighbors_batch(seeds): the same raw counts for many seed features at
    once, as B.T @ B[:, seeds] where B is the (window x feature) binary
    occurrence matrix - built directly from feat_ptr/feat_wins alone (its
    CSC form *is* B; B.T as CSR reuses the same three arrays with the
    shape swapped, so no separate transpose/copy is needed). Result is a
    dense [n_features, len(seeds)] int64 matrix; each seed's self-count
    (row seed, its own column) is zeroed the same way as neighbors().

3c. weight_and_cut(): min_count is a hard filter applied *before* NPMI is
    computed (drops noisy low-count pairs and avoids computing log() over
    them at all). NPMI is then computed from the surviving counts, df, and
    n_windows, and results are cut down to the top-k by NPMI (--k), with
    an optional additional NPMI floor (--min-npmi). Both are CLI
    parameters, not hardcoded.

    For each surviving (seed, neighbor) pair, shared_windows_and_docs()
    additionally reports the actual shared windows - intersect1d(feat_wins[f],
    feat_wins[g]) (both postings lists are sorted and duplicate-free per
    build_index.py, so assume_unique=True is valid and skips an internal
    sort) - and n_distinct_docs, the number of distinct source documents
    (window_doc_id, from Step 2c) those shared windows come from. This is
    only computed for the already top-k-cut neighbors, not the full
    candidate pool, so the extra Python-level loop (over at most k pairs)
    is cheap.
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ========== graph loading ==========

class Graph:
    __slots__ = ("win_ptr", "win_feats", "feat_ptr", "feat_wins", "df",
                 "window_doc_id", "n_windows", "n_features")

    def __init__(self, win_ptr, win_feats, feat_ptr, feat_wins, df, window_doc_id, n_windows, n_features):
        self.win_ptr = win_ptr
        self.win_feats = win_feats
        self.feat_ptr = feat_ptr
        self.feat_wins = feat_wins
        self.df = df
        self.window_doc_id = window_doc_id
        self.n_windows = n_windows
        self.n_features = n_features


def load_graph(graph_dir):
    required = ("win_ptr.npy", "win_feats.npy", "feat_ptr.npy", "feat_wins.npy", "df.npy", "window_doc_id.npy")
    missing = [f for f in required if not os.path.exists(os.path.join(graph_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"{graph_dir}: missing {missing}; run build_window_graph.py (2) and build_index.py (3a) first."
        )
    win_ptr = np.load(os.path.join(graph_dir, "win_ptr.npy"))
    win_feats = np.load(os.path.join(graph_dir, "win_feats.npy"))
    feat_ptr = np.load(os.path.join(graph_dir, "feat_ptr.npy"))
    feat_wins = np.load(os.path.join(graph_dir, "feat_wins.npy"))
    df = np.load(os.path.join(graph_dir, "df.npy"))
    window_doc_id = np.load(os.path.join(graph_dir, "window_doc_id.npy"))
    n_windows = len(win_ptr) - 1
    n_features = len(df)
    if len(window_doc_id) != n_windows:
        raise RuntimeError(
            f"{graph_dir}: window_doc_id has {len(window_doc_id)} entries but win_ptr implies {n_windows} windows"
        )
    return Graph(win_ptr, win_feats, feat_ptr, feat_wins, df, window_doc_id, n_windows, n_features)


# ========== 3b: query core ==========

def _ragged_gather(source, starts, lengths):
    """Vectorized concatenation of len(starts) variable-length slices
    source[starts[i]:starts[i]+lengths[i]] into one flat array, without a
    Python loop over the groups."""
    total = int(lengths.sum())
    if total == 0:
        return source[:0]
    group_start_pos = np.repeat(np.cumsum(lengths) - lengths, lengths)
    idx_within = np.arange(total, dtype=np.int64) - group_start_pos
    gather_idx = np.repeat(starts, lengths) + idx_within
    return source[gather_idx]


def neighbors(f, graph):
    """Raw co-occurrence count vector [n_features] for feature f, self-count zeroed."""
    wins_f = graph.feat_wins[graph.feat_ptr[f]:graph.feat_ptr[f + 1]].astype(np.int64)
    if len(wins_f) == 0:
        return np.zeros(graph.n_features, dtype=np.int64)

    starts = graph.win_ptr[wins_f]
    ends = graph.win_ptr[wins_f + 1]
    lengths = ends - starts

    flat_feats = _ragged_gather(graph.win_feats, starts, lengths).astype(np.int64)
    counts = np.bincount(flat_feats, minlength=graph.n_features)
    if len(counts) > graph.n_features:
        raise RuntimeError(f"win_feats contains an id >= n_features ({graph.n_features}); corrupt index?")
    counts[f] = 0
    return counts


def _build_B_matrices(graph):
    """B (window x feature, CSC) and B.T (feature x window, CSR), both built
    directly from feat_ptr/feat_wins (Step 3a's index *is* B in CSC form;
    B.T as CSR reuses the identical three arrays with the shape swapped -
    no transpose/copy needed)."""
    data = np.ones(len(graph.feat_wins), dtype=np.int32)
    B_csc = sp.csc_matrix((data, graph.feat_wins, graph.feat_ptr), shape=(graph.n_windows, graph.n_features))
    BT_csr = sp.csr_matrix((data, graph.feat_wins, graph.feat_ptr), shape=(graph.n_features, graph.n_windows))
    return B_csc, BT_csr


def neighbors_batch(seeds, graph):
    """Raw co-occurrence counts for many seeds at once: B.T @ B[:, seeds],
    dense [n_features, len(seeds)] int64, each seed's self-count zeroed."""
    seeds = np.asarray(seeds, dtype=np.int64)
    B_csc, BT_csr = _build_B_matrices(graph)
    sub = B_csc[:, seeds]          # [n_windows, |S|]
    result = (BT_csr @ sub).toarray().astype(np.int64)  # [n_features, |S|]
    for j, s in enumerate(seeds):
        result[s, j] = 0
    return result


# ========== 3c: NPMI weighting + cut ==========

def compute_npmi(counts, df_other, df_seed, n_windows):
    """NPMI from raw co-occurrence counts, computed in log space. counts
    must be > 0 (call after the min_count filter). The count == n_windows
    edge case (pair co-occurs in literally every window, so -log(p_fg) ==
    0) is mapped to the standard NPMI=1.0 convention instead of 0/0."""
    counts = counts.astype(np.float64)
    log_n = np.log(n_windows)
    log_c = np.log(counts)
    pmi = log_c + log_n - np.log(df_other.astype(np.float64)) - np.log(float(df_seed))
    denom = log_n - log_c
    return np.where(denom > 0, pmi / np.where(denom > 0, denom, 1.0), 1.0)


def weight_and_cut(counts, df, seed, n_windows, min_count=10, k=50, min_npmi=None):
    """counts: raw co-occurrence vector [n_features] for `seed` (self
    already zeroed). Returns (neighbor_ids, npmi, counts) for up to k
    neighbors, sorted descending by NPMI, after the hard min_count filter
    and the optional min_npmi floor."""
    candidate_ids = np.nonzero(counts >= min_count)[0]
    if len(candidate_ids) == 0:
        return (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.int64))

    candidate_counts = counts[candidate_ids]
    npmi = compute_npmi(candidate_counts, df[candidate_ids], df[seed], n_windows)

    if min_npmi is not None:
        keep = npmi >= min_npmi
        candidate_ids = candidate_ids[keep]
        candidate_counts = candidate_counts[keep]
        npmi = npmi[keep]

    order = np.argsort(-npmi, kind="stable")[:k]
    return candidate_ids[order], npmi[order], candidate_counts[order]


def shared_windows_and_docs(seed, neighbor_ids, graph):
    """For each id in neighbor_ids (already top-k cut - this is O(k), not
    called over the full candidate pool), the actual shared windows with
    `seed` and how many distinct source documents they come from.

    Returns (shared_wins: list[np.ndarray] of window ids, one per neighbor,
             n_distinct_docs: np.ndarray[int64], same order as neighbor_ids).
    """
    wins_seed = graph.feat_wins[graph.feat_ptr[seed]:graph.feat_ptr[seed + 1]]
    shared_wins = []
    n_distinct_docs = np.empty(len(neighbor_ids), dtype=np.int64)
    for i, g in enumerate(neighbor_ids.tolist()):
        wins_g = graph.feat_wins[graph.feat_ptr[g]:graph.feat_ptr[g + 1]]
        shared = np.intersect1d(wins_seed, wins_g, assume_unique=True)
        shared_wins.append(shared)
        n_distinct_docs[i] = np.unique(graph.window_doc_id[shared]).size if len(shared) else 0
    return shared_wins, n_distinct_docs


def ego_network(f, graph, min_count=10, k=50, min_npmi=None):
    """Returns (neighbor_ids, npmi, counts, shared_wins, n_distinct_docs).
    len(shared_wins[i]) always equals counts[i] by construction (the raw
    co-occurrence count *is* the number of shared windows) - a useful
    consistency check on the index if it ever doesn't."""
    counts = neighbors(f, graph)
    ids, npmi, cnts = weight_and_cut(counts, graph.df, f, graph.n_windows, min_count, k, min_npmi)
    shared_wins, n_distinct_docs = shared_windows_and_docs(f, ids, graph)
    return ids, npmi, cnts, shared_wins, n_distinct_docs


def ego_networks_batch(seeds, graph, min_count=10, k=50, min_npmi=None):
    """Returns {seed: (neighbor_ids, npmi, counts, shared_wins, n_distinct_docs)}.
    The expensive part (raw counts for all seeds) is one batched scipy
    matmul; the top-k ranking and shared-window enrichment are then done
    per seed (cheap - O(len(seeds) * k) over already-small arrays)."""
    seeds = np.asarray(seeds, dtype=np.int64)
    counts_matrix = neighbors_batch(seeds, graph)  # [n_features, |S|]
    results = {}
    for j, s in enumerate(seeds):
        ids, npmi, cnts = weight_and_cut(counts_matrix[:, j], graph.df, s, graph.n_windows, min_count, k, min_npmi)
        shared_wins, n_distinct_docs = shared_windows_and_docs(int(s), ids, graph)
        results[int(s)] = (ids, npmi, cnts, shared_wins, n_distinct_docs)
    return results


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="Step 3b/3c: query top-k NPMI-weighted neighbors for one or more SAE features.",
    )
    parser.add_argument("--graph-dir", type=str, required=True,
                         help="graph/w{w}_s{stride}_tau{tau}/ directory (needs Step 2 + Step 3a outputs)")
    parser.add_argument("--features", type=str, required=True,
                         help="Comma-separated feature ids to query, e.g. '42' or '42,100,7'")
    parser.add_argument("--min-count", type=int, default=10, help="Hard co-occurrence count filter before NPMI")
    parser.add_argument("--k", type=int, default=50, help="Top-k neighbors by NPMI")
    parser.add_argument("--min-npmi", type=float, default=None, help="Optional additional NPMI floor")
    parser.add_argument("--out", type=str, default=None, help="Write results as TSV here instead of printing")
    args = parser.parse_args()

    if args.min_count <= 0:
        raise ValueError("--min-count must be > 0 (NPMI requires count > 0)")
    if args.k <= 0:
        raise ValueError("--k must be > 0")

    seeds = [int(x) for x in args.features.split(",") if x.strip() != ""]
    graph = load_graph(args.graph_dir)
    for s in seeds:
        if not (0 <= s < graph.n_features):
            raise ValueError(f"feature {s} out of range [0, {graph.n_features})")

    if len(seeds) == 1:
        s = seeds[0]
        ids, npmi, counts, shared_wins, n_distinct_docs = ego_network(s, graph, args.min_count, args.k, args.min_npmi)
        results = {s: (ids, npmi, counts, shared_wins, n_distinct_docs)}
    else:
        results = ego_networks_batch(seeds, graph, args.min_count, args.k, args.min_npmi)

    rows = ["seed\trank\tneighbor\tcount\tnpmi\tn_distinct_docs\tshared_wins"]
    for s in seeds:
        ids, npmi, counts, shared_wins, n_distinct_docs = results[s]
        for rank, (nid, c, sc, sw, ndd) in enumerate(
            zip(ids.tolist(), counts.tolist(), npmi.tolist(), shared_wins, n_distinct_docs.tolist()), start=1,
        ):
            sw_str = ",".join(map(str, sw.tolist()))
            rows.append(f"{s}\t{rank}\t{nid}\t{c}\t{sc:.6f}\t{ndd}\t{sw_str}")
        if len(ids) == 0:
            print(f"[INFO] seed {s}: no neighbors survived min_count={args.min_count}"
                  + (f" / min_npmi={args.min_npmi}" if args.min_npmi is not None else ""))

    output = "\n".join(rows)
    if args.out:
        with open(args.out, "w", encoding="utf8") as f:
            f.write(output + "\n")
        print(f"Wrote {len(rows) - 1} rows to {args.out}")
    else:
        print(output)


if __name__ == "__main__":
    main()
