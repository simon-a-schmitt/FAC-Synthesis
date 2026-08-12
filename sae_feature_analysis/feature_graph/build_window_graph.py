"""Step 2 of the feature graph pipeline: turn Step 1's per-token shards into
windowed, binarized feature sets ready for Step 3 (NPMI graph construction).

2a. Merge (cached / "einmalig"): reuse merge_shards.run_merge() to build/
    load the flat feat_ids/mags/n_active/doc_offsets/doc_ids arrays. If a
    complete merge already exists under --merged-dir it is reused as-is
    (pass --force-remerge to redo it). run_merge()'s coverage check already
    validates every rank's shards tile its assigned doc range exactly -
    that's a strictly stronger guarantee than just comparing shard counts.

2b. Load sae_feature_analysis/interpret_features_exp/xxx/feature_activation_baseline_agg.tsv
    and read the per-feature p95 column as ref_f (shape [n_features]).
    Features absent from the baseline (never fired in that ~60M-token run,
    i.e. dead/near-dead features - about 87% of the 65536 features here)
    get ref_f = +inf, so their normalized peak magnitude is always 0 and
    they can never binarize ON. Normalization itself happens in 2d.

2c. Window the token axis per document (doc boundaries from doc_offsets,
    windows never cross them), parametrized by --w/--stride. A trailing
    remainder shorter than w/2 is dropped; a remainder >= w/2 becomes one
    final (shorter-than-w) window. Produces window_index.npy [n_windows,2]
    (global token start/end) and window_doc_id.npy [n_windows] (source
    doc_id, for later per-document diagnostics).

2d. Binarize each window: per feature, take the peak (max) raw activation
    magnitude across the window's tokens, divide by ref_f, threshold at
    --tau. Stored as CSR: win_ptr [n_windows+1] / win_feats [nnz], each
    window's feature ids deduplicated and ascending. Windowing (2c) and the
    per-window peak/ref_f normalization don't depend on tau, so --tau
    accepts a comma-separated list: one pass computes the normalized peaks
    once and re-thresholds them per tau value, avoiding redundant work in a
    tau sweep. --w/--stride still require a full separate run each (they
    change the windows themselves).

2e. Each (w, stride, tau) combination is written to its own self-contained
    directory graph/w{w}_s{stride}_tau{tau}/, with win_ptr.npy, win_feats.npy,
    window_index.npy, window_doc_id.npy, df.npy (per-feature window
    frequency - int64 [n_features], "free" since it falls out of the same
    loop and is needed by Step 3's NPMI calc), and config.json (tau, w,
    stride, ref_f definition, filter thresholds, input hashes, a short
    report). Re-running with different params never overwrites another
    combination's directory.
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time

import numpy as np
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from merge_shards import run_merge, git_commit  # noqa: E402
from _common import sha256_of_file  # noqa: E402

DEFAULT_BASELINE_TSV = os.path.abspath(os.path.join(
    SCRIPT_DIR, "..", "interpret_features_exp", "xxx", "feature_activation_baseline_agg.tsv",
))


# ========== 2a: merge (cached) ==========

MERGE_OUTPUTS = ("feat_ids.npy", "mags.npy", "n_active.npy", "doc_offsets.npy", "doc_ids.npy", "meta.json")


def ensure_merged(shards_dir, merged_dir, allow_incomplete, force_remerge):
    have_all = all(os.path.exists(os.path.join(merged_dir, f)) for f in MERGE_OUTPUTS)
    if have_all and not force_remerge:
        print(f"[INFO] Reusing existing merge at {merged_dir} (pass --force-remerge to redo it)")
        with open(os.path.join(merged_dir, "meta.json"), "r", encoding="utf8") as f:
            return json.load(f)
    print(f"[INFO] Merging shards from {shards_dir} -> {merged_dir}")
    return run_merge(shards_dir, merged_dir, allow_incomplete=allow_incomplete)


# ========== 2b: baseline ref_f ==========

def load_ref_f(baseline_tsv, n_features, percentile_col="p95"):
    """Returns (ref_f [n_features] float64, n_covered int). Missing or
    non-positive baseline entries get ref_f = +inf (can never binarize ON,
    since peak_mag / inf == 0 < any positive tau)."""
    ref_f = np.full(n_features, np.inf, dtype=np.float64)
    n_covered = 0
    with open(baseline_tsv, "r", newline="", encoding="utf8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if percentile_col not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{percentile_col}' not found in {baseline_tsv}; available: {reader.fieldnames}"
            )
        for row in reader:
            fid = int(row["feature_id"])
            if not (0 <= fid < n_features):
                continue
            val = float(row[percentile_col])
            if val <= 0:
                continue
            ref_f[fid] = val
            n_covered += 1
    return ref_f, n_covered


# ========== 2c: windowing ==========

def make_windows_for_doc(doc_start, doc_end, w, stride):
    """Strided windows of length w within [doc_start, doc_end), never
    crossing the boundary. A trailing remainder >= w/2 becomes one final
    (possibly shorter than w) window; a remainder < w/2 is dropped. If the
    whole doc is shorter than w, this reduces to: keep it as a single
    (shorter) window iff its length >= w/2, else drop the doc entirely."""
    half = w / 2.0
    windows = []
    pos = doc_start
    while pos + w <= doc_end:
        windows.append((pos, pos + w))
        pos += stride
    covered_end = windows[-1][1] if windows else doc_start
    remainder = doc_end - covered_end
    if remainder >= half:
        windows.append((covered_end, doc_end))
    return windows


# ========== 2d: per-window peak/ref_f normalization + tau binarization ==========

def window_peak_normalized(feat_win, mag_win, active_win, ref_f):
    """For one window's token slice (feat_win/mag_win [n_tok,20],
    active_win [n_tok]), return (uniq_feat asc, normalized) over the
    window's actually-active (feat_id, mag) pairs: peak raw magnitude per
    feature divided by ref_f. Independent of tau."""
    n_tok, k = feat_win.shape
    col_idx = np.arange(k)
    valid = col_idx[None, :] < active_win[:, None]
    if not valid.any():
        return np.empty((0,), dtype=np.uint16), np.empty((0,), dtype=np.float32)

    feat_flat = feat_win[valid]
    mag_flat = mag_win[valid].astype(np.float32)

    order = np.argsort(feat_flat, kind="stable")
    feat_sorted = feat_flat[order]
    mag_sorted = mag_flat[order]

    uniq_feat, first_idx = np.unique(feat_sorted, return_index=True)
    peak_mag = np.maximum.reduceat(mag_sorted, first_idx)
    normalized = (peak_mag / ref_f[uniq_feat]).astype(np.float32)
    return uniq_feat, normalized


def build_graph_for_taus(feat_ids, mags, n_active, doc_offsets, doc_ids, ref_f,
                          w, stride, tau_list, n_features):
    """Single pass over docs: computes windows (2c) and, for each window,
    the tau-independent peak/ref_f normalization (2d) once, then re-
    thresholds it for every value in tau_list."""
    starts, ends, wdoc_ids = [], [], []
    win_feats_parts = {tau: [] for tau in tau_list}
    win_ptr = {tau: [0] for tau in tau_list}
    df = {tau: np.zeros(n_features, dtype=np.int64) for tau in tau_list}

    n_docs = len(doc_ids)
    for doc_idx in tqdm.tqdm(range(n_docs), desc="windowing + binarizing docs"):
        doc_start, doc_end = int(doc_offsets[doc_idx]), int(doc_offsets[doc_idx + 1])
        windows = make_windows_for_doc(doc_start, doc_end, w, stride)
        if not windows:
            continue
        doc_id = int(doc_ids[doc_idx])
        feat_doc = feat_ids[doc_start:doc_end]
        mag_doc = mags[doc_start:doc_end]
        active_doc = n_active[doc_start:doc_end]

        for (ws, we) in windows:
            starts.append(ws)
            ends.append(we)
            wdoc_ids.append(doc_id)

            ls, le = ws - doc_start, we - doc_start
            uniq_feat, normalized = window_peak_normalized(
                feat_doc[ls:le], mag_doc[ls:le], active_doc[ls:le], ref_f,
            )
            for tau in tau_list:
                on_feats = uniq_feat[normalized >= tau]
                win_feats_parts[tau].append(on_feats)
                win_ptr[tau].append(win_ptr[tau][-1] + len(on_feats))
                if len(on_feats):
                    df[tau][on_feats] += 1

    window_index = np.stack(
        [np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)], axis=1,
    ) if starts else np.zeros((0, 2), dtype=np.int64)
    window_doc_id = np.array(wdoc_ids, dtype=np.int64)

    results = {}
    for tau in tau_list:
        win_feats = (
            np.concatenate(win_feats_parts[tau]).astype(np.uint16)
            if win_feats_parts[tau] else np.zeros((0,), dtype=np.uint16)
        )
        results[tau] = {
            "win_ptr": np.array(win_ptr[tau], dtype=np.int64),
            "win_feats": win_feats,
            "df": df[tau],
        }
    return window_index, window_doc_id, results


# ========== 2e: output + report ==========

def fmt_tau(tau):
    return f"{tau:g}"


def validate_csr(win_ptr, win_feats, n_windows):
    if len(win_ptr) != n_windows + 1:
        raise RuntimeError(f"len(win_ptr) {len(win_ptr)} != n_windows+1 {n_windows + 1}")
    if win_ptr[0] != 0:
        raise RuntimeError("win_ptr[0] != 0")
    if int(win_ptr[-1]) != len(win_feats):
        raise RuntimeError("win_ptr[-1] != len(win_feats)")
    if n_windows and np.any(np.diff(win_ptr) < 0):
        raise RuntimeError("win_ptr is not monotonically non-decreasing")


def write_output(out_dir, window_index, window_doc_id, win_ptr, win_feats, df, config):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "window_index.npy"), window_index)
    np.save(os.path.join(out_dir, "window_doc_id.npy"), window_doc_id)
    np.save(os.path.join(out_dir, "win_ptr.npy"), win_ptr)
    np.save(os.path.join(out_dir, "win_feats.npy"), win_feats)
    np.save(os.path.join(out_dir, "df.npy"), df)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: window + binarize Step 1's per-token feature activations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--shards-dir", type=str, default=os.path.join(SCRIPT_DIR, "shards"))
    parser.add_argument("--merged-dir", type=str, default=os.path.join(SCRIPT_DIR, "merged"))
    parser.add_argument("--allow-incomplete-merge", action="store_true",
                         help="Passed through to merge_shards.run_merge if a (re)merge is needed")
    parser.add_argument("--force-remerge", action="store_true", help="Redo 2a even if merged outputs already exist")

    parser.add_argument("--baseline-tsv", type=str, default=DEFAULT_BASELINE_TSV,
                         help="feature_activation_baseline_agg.tsv path")
    parser.add_argument("--percentile-col", type=str, default="p95")
    parser.add_argument("--n-features", type=int, default=65536)

    parser.add_argument("--w", type=int, default=100, help="Window length in tokens")
    parser.add_argument("--stride", type=int, default=50, help="Window stride in tokens")
    parser.add_argument("--tau", type=str, default="1.0",
                         help="Comma-separated list of thresholds, e.g. '0.5,1.0,1.5' (one merge+window pass, "
                              "cheaply re-thresholded per value)")

    parser.add_argument("--graph-root", type=str, default=os.path.join(SCRIPT_DIR, "graph"))
    args = parser.parse_args()

    if args.w <= 0 or args.stride <= 0:
        raise ValueError("--w and --stride must be > 0")
    if args.stride > args.w:
        print(f"[WARN] --stride ({args.stride}) > --w ({args.w}): windows will leave gaps between them.")
    tau_list = sorted({float(x) for x in args.tau.split(",") if x.strip() != ""})
    if not tau_list:
        raise ValueError("--tau must contain at least one value")

    t0 = time.perf_counter()

    # 2a
    merge_meta = ensure_merged(args.shards_dir, args.merged_dir, args.allow_incomplete_merge, args.force_remerge)

    feat_ids = np.load(os.path.join(args.merged_dir, "feat_ids.npy"))
    mags = np.load(os.path.join(args.merged_dir, "mags.npy"))
    n_active = np.load(os.path.join(args.merged_dir, "n_active.npy"))
    doc_offsets = np.load(os.path.join(args.merged_dir, "doc_offsets.npy"))
    doc_ids = np.load(os.path.join(args.merged_dir, "doc_ids.npy"))

    n_docs = len(doc_ids)
    n_tokens = feat_ids.shape[0]
    if len(doc_offsets) != n_docs + 1:
        raise RuntimeError("merged doc_offsets/doc_ids length mismatch")
    if int(doc_offsets[-1]) != n_tokens:
        raise RuntimeError("merged doc_offsets[-1] != n_tokens")
    if n_docs and np.any(np.diff(doc_offsets) <= 0):
        raise RuntimeError("merged doc_offsets not strictly monotonic")
    if len(np.unique(doc_ids)) != n_docs:
        raise RuntimeError("merged doc_ids contain duplicates")
    if n_tokens and int(feat_ids.max()) >= args.n_features:
        raise RuntimeError(
            f"feat_ids contains an id >= --n-features ({args.n_features}); pass the correct --n-features."
        )
    print(f"[INFO] Loaded merge: {n_docs} docs, {n_tokens} tokens")

    # 2b
    ref_f, n_covered = load_ref_f(args.baseline_tsv, args.n_features, args.percentile_col)
    print(f"[INFO] ref_f: {n_covered}/{args.n_features} features have a baseline "
          f"({args.percentile_col}); the rest can never binarize ON")

    # 2c + 2d
    window_index, window_doc_id, per_tau = build_graph_for_taus(
        feat_ids, mags, n_active, doc_offsets, doc_ids, ref_f,
        args.w, args.stride, tau_list, args.n_features,
    )
    n_windows = len(window_doc_id)
    print(f"[INFO] {n_windows} windows (w={args.w}, stride={args.stride})")

    input_hashes = {
        name: sha256_of_file(os.path.join(args.merged_dir, name))
        for name in ("feat_ids.npy", "mags.npy", "n_active.npy", "doc_offsets.npy", "doc_ids.npy")
    }
    input_hashes["baseline_tsv"] = sha256_of_file(args.baseline_tsv)

    repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    wall_clock = time.perf_counter() - t0

    # 2e
    for tau in tau_list:
        win_ptr = per_tau[tau]["win_ptr"]
        win_feats = per_tau[tau]["win_feats"]
        df = per_tau[tau]["df"]
        validate_csr(win_ptr, win_feats, n_windows)

        feats_per_window = np.diff(win_ptr)
        mean_fpw = float(feats_per_window.mean()) if n_windows else 0.0
        median_fpw = float(np.median(feats_per_window)) if n_windows else 0.0
        frac_ubiquitous = float((df / n_windows > 0.3).mean()) if n_windows else 0.0

        out_dir = os.path.join(args.graph_root, f"w{args.w}_s{args.stride}_tau{fmt_tau(tau)}")
        config = {
            "w": args.w,
            "stride": args.stride,
            "tau": tau,
            "n_windows": n_windows,
            "n_features": args.n_features,
            "ref_f_definition": {
                "source_tsv": os.path.abspath(args.baseline_tsv),
                "percentile_column": args.percentile_col,
                "n_features_with_baseline": n_covered,
                "n_features_total": args.n_features,
                "missing_or_nonpositive_baseline_policy": (
                    "ref_f = +inf (peak_mag / inf == 0, so the feature can never binarize ON)"
                ),
            },
            "filter_thresholds": {
                "tail_discard_min_tokens": args.w / 2.0,
                "tau": tau,
            },
            "window_construction": (
                "strided windows of length w within each doc (doc_offsets), never crossing doc "
                "boundaries; trailing remainder >= w/2 kept as one shorter final window, < w/2 dropped"
            ),
            "input_hashes": input_hashes,
            "merged_meta": {"n_docs": n_docs, "n_tokens": n_tokens, "meta": merge_meta},
            "report": {
                "n_windows": n_windows,
                "mean_features_per_window": mean_fpw,
                "median_features_per_window": median_fpw,
                "frac_features_with_df_over_0.3": frac_ubiquitous,
            },
            "wall_clock_seconds": wall_clock,
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "git_commit": git_commit(repo_root),
        }
        write_output(out_dir, window_index, window_doc_id, win_ptr, win_feats, df, config)

        print(f"[tau={tau}] -> {out_dir}")
        print(f"  n_windows={n_windows}  mean_feats/window={mean_fpw:.2f}  "
              f"median_feats/window={median_fpw:.1f}  frac(df/n_win>0.3)={frac_ubiquitous:.4f}")

    print(f"[INFO] Done in {wall_clock:.1f}s")


if __name__ == "__main__":
    main()
