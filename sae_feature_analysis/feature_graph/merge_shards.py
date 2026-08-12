"""Merge per-rank shards produced by collect_feature_activations.py into the
final flat feature-graph arrays: feat_ids.npy, mags.npy, n_active.npy,
doc_offsets.npy, doc_ids.npy, plus meta.json.

n_active.npy is written alongside the other four because the mounted SAE
has 65536 features, so every uint16 value in feat_ids is a legitimate
feature id - padding beyond n_active[i] is zero-filled but not otherwise
distinguishable from a real hit on feature 0. Consumers must slice
feat_ids[i, :n_active[i]] / mags[i, :n_active[i]].

Each rank owns a contiguous, order-preserving block of source documents
(see _common.partition_bounds), and each shard's `doc_offsets`/`doc_ids`
are at *document* granularity (one entry per kept source doc, regardless of
how many overlapping segments it was split into - see
collect_feature_activations.py's module docstring). Concatenating ranks
0..world_size-1 in order, and shards within a rank in manifest order, is
therefore already a globally doc_id-sorted merge; this script re-verifies
that rather than assuming it.

doc_offsets are recomputed with a cumulative token offset across the whole
merge (the per-shard offsets are only locally valid). Validation: the
merged array must have len(doc_offsets) == n_docs + 1, be strictly
monotonic, and doc_ids must be duplicate-free.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "interpret_features"))
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)

from corpus import CorpusSearchIndex  # noqa: E402

from _common import partition_bounds, sha256_of_file  # noqa: E402


def discover_ranks(shards_dir):
    ranks = []
    for name in sorted(os.listdir(shards_dir)):
        full = os.path.join(shards_dir, name)
        if name.startswith("rank") and os.path.isdir(full):
            try:
                ranks.append(int(name[len("rank"):]))
            except ValueError:
                continue
    return sorted(ranks)


def load_manifest_entries(rank_dir, rank):
    path = os.path.join(rank_dir, f"manifest_rank{rank}.jsonl")
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    entries.sort(key=lambda e: e["doc_id_range"][0])
    return entries


def check_coverage(entries, rank_start, rank_end, rank):
    cursor = rank_start
    for e in entries:
        s, ed = e["doc_id_range"]
        if s != cursor:
            raise RuntimeError(
                f"rank {rank}: gap/overlap in shard coverage at doc {cursor} "
                f"(next shard starts at {s}); finish collection before merging "
                f"(or pass --allow-incomplete to merge a partial run)."
            )
        cursor = ed
    if cursor != rank_end:
        raise RuntimeError(
            f"rank {rank}: incomplete shard coverage, reached doc {cursor} of {rank_end}; "
            f"finish collection before merging (or pass --allow-incomplete)."
        )


def git_commit(repo_dir):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def run_merge(shards_dir, out_dir, allow_incomplete=False, quiet=False):
    """Merge all shards/rank{r}/*.npz under shards_dir into out_dir. Returns
    the meta dict on success (also written to out_dir/meta.json)."""
    log = (lambda *a, **k: None) if quiet else print

    ranks = discover_ranks(shards_dir)
    if not ranks:
        raise RuntimeError(f"No rank directories found under {shards_dir}")

    run_configs = {}
    manifests = {}
    for r in ranks:
        rank_dir = os.path.join(shards_dir, f"rank{r}")
        cfg_path = os.path.join(rank_dir, "run_config.json")
        if not os.path.exists(cfg_path):
            raise RuntimeError(f"Missing run_config.json for rank {r} at {cfg_path}")
        with open(cfg_path, "r", encoding="utf8") as f:
            run_configs[r] = json.load(f)
        manifests[r] = load_manifest_entries(rank_dir, r)

    reference = {k: v for k, v in run_configs[ranks[0]].items() if k != "rank"}
    for r in ranks[1:]:
        cur = {k: v for k, v in run_configs[r].items() if k != "rank"}
        if cur != reference:
            diff = {k: (reference.get(k), cur.get(k)) for k in reference if reference.get(k) != cur.get(k)}
            raise RuntimeError(f"run_config.json mismatch between rank {ranks[0]} and rank {r}: {diff}")

    world_size = reference["world_size"]
    if sorted(ranks) != list(range(world_size)):
        raise RuntimeError(f"Expected rank dirs 0..{world_size - 1}, found {ranks}")

    data_path = reference["data_path"]
    n_source_lines = len(CorpusSearchIndex(data_path, cache_freq=1000, sampling=None))
    bounds = partition_bounds(n_source_lines, world_size)

    ordered_entries = []
    for r in ranks:
        rank_start, rank_end = bounds[r]
        entries = manifests[r]
        if not allow_incomplete:
            check_coverage(entries, rank_start, rank_end, r)
        rank_dir = os.path.join(shards_dir, f"rank{r}")
        for e in entries:
            ordered_entries.append((r, rank_dir, e))

    feat_chunks, mag_chunks, active_chunks = [], [], []
    doc_id_chunks, seg_chunks, length_chunks = [], [], []
    wall_clock_by_rank = {r: 0.0 for r in ranks}

    for rank, rank_dir, entry in ordered_entries:
        shard_path = entry["shard_path"]
        if not os.path.isabs(shard_path):
            shard_path = os.path.join(rank_dir, shard_path)

        actual_hash = sha256_of_file(shard_path)
        if actual_hash != entry["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {shard_path}: manifest={entry['sha256']} actual={actual_hash}")

        d = np.load(shard_path)
        local_offsets = d["doc_offsets"]
        local_doc_ids = d["doc_ids"]
        n_segments = d["n_segments"]

        if len(local_offsets) != len(local_doc_ids) + 1:
            raise RuntimeError(f"{shard_path}: len(doc_offsets) != len(doc_ids) + 1")
        if len(local_offsets) > 1 and np.any(np.diff(local_offsets) <= 0):
            raise RuntimeError(f"{shard_path}: doc_offsets is not strictly monotonic")
        if int(local_offsets[-1]) != d["feat_ids"].shape[0]:
            raise RuntimeError(f"{shard_path}: doc_offsets[-1] != n_tokens in feat_ids")

        feat_chunks.append(d["feat_ids"])
        mag_chunks.append(d["mags"])
        active_chunks.append(d["n_active"])
        doc_id_chunks.append(local_doc_ids)
        seg_chunks.append(n_segments)
        length_chunks.append(np.diff(local_offsets))
        wall_clock_by_rank[rank] += float(entry.get("wall_clock_seconds", 0.0))

    feat_ids_all = np.concatenate(feat_chunks, axis=0) if feat_chunks else np.zeros((0, 20), dtype=np.uint16)
    mags_all = np.concatenate(mag_chunks, axis=0) if mag_chunks else np.zeros((0, 20), dtype=np.float16)
    n_active_all = np.concatenate(active_chunks, axis=0) if active_chunks else np.zeros((0,), dtype=np.uint8)
    doc_ids_all = np.concatenate(doc_id_chunks, axis=0) if doc_id_chunks else np.zeros((0,), dtype=np.int64)
    n_segments_all = np.concatenate(seg_chunks, axis=0) if seg_chunks else np.zeros((0,), dtype=np.int16)
    lengths_all = np.concatenate(length_chunks, axis=0) if length_chunks else np.zeros((0,), dtype=np.int64)

    if len(np.unique(doc_ids_all)) != len(doc_ids_all):
        raise RuntimeError("Duplicate doc_id detected across merged shards.")
    if len(doc_ids_all) > 1 and np.any(np.diff(doc_ids_all) <= 0):
        raise RuntimeError(
            "doc_ids are not strictly increasing after merge; this should be impossible given "
            "contiguous per-rank partitioning and in-order shard concatenation - investigate."
        )

    n_docs = len(doc_ids_all)
    n_tokens = int(feat_ids_all.shape[0])

    doc_offsets_all = np.zeros(n_docs + 1, dtype=np.int64)
    if n_docs:
        np.cumsum(lengths_all, out=doc_offsets_all[1:])
    if len(doc_offsets_all) != n_docs + 1:
        raise RuntimeError("len(doc_offsets) != n_docs + 1")
    if n_docs and np.any(np.diff(doc_offsets_all) <= 0):
        raise RuntimeError("merged doc_offsets is not strictly monotonic")
    if int(doc_offsets_all[-1]) != n_tokens:
        raise RuntimeError("final doc_offsets entry != n_tokens")

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "feat_ids.npy"), feat_ids_all)
    np.save(os.path.join(out_dir, "mags.npy"), mags_all)
    np.save(os.path.join(out_dir, "n_active.npy"), n_active_all)
    np.save(os.path.join(out_dir, "doc_offsets.npy"), doc_offsets_all)
    np.save(os.path.join(out_dir, "doc_ids.npy"), doc_ids_all)

    frac_docs_split = float((n_segments_all > 1).mean()) if n_docs else 0.0
    repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

    meta = {
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "n_source_lines": n_source_lines,
        "data_path": data_path,
        "model_id": reference["model_id"],
        "model_key": reference["model_key"],
        "sae_path": reference["sae_path"],
        "sae_name": reference["sae_name"],
        "layer": reference["layer"],
        "topk": reference["topk"],
        "masking_logic": reference["masking_logic"],
        "padding_scheme": (
            "feat_ids/mags rows are zero-padded past n_active[i] (0..20); "
            "the SAE has 65536 features so no uint16 sentinel is available - "
            "always slice with n_active, not by scanning for a marker value."
        ),
        "split_params": {
            "max_seq_len": reference["max_seq_len"],
            "stride": reference["stride"],
            "overlap": reference["max_seq_len"] - reference["stride"],
            "min_content_tokens": reference["min_content_tokens"],
        },
        "frac_docs_split": frac_docs_split,
        "docs_per_shard": reference["docs_per_shard"],
        "world_size": world_size,
        "gpu_wall_clock_seconds_by_rank": wall_clock_by_rank,
        "gpu_wall_clock_seconds_total": sum(wall_clock_by_rank.values()),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit(repo_root),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    log(f"Merged {n_docs} docs / {n_tokens} tokens from {len(ordered_entries)} shards across {len(ranks)} ranks.")
    log(f"Wrote {out_dir}/{{feat_ids,mags,n_active,doc_offsets,doc_ids}}.npy + meta.json")
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Merge feature_graph shards into flat feat_ids/mags/doc_offsets/doc_ids arrays.",
    )
    parser.add_argument("--shards-dir", type=str, default=os.path.join(SCRIPT_DIR, "shards"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(SCRIPT_DIR, "merged"))
    parser.add_argument("--allow-incomplete", action="store_true",
                         help="Skip per-rank coverage validation and merge whatever has been collected so far")
    args = parser.parse_args()
    run_merge(args.shards_dir, args.out_dir, allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
