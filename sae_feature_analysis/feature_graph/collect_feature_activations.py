"""Collect per-token SAE feature activations over fineweb_edu for the feature graph.

Model: Llama-3.1-8B-Instruct + TopK SAE (layer 16, top-k=20), run in the
`fac_env` environment (transformers==4.43.4). Special-token masking reuses
the exact helpers from
`sae_feature_analysis/interpret_features_exp/collect_reservoir_samples.py`
(`_special_token_id_set`, `_is_newline_only_token`) but drops all
reservoir/top-token bookkeeping: only raw (feat_id, magnitude) pairs for
content tokens are written to disk.

Documents are never truncated. Docs with < 32 content tokens are skipped.
Docs with <= 4096 content tokens are processed whole; longer docs are split
into overlapping segments (window 4096, stride 3584, overlap 512) that are
each run through the model separately and then concatenated back into a
single per-document token block (so overlapping tokens legitimately appear
twice, once per context window - that's the point of the overlap).

Multi-GPU: launch with `torchrun --nproc_per_node=<N> collect_feature_activations.py
...` (or via SLURM/srun - RANK/WORLD_SIZE/LOCAL_RANK are read from the
environment, with SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID and --rank/
--world-size/--local-rank as fallbacks). The document list is split into
`world_size` contiguous, order-preserving blocks (NOT `docs[r::world_size]`)
so each rank's shards stay sorted and the merge step is a plain
concatenation. Each rank writes only under its own `shards/rank{r}/`
directory - no cross-rank IO.

Shard format (`shards/rank{r}/shard_{r}_{seq:04d}.npz`), one shard per
`--docs-per-shard` (default 250) consecutive source documents:
  feat_ids    uint16  [n_tok, 20]  active feature ids per content token,
                                    descending by magnitude, zero-padded
  mags        float16 [n_tok, 20]  matching activation magnitudes, zero-padded
  n_active    uint8   [n_tok]      number of real (non-padding) entries in
                                    feat_ids[i]/mags[i] (0..20). The mounted
                                    SAE has 65536 features, so every uint16
                                    value is a legitimate feature id - there
                                    is no spare sentinel to mark padding, and
                                    feat_ids[i, n_active[i]:] must be treated
                                    as padding regardless of its value.
  doc_offsets int64   [n_docs+1]   local token offsets delimiting documents
  doc_ids     int64   [n_docs]     source line index (0-based) of each doc
  n_segments  int16   [n_docs]     number of overlapping segments merged
                                    into that doc's block (>1 => was split)

Resumability: completed shards are recorded in
`shards/rank{r}/manifest_rank{r}.jsonl` (one JSON line per shard: path,
doc_id_range, n_tokens, sha256, wall_clock_seconds, ...). On startup the
manifest is read and already-completed shards are skipped. Shards are
written atomically (tmp file + os.replace) so a killed job never leaves a
half-written shard behind.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time


def _preresolve_rank_env():
    """Determine rank/world_size/local_rank and pin CUDA_VISIBLE_DEVICES
    *before* anything below imports torch/transformers. Several of our
    imports (generator.py -> transformers.set_seed) touch CUDA as a side
    effect of import, which permanently locks in device visibility for the
    process - setting the env var any later (e.g. inside __main__, after
    those imports already ran) would silently no-op."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--rank", type=int, default=None)
    pre.add_argument("--world-size", type=int, default=None)
    pre.add_argument("--local-rank", type=int, default=None)
    pre.add_argument("--device", type=str, default="cuda")
    known, _ = pre.parse_known_args()

    rank = known.rank if known.rank is not None else int(
        os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0))
    )
    world_size = known.world_size if known.world_size is not None else int(
        os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1))
    )
    local_rank = known.local_rank if known.local_rank is not None else int(
        os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", rank))
    )
    if not (0 <= rank < max(world_size, 1)):
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")

    if known.device.lower() == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return rank, world_size, local_rank


RANK, WORLD_SIZE, LOCAL_RANK = _preresolve_rank_env()

import numpy as np  # noqa: E402
import torch as tc  # noqa: E402
import tqdm  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTERPRET_EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "interpret_features_exp"))
LEGACY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "interpret_features"))
for _p in (INTERPRET_EXP_DIR, LEGACY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from corpus import CorpusSearchIndex  # noqa: E402
from generator import Generator  # noqa: E402
from llm_surgery import mount_function, switch_mode  # noqa: E402
from autoencoder import load_pretrained  # noqa: E402
from collect_reservoir_samples import (  # noqa: E402
    _special_token_id_set,
    _is_newline_only_token,
    resolve_runtime_device,
)

from _common import MAX_FEATURE_ID, partition_bounds, chunk_ranges  # noqa: E402

MASKING_LOGIC_DESC = (
    "content tokens only: mask = (token in [content_start, content_end)) "
    "and (token_id not in special_ids); special_ids and the leading-blank "
    "trim reuse _special_token_id_set/_is_newline_only_token from "
    "interpret_features_exp/collect_reservoir_samples.py; content_start/"
    "content_end are exact (derived from chat-template prefix/suffix ids "
    "spliced around each segment), not marker-searched."
)


# ========== segmentation & chat-template splicing ==========

def make_segments(n_content, max_seq_len, stride):
    """Split [0, n_content) into overlapping (start, end) windows.

    A single window covering the whole range is returned when it already
    fits within max_seq_len. Otherwise windows of length max_seq_len are
    taken every `stride` tokens; the final window is clipped to n_content
    (so it may be shorter than max_seq_len, but is always > overlap tokens
    given the default max_seq_len/stride)."""
    if n_content <= max_seq_len:
        return [(0, n_content)]
    segments = []
    pos = 0
    while True:
        end = min(pos + max_seq_len, n_content)
        segments.append((pos, end))
        if end == n_content:
            break
        pos += stride
    return segments


def _trim_leading_blank_tokens(token_ids, tokenizer, probe=64):
    """Drop a leading run of newline-only tokens, analogous to how
    collect_reservoir_samples skips blank lines right after the chat
    template's user-header marker before content officially starts."""
    if not token_ids:
        return token_ids
    tokens = tokenizer.convert_ids_to_tokens(token_ids[:probe])
    cut = 0
    while cut < len(tokens) and _is_newline_only_token(tokens[cut]):
        cut += 1
    return token_ids[cut:]


def _template_ids(tokenizer, content):
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def build_template_affixes(tokenizer):
    """Determine the exact fixed token ids surrounding user content in the
    chat template, by comparing the template rendered with empty content
    against the template rendered with a sentinel. This is exact (no marker
    search, no decode/re-encode of segment text) because added special
    tokens (e.g. <|eot_id|>) are split from surrounding text by the
    tokenizer's trie *before* BPE, so no merge can occur across that
    boundary regardless of what content precedes it."""
    sentinel = "XFEATUREGRAPHSENTINELX"
    ids_empty = _template_ids(tokenizer, "")
    ids_sent = _template_ids(tokenizer, sentinel)

    prefix_len = 0
    while (
        prefix_len < len(ids_empty)
        and prefix_len < len(ids_sent)
        and ids_empty[prefix_len] == ids_sent[prefix_len]
    ):
        prefix_len += 1

    suffix_len = 0
    while (
        suffix_len < len(ids_empty) - prefix_len
        and suffix_len < len(ids_sent) - prefix_len
        and ids_empty[len(ids_empty) - 1 - suffix_len] == ids_sent[len(ids_sent) - 1 - suffix_len]
    ):
        suffix_len += 1

    prefix_ids = ids_sent[:prefix_len]
    suffix_ids = ids_sent[len(ids_sent) - suffix_len:]
    if prefix_ids + suffix_ids != ids_empty:
        raise RuntimeError(
            "Could not robustly determine chat-template prefix/suffix token ids "
            "(empty-content reconstruction mismatch); refusing to guess."
        )
    return prefix_ids, suffix_ids


# ========== activation extraction ==========

def compute_segment_features(full_ids, content_start, content_end, model, sae, special_ids):
    """Run one segment through the model and return (feat_ids, mags,
    n_active) for its content tokens only (uint16 [n_tok,20], float16
    [n_tok,20], uint8 [n_tok]), already filtered to drop any position whose
    token id is a special token id. Padding slots (beyond n_active[i]) are
    zero-filled but are not distinguishable from a real feature-id-0 hit by
    value alone - callers must slice on n_active. Returns (None, None, None)
    if no content tokens survive filtering."""
    ids = tc.tensor(full_ids, dtype=tc.long, device=model._device)

    try:
        with tc.no_grad():
            model.get_activates(ids)
    except RuntimeError:
        pass

    token_actvs = sae.actvs.squeeze()
    seg_actvs = token_actvs[content_start:content_end]

    seg_token_ids = full_ids[content_start:content_end]
    keep = tc.tensor(
        [tok_id not in special_ids for tok_id in seg_token_ids],
        dtype=tc.bool,
        device=seg_actvs.device,
    )
    if int(keep.sum().item()) == 0:
        return None, None, None
    seg_actvs = seg_actvs[keep]

    vals, idxs = tc.topk(seg_actvs, k=20, dim=-1)
    active = vals > 0
    idxs_out = tc.where(active, idxs, tc.zeros_like(idxs))
    vals_out = tc.where(active, vals, tc.zeros_like(vals))

    feat_ids = idxs_out.to(device="cpu", dtype=tc.int64).numpy().astype(np.uint16)
    mags = vals_out.to(device="cpu", dtype=tc.float32).numpy().astype(np.float16)
    n_active = active.sum(dim=-1).to(device="cpu", dtype=tc.int64).numpy().astype(np.uint8)
    return feat_ids, mags, n_active


def process_doc(text, tokenizer, model, sae, special_ids, prefix_ids, suffix_ids,
                 min_content_tokens, max_seq_len, stride):
    """Returns (feat_ids [n_tok,20] uint16, mags [n_tok,20] float16,
    n_active [n_tok] uint8, n_segments int) for one source document, or None
    if it was skipped (too short, or nothing survived special-token
    filtering)."""
    content_ids = tokenizer.encode(text, add_special_tokens=False)
    content_ids = _trim_leading_blank_tokens(content_ids, tokenizer)
    if len(content_ids) < min_content_tokens:
        return None

    segments = make_segments(len(content_ids), max_seq_len, stride)

    feat_parts, mag_parts, active_parts = [], [], []
    for (s, e) in segments:
        seg_ids = content_ids[s:e]
        full_ids = prefix_ids + seg_ids + suffix_ids
        content_start = len(prefix_ids)
        content_end = content_start + len(seg_ids)
        feat_ids, mags, n_active = compute_segment_features(
            full_ids, content_start, content_end, model, sae, special_ids
        )
        if feat_ids is not None and feat_ids.shape[0] > 0:
            feat_parts.append(feat_ids)
            mag_parts.append(mags)
            active_parts.append(n_active)

    if not feat_parts:
        return None

    doc_feat = np.concatenate(feat_parts, axis=0)
    doc_mag = np.concatenate(mag_parts, axis=0)
    doc_active = np.concatenate(active_parts, axis=0)
    return doc_feat, doc_mag, doc_active, len(segments)


# ========== shard writing / manifest / resumability ==========

def atomic_write_npz(path, **arrays):
    root = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_shard_", suffix=".npz", dir=root)
    os.close(fd)
    try:
        np.savez(tmp_path, **arrays)
        with open(tmp_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        os.replace(tmp_path, path)
        return digest
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_manifest(manifest_path):
    """Returns {doc_id_range_start: entry} for shards on disk with a
    matching manifest entry (missing shard files are treated as
    not-completed so they get reprocessed)."""
    completed = {}
    if not os.path.exists(manifest_path):
        return completed
    rank_dir = os.path.dirname(manifest_path)
    with open(manifest_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            shard_path = entry["shard_path"]
            if not os.path.isabs(shard_path):
                shard_path = os.path.join(rank_dir, shard_path)
            if not os.path.exists(shard_path):
                print(f"[WARN] Manifest references missing shard {shard_path}; will reprocess.")
                continue
            completed[entry["doc_id_range"][0]] = entry
    return completed


def append_manifest(manifest_path, entry):
    with open(manifest_path, "a", encoding="utf8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_or_validate_run_config(rank_dir, config, has_existing_progress):
    path = os.path.join(rank_dir, "run_config.json")
    check_keys = (
        "data_path", "sae_path", "layer", "topk", "max_seq_len", "stride",
        "min_content_tokens", "docs_per_shard", "world_size",
    )
    if os.path.exists(path):
        with open(path, "r", encoding="utf8") as f:
            existing = json.load(f)
        mismatch = {k: (existing.get(k), config[k]) for k in check_keys if existing.get(k) != config[k]}
        if mismatch and has_existing_progress:
            raise RuntimeError(
                f"run_config.json mismatch with existing progress for rank {config['rank']}: {mismatch}"
            )
        if mismatch:
            print(f"[WARN] Overwriting stale run_config.json (no progress recorded yet): {mismatch}")
    with open(path, "w", encoding="utf8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def process_shard(rank, seq, doc_range, corpus, tokenizer, model, sae, special_ids,
                   prefix_ids, suffix_ids, min_content_tokens, max_seq_len, stride,
                   rank_dir):
    start_t = time.perf_counter()

    feat_chunks, mag_chunks, active_chunks = [], [], []
    lengths, kept_doc_ids, seg_counts = [], [], []
    for doc_id in range(*doc_range):
        text = corpus[doc_id]
        result = process_doc(
            text, tokenizer, model, sae, special_ids, prefix_ids, suffix_ids,
            min_content_tokens, max_seq_len, stride,
        )
        if result is None:
            continue
        doc_feat, doc_mag, doc_active, n_seg = result
        feat_chunks.append(doc_feat)
        mag_chunks.append(doc_mag)
        active_chunks.append(doc_active)
        lengths.append(doc_feat.shape[0])
        kept_doc_ids.append(doc_id)
        seg_counts.append(n_seg)

    wall_clock = time.perf_counter() - start_t

    feat_ids_arr = np.concatenate(feat_chunks, axis=0) if feat_chunks else np.zeros((0, 20), dtype=np.uint16)
    mags_arr = np.concatenate(mag_chunks, axis=0) if mag_chunks else np.zeros((0, 20), dtype=np.float16)
    n_active_arr = np.concatenate(active_chunks, axis=0) if active_chunks else np.zeros((0,), dtype=np.uint8)
    doc_offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    if lengths:
        np.cumsum(np.array(lengths, dtype=np.int64), out=doc_offsets[1:])
    doc_ids_arr = np.array(kept_doc_ids, dtype=np.int64)
    n_segments_arr = np.array(seg_counts, dtype=np.int16)

    shard_name = f"shard_{rank}_{seq:04d}.npz"
    shard_path = os.path.join(rank_dir, shard_name)
    digest = atomic_write_npz(
        shard_path,
        feat_ids=feat_ids_arr,
        mags=mags_arr,
        n_active=n_active_arr,
        doc_offsets=doc_offsets,
        doc_ids=doc_ids_arr,
        n_segments=n_segments_arr,
    )

    return {
        "shard_path": shard_name,
        "seq": seq,
        "doc_id_range": [doc_range[0], doc_range[1]],
        "n_docs_written": len(kept_doc_ids),
        "n_tokens": int(feat_ids_arr.shape[0]),
        "sha256": digest,
        "wall_clock_seconds": wall_clock,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_rank(rank, docs_per_shard, corpus, tokenizer, model, sae, special_ids,
             prefix_ids, suffix_ids, min_content_tokens, max_seq_len, stride,
             rank_start, rank_end, rank_dir, manifest_path, completed):
    chunk_bounds = list(chunk_ranges(rank_start, rank_end, docs_per_shard))
    bar = tqdm.tqdm(chunk_bounds, desc=f"[rank {rank}] shards")
    for seq, (s, e) in enumerate(chunk_bounds):
        if s in completed:
            bar.update(1)
            continue
        entry = process_shard(
            rank, seq, (s, e), corpus, tokenizer, model, sae, special_ids,
            prefix_ids, suffix_ids, min_content_tokens, max_seq_len, stride, rank_dir,
        )
        append_manifest(manifest_path, entry)
        completed[s] = entry
        bar.update(1)


if __name__ == "__main__":
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    DEFAULT_DATA_PATH = os.path.abspath(os.path.join(
        SCRIPT_DIR, "..", "..", "feature_annotation_data", "fineweb_edu", "fineweb_edu_sample.txt",
    ))

    parser = argparse.ArgumentParser(
        description="Collect per-token SAE feature activations over fineweb_edu for the feature graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="Path to fineweb_edu text file")
    parser.add_argument("--sae-path", type=str, required=True, help="SAE checkpoint path")
    parser.add_argument("--sae-layer", type=int, default=None, help="Override SAE layer ID (default: from checkpoint name)")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device: 'cpu' or 'cuda'")
    parser.add_argument("--out-dir", type=str, default=os.path.join(SCRIPT_DIR, "shards"), help="Root shard output directory")
    parser.add_argument("--docs-per-shard", type=int, default=250, help="Source documents per shard file")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Segment length in content tokens")
    parser.add_argument("--stride", type=int, default=3584, help="Segment stride in content tokens (overlap = max-seq-len - stride)")
    parser.add_argument("--min-content-tokens", type=int, default=32, help="Skip documents with fewer content tokens than this")
    parser.add_argument("--topk", type=int, default=20, help="SAE top-k override while collecting")
    # --rank/--world-size/--local-rank/--device are already consumed by
    # _preresolve_rank_env() above; redeclared here only so --help documents
    # them and so args.rank/etc. are available below for logging/validation.
    parser.add_argument("--rank", type=int, default=None, help="Overrides RANK/SLURM_PROCID env var")
    parser.add_argument("--world-size", type=int, default=None, help="Overrides WORLD_SIZE/SLURM_NTASKS env var")
    parser.add_argument("--local-rank", type=int, default=None, help="Overrides LOCAL_RANK/SLURM_LOCALID env var (GPU pinning)")
    parser.add_argument("--resume", type=str, default="auto", choices=["auto", "never", "require"], help="Resume mode")
    parser.add_argument("--enable-slurm", action="store_true", help="Enable SLURM integration logging")
    args = parser.parse_args()

    if args.max_seq_len <= 0 or args.stride <= 0 or args.stride > args.max_seq_len:
        raise ValueError("--stride must be in (0, --max-seq-len]")
    if args.docs_per_shard <= 0:
        raise ValueError("--docs-per-shard must be > 0")
    if args.min_content_tokens <= 0:
        raise ValueError("--min-content-tokens must be > 0")

    rank, world_size, local_rank = RANK, WORLD_SIZE, LOCAL_RANK
    device = resolve_runtime_device(args.device.lower())

    logger.info(f"Rank {rank}/{world_size} (local_rank={local_rank}), device={device}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"SAE: {args.sae_path}")
    logger.info(f"Resume mode: {args.resume}")
    if args.enable_slurm:
        logger.info("SLURM Integration: Enabled")
        logger.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<auto>')}")
        logger.info(f"  SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', '<none>')}")
        logger.info(f"  SLURM_PROCID: {os.environ.get('SLURM_PROCID', '<none>')}")

    try:
        logger.info(f"Loading SAE from: {args.sae_path}")
        name, layer, sae = load_pretrained(args.sae_path, device=device)
        if args.sae_layer is not None:
            logger.info(f"Overriding layer: {layer} -> {args.sae_layer}")
            layer = args.sae_layer
        if sae.dims[1] > MAX_FEATURE_ID + 1:
            raise ValueError(
                f"SAE hidden dim {sae.dims[1]} exceeds {MAX_FEATURE_ID + 1}; feature ids "
                "would not fit in the uint16 feat_ids arrays."
            )

        model_key, model_ckpt = "llama", "llama3-8b"
        dtype = "float32" if device == "cpu" else "bfloat16"
        logger.info(f"Loading model '{model_ckpt}' in dtype={dtype}")
        generator = Generator(model_ckpt, device=device, dtype=dtype)
        tokenizer = generator._tokenizer

        logger.info(f"Mounting SAE to model layer {layer}")
        mount_function(generator._model, model_key, int(layer), sae)

        sae.eval()
        sae.MaskTopK = False
        generator._model.eval()
        switch_mode(sae, "train")
        sae.early_stop = True
        sae.topk = int(args.topk)

        logger.info("Deriving chat-template prefix/suffix token ids...")
        prefix_ids, suffix_ids = build_template_affixes(tokenizer)
        special_ids = _special_token_id_set(tokenizer)

        logger.info(f"Loading corpus from: {args.data_path}")
        corpus = CorpusSearchIndex(args.data_path, cache_freq=1000, sampling=None)
        n_docs_total = len(corpus)
        rank_start, rank_end = partition_bounds(n_docs_total, world_size)[rank]
        logger.info(f"Rank {rank} owns docs [{rank_start}, {rank_end}) of {n_docs_total} total")

        rank_dir = os.path.join(args.out_dir, f"rank{rank}")
        os.makedirs(rank_dir, exist_ok=True)
        manifest_path = os.path.join(rank_dir, f"manifest_rank{rank}.jsonl")

        if args.resume == "never" and os.path.exists(manifest_path):
            raise RuntimeError(
                f"--resume never but manifest already exists at {manifest_path}; "
                f"remove shards/rank{rank}/ manually if you want to restart from scratch."
            )
        completed = {} if args.resume == "never" else load_manifest(manifest_path)
        if args.resume == "require" and not os.path.exists(manifest_path):
            raise FileNotFoundError(f"--resume require but no manifest found: {manifest_path}")

        run_config = {
            "rank": rank,
            "world_size": world_size,
            "data_path": os.path.abspath(args.data_path),
            "sae_path": os.path.abspath(args.sae_path),
            "sae_name": name,
            "model_id": generator._name,
            "model_key": model_key,
            "layer": int(layer),
            "topk": int(args.topk),
            "max_seq_len": int(args.max_seq_len),
            "stride": int(args.stride),
            "min_content_tokens": int(args.min_content_tokens),
            "docs_per_shard": int(args.docs_per_shard),
            "masking_logic": MASKING_LOGIC_DESC,
        }
        write_or_validate_run_config(rank_dir, run_config, has_existing_progress=bool(completed))

        logger.info(f"Starting collection (rank {rank}/{world_size})...")
        with tc.no_grad():
            run_rank(
                rank, args.docs_per_shard, corpus, tokenizer, generator, sae, special_ids,
                prefix_ids, suffix_ids, args.min_content_tokens, args.max_seq_len, args.stride,
                rank_start, rank_end, rank_dir, manifest_path, completed,
            )

        logger.info(f"Rank {rank} completed successfully!")

    except Exception as exc:
        logger.error(f"Fatal error on rank {rank}:")
        logger.error(f"{type(exc).__name__}: {exc}")
        raise
