"""Shared helpers for the feature_graph collection + merge pipeline."""

import hashlib

# Max feature id representable in the uint16 feat_ids arrays (inclusive).
# The mounted SAE's hidden dim must not exceed this + 1 (see
# collect_feature_activations.py, which asserts this at load time). Padding
# slots (a token has fewer than 20 active features) are NOT marked with a
# sentinel value - a hidden dim of 65536 makes every uint16 value a valid
# feature id, so there is no id left over to use as one. Instead each token
# carries an explicit n_active count (see shard/merge format docs); padding
# slots are zero-filled but must be ignored beyond n_active.
MAX_FEATURE_ID = 65535


def partition_bounds(n_items, world_size):
    """Split n_items into world_size contiguous, order-preserving blocks.

    Returns a list of (start, end) half-open index ranges, one per rank.
    Blocks differ in size by at most one element (the first `n_items %
    world_size` ranks get one extra item).
    """
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    base, rem = divmod(n_items, world_size)
    bounds = []
    start = 0
    for r in range(world_size):
        size = base + (1 if r < rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def chunk_ranges(start, end, size):
    """Yield contiguous (s, e) sub-ranges of [start, end) of length <= size."""
    s = start
    while s < end:
        yield (s, min(s + size, end))
        s += size


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
