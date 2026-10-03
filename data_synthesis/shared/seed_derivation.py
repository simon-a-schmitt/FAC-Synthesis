"""Derives a deterministic seed from the content of a seed group, shared by all generation arms
(blackbox/, feature_guided/, hybrid/) so that every arm gets the same seed for the same group.

  1. Each seed example is canonicalized: Unicode NFC, line breaks unified to "\\n", leading and
     trailing whitespace stripped. Nothing else (no lowercasing, no cleanup) - the text is to be
     identified, not changed.
  2. Each canonical example is hashed on its own (SHA-256).
  3. The per-example hashes are sorted, so the group is treated as a set and its order in the
     seed file does not matter.
  4. The final SHA-256 is taken over benchmark + the sorted per-example hashes + purpose.

The examples are read straight from the seed TSV (first column, CSV-unquoted) rather than taken
from an arm's own loader: the arms clean the texts differently for their prompts, and the seed must
not depend on that.
"""

from __future__ import annotations

import csv
import hashlib
import unicodedata
from pathlib import Path

BENCHMARKS = ("toxicity_detection", "claudette_tos", "cti_vsp")

PURPOSE_GENERATION = "generation_seed"
# Not used yet; reserved for seeding fine-tuning runs on the same seed group.
PURPOSE_FINE_TUNING = "fine_tuning_seed"
# Seeds the sampling of the gold fine-tuning set that belongs to a seed group.
PURPOSE_GOLD_SAMPLING = "gold_sampling_seed"
PURPOSES = (PURPOSE_GENERATION, PURPOSE_FINE_TUNING, PURPOSE_GOLD_SAMPLING)


def canonicalize_example(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def hash_example(text: str) -> str:
    return hashlib.sha256(canonicalize_example(text).encode("utf-8")).hexdigest()


def derive_seed_digest(benchmark: str, seed_examples: list[str], purpose: str) -> str:
    """Hex SHA-256 identifying (benchmark, seed group as a set, purpose)."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark!r}; expected one of {BENCHMARKS}.")
    if purpose not in PURPOSES:
        raise ValueError(f"Unknown purpose {purpose!r}; expected one of {PURPOSES}.")
    if not seed_examples:
        raise ValueError("Cannot derive a seed from an empty seed group.")
    example_hashes = sorted(hash_example(text) for text in seed_examples)
    # Fields contain no newlines (fixed names, hex digests), so this joining is unambiguous.
    payload = "\n".join([benchmark, *example_hashes, purpose])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_seed(benchmark: str, seed_examples: list[str], purpose: str) -> int:
    """Seed as an unsigned 32-bit int (first 8 hex digits of the digest), so it is accepted by
    random.Random as well as numpy / torch / transformers.set_seed."""
    return int(derive_seed_digest(benchmark, seed_examples, purpose)[:8], 16)


def load_seed_texts(seed_file: Path) -> list[str]:
    """Raw seed texts (first column) of a seed TSV, only CSV-unquoted, otherwise unchanged."""
    texts = []
    with open(seed_file, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and canonicalize_example(row[0]):
                texts.append(row[0])
    if not texts:
        raise ValueError(f"No seed examples found in {seed_file}")
    return texts


def derive_seed_from_file(benchmark: str, seed_file: Path, purpose: str) -> int:
    return derive_seed(benchmark, load_seed_texts(seed_file), purpose)
