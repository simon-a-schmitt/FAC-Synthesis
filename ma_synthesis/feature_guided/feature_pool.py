"""Loads the feature-guided seed file and samples feature pairs for step 0.

The seed file (data/seeds/feature_seed_casehold.jsonl) holds one
`subdomain_features` line per seed example (5 total). Each line carries
that example's features sorted by log-odds (most characteristic first).
We take the top TOP_K_PER_SEED features per seed example, giving a flat
pool of 5 * TOP_K_PER_SEED entries tagged with their originating seed
example, and draw two features per generation from two different seed
examples.
"""

import json
import os
import random
from pathlib import Path

TOP_K_PER_SEED = 10

# See blackbox/step1.py: `benchmark_play_ground.model_wrapper` fixes the
# global `random` module's seed at import time, so a private RNG seeded
# from OS entropy is used here too, to get different feature draws across
# separate script invocations (i.e. across runs of the same dataset).
_rng = random.Random(os.urandom(16))


def _clean_token(token: str) -> str:
    # SAE feature tokens use the GPT-2/BPE 'Ġ' marker for a leading space.
    return token.replace("Ġ", " ").strip()


def _format_top_tokens(top_tokens: list) -> str:
    words = [_clean_token(tok) for _, tok, _ in top_tokens]
    return ", ".join(words)


def _format_reference_spans(descriptions: list) -> str:
    text = descriptions[0] if descriptions else ""
    return text.replace("\\n", "\n").strip()


def load_feature_pool(path: Path) -> list[dict]:
    """Build the flat pool of (seed_index, feature) entries from the seed file.

    Only `subdomain_features` lines are used (the `template_features` line
    is unrelated generic data and is skipped). Each pool entry has keys
    "seed_index", "feature_id", "top_tokens" and "reference_spans", the
    latter two already rendered as plain text for the step-0 prompt.
    """
    pool: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") != "subdomain_features":
                continue
            seed_index = entry["example_index"]
            for feature in entry["features"][:TOP_K_PER_SEED]:
                pool.append({
                    "seed_index": seed_index,
                    "feature_id": feature["feature_id"],
                    "top_tokens": _format_top_tokens(feature["top_tokens"]),
                    "reference_spans": _format_reference_spans(feature["descriptions"]),
                })
    return pool


def sample_two_features(pool: list[dict]) -> tuple[dict, dict]:
    """Draw two pool entries from two distinct seed examples."""
    seed_indices = sorted({entry["seed_index"] for entry in pool})
    seed_a, seed_b = _rng.sample(seed_indices, 2)
    feature_a = _rng.choice([entry for entry in pool if entry["seed_index"] == seed_a])
    feature_b = _rng.choice([entry for entry in pool if entry["seed_index"] == seed_b])
    return feature_a, feature_b
