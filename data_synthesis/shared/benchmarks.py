"""Benchmark data read by every generation arm and by labeling: seed groups and feature relevance
scores, both loaded straight from benchmarks/<domain>/ (the single source of truth - nothing is
copied into data_synthesis/).

  benchmarks/<domain>/seed_groups/<name>_seed_group_<id>.tsv          --seed-set k5 (default)
  benchmarks/<domain>/seed_groups_k10/<name>_seed_group_k10_<id>.tsv  --seed-set k10
  benchmarks/<domain>/feature_scores/<domain>_feature_relevance_scores.jsonl
"""

from __future__ import annotations

import csv
from pathlib import Path

from shared.text_cleaning import clean_seed_text

DATA_SYNTHESIS_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = DATA_SYNTHESIS_DIR.parent
BENCHMARKS_DIR = PROJECT_DIR / "benchmarks"

DOMAINS = ("claudette_tos", "cti_vsp", "toxicity_detection")
SEED_GROUPS = ("01", "02", "03", "04", "05")
# --seed-set -> (directory under benchmarks/<domain>/, infix of the file name before the group id)
SEED_SETS = {
    "k5": ("seed_groups", ""),
    "k10": ("seed_groups_k10", "k10_"),
}
DEFAULT_SEED_SET = "k5"


def find_seed_file(domain: str, seed_set: str, seed_group: str) -> Path:
    dirname, infix = SEED_SETS[seed_set]
    seed_dir = BENCHMARKS_DIR / domain / dirname
    pattern = f"*_seed_group_{infix}{seed_group}.tsv"
    matches = sorted(seed_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one seed file matching '{pattern}' in {seed_dir}, found {matches}.")
    return matches[0]


def find_seed_file_by_name(domain: str, file_name: str) -> Path | None:
    """The seed file of any seed set with this file name (for logs that only recorded a path)."""
    for dirname, _ in SEED_SETS.values():
        candidate = BENCHMARKS_DIR / domain / dirname / file_name
        if candidate.exists():
            return candidate
    return None


def load_seed_labels(seed_file: Path) -> list[str]:
    """Labels (second column) of the seed rows, aligned index by index with
    shared.text_cleaning.load_seed_examples() (same rows skipped)."""
    labels = []
    with open(seed_file, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and clean_seed_text(row[0]):
                labels.append(row[1].strip() if len(row) > 1 else "")
    return labels


def load_raw_seed_texts(seed_file: Path) -> list[str]:
    """Seed texts as stored in the TSV (CSV-unquoted, stripped, otherwise unchanged) - what the SAE
    coverage check runs on, as active_feature_identification/get_active_features does."""
    with open(seed_file, "r", encoding="utf-8", newline="") as f:
        return [row[0].strip() for row in csv.reader(f, delimiter="\t") if row and row[0].strip()]


def feature_scores_path(domain: str) -> Path:
    return BENCHMARKS_DIR / domain / "feature_scores" / f"{domain}_feature_relevance_scores.jsonl"
