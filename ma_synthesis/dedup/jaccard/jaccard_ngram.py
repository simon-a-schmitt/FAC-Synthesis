"""Compute pairwise n-gram Jaccard and Overlap similarities for CaseHOLD examples.

Jaccard(A, B)  = |A ∩ B| / |A ∪ B|        -> shared share of the combined material
Overlap(A, B)  = |A ∩ B| / min(|A|, |B|)  -> shared share of the smaller document

Reads a TSV file with one example per line (text<TAB>label, no header) and
writes the full pairwise similarity matrices to a JSON file.
"""

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

DEFAULT_N = 5
NORMALIZE_VERSION = "v1-nfkc-casefold-word"

_WS = re.compile(r"\s+")
_TOK = re.compile(r"\w+", re.UNICODE)


def read_texts(tsv_path: Path, text_column: int) -> list[str]:
    with tsv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        return [row[text_column] for row in reader if row]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return _WS.sub(" ", text.casefold()).strip()


def tokenize(text: str) -> list[str]:
    return _TOK.findall(normalize(text))


def ngram_set(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard_and_overlap(a: set, b: set) -> tuple[float, float]:
    if not a and not b:
        return 0.0, 0.0
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    jaccard = intersection / union if union else 0.0
    smaller = min(len(a), len(b))
    overlap = intersection / smaller if smaller else 0.0
    return jaccard, overlap


def compute_matrices(ngram_sets: list[set]) -> tuple[list[list[float]], list[list[float]]]:
    num = len(ngram_sets)
    jaccard_matrix = [[0.0] * num for _ in range(num)]
    overlap_matrix = [[0.0] * num for _ in range(num)]
    for i in range(num):
        jaccard_matrix[i][i] = 1.0
        overlap_matrix[i][i] = 1.0

    start = time.monotonic()
    total_pairs = num * (num - 1) // 2
    done = 0
    for i in range(num):
        set_i = ngram_sets[i]
        for j in range(i + 1, num):
            jaccard, overlap = jaccard_and_overlap(set_i, ngram_sets[j])
            jaccard_matrix[i][j] = jaccard_matrix[j][i] = jaccard
            overlap_matrix[i][j] = overlap_matrix[j][i] = overlap
        done += num - i - 1
        if i % 50 == 0 or i == num - 1:
            elapsed = time.monotonic() - start
            print(f"  {done}/{total_pairs} pairs ({elapsed:.1f}s)", file=sys.stderr)

    return jaccard_matrix, overlap_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "casehold_jac_val.tsv",
        help="Path to the input TSV file (default: casehold_jac_val.tsv next to this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "casehold_jac_val_similarities.json",
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"n-gram size (default: {DEFAULT_N})",
    )
    parser.add_argument(
        "--text-column",
        type=int,
        default=0,
        help="Index of the TSV column that contains the example text (default: 0)",
    )
    args = parser.parse_args()

    print(f"Reading examples from {args.input}", file=sys.stderr)
    texts = read_texts(args.input, args.text_column)
    print(f"Loaded {len(texts)} examples, building {args.n}-gram sets", file=sys.stderr)

    ngram_sets = [ngram_set(tokenize(text), args.n) for text in texts]

    print("Computing pairwise Jaccard and Overlap similarities", file=sys.stderr)
    jaccard_matrix, overlap_matrix = compute_matrices(ngram_sets)

    result = {
        "n": args.n,
        "num_examples": len(texts),
        "normalize_version": NORMALIZE_VERSION,
        "jaccard": jaccard_matrix,
        "overlap": overlap_matrix,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f)

    print(f"Wrote similarity matrices to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
