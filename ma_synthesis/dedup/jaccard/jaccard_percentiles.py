"""Print percentile distributions of the pairwise Jaccard and Overlap similarities.

Reads the JSON output of jaccard_ngram.py (full symmetric similarity matrices)
and prints the requested percentiles for both metrics to the console.
"""

import argparse
import json
from pathlib import Path

PERCENTILES = list(range(1, 11)) + list(range(20, 91, 10)) + list(range(91, 100))


def upper_triangle(matrix: list[list[float]]) -> list[float]:
    n = len(matrix)
    return [matrix[i][j] for i in range(n) for j in range(i + 1, n)]


def percentile(sorted_values: list[float], p: float) -> float:
    k = (len(sorted_values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def print_percentiles(name: str, values: list[float]) -> None:
    sorted_values = sorted(values)
    print(f"\n{name} percentiles (n={len(sorted_values)} pairs):")
    for p in PERCENTILES:
        print(f"  {p:>3}%: {percentile(sorted_values, p):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "casehold_jac_val_similarities.json",
        help="Path to the similarity matrices JSON produced by jaccard_ngram.py",
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as f:
        data = json.load(f)

    jaccard_values = upper_triangle(data["jaccard"])
    overlap_values = upper_triangle(data["overlap"])

    print(f"n-gram size: {data['n']}, examples: {data['num_examples']}")
    print_percentiles("Jaccard", jaccard_values)
    print_percentiles("Overlap", overlap_values)


if __name__ == "__main__":
    main()
