"""Berechnet Längen-Verteilungsstatistiken (Zeichen/Wörter) für das "text"-Feld
in JSON-Dateien wie sie z.B. unter blackbox_generation/toxicity_detection/output
liegen (Liste von Objekten mit einem "text"-Feld).

Nutzung:
    python length_analysis.py <input1.json> [<input2.json> ...]

Für jede Eingabedatei wird eine Ergebnisdatei
    <input-dateiname-ohne-endung>_length_distribution.json
im Unterordner "length_distribution" des Eingabedatei-Ordners geschrieben.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PERCENTILES = [50, 75, 90, 95, 99]


def compute_distribution(values: list[int]) -> dict:
    arr = np.array(values, dtype=float)
    stats = {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
    }
    for p in PERCENTILES:
        stats[f"p{p}"] = float(np.percentile(arr, p))
    stats["max"] = float(np.max(arr))
    return stats


def analyze_file(input_path: Path) -> dict:
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    texts = [entry["text"] for entry in data]
    char_counts = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]

    return {
        "chars": compute_distribution(char_counts),
        "words": compute_distribution(word_counts),
    }


def print_table(result: dict) -> None:
    rows = ["count", "mean", "std", "min"] + [f"p{p}" for p in PERCENTILES] + ["max"]
    header = f"{'':6}{'chars':>15}{'words':>15}"
    print(header)
    for row in rows:
        chars_val = result["chars"][row]
        words_val = result["words"][row]
        print(f"{row:6}{chars_val:15.6f}{words_val:15.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_files", nargs="+", help="Pfad(e) zu Eingabe-JSON-Dateien")
    args = parser.parse_args()

    for input_arg in args.input_files:
        input_path = Path(input_arg)
        if not input_path.exists():
            print(f"Datei nicht gefunden: {input_path}", file=sys.stderr)
            continue

        print(f"\n=== {input_path.name} ===")
        result = analyze_file(input_path)
        print_table(result)

        output_dir = input_path.parent / "length_distribution"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_length_distribution.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"-> geschrieben nach {output_path}")


if __name__ == "__main__":
    main()
