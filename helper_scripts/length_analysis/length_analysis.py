"""Berechnet Längen-Verteilungsstatistiken (Zeichen/Wörter, optional Tokens) für Texte aus
- JSON-Dateien wie sie z.B. unter blackbox_generation/toxicity_detection/output liegen
  (Liste von Objekten mit einem "text"-Feld), oder
- TSV-Dateien (zwei Spalten pro Zeile: Text, Label; Text = erste Spalte), wie die
  Trainings- und Test-TSVs der Benchmarks.

Wörter = Whitespace-Split. Tokens = input_ids des übergebenen Tokenizers auf dem reinen
Text (ohne Chat-Template, ohne Special Tokens); benötigt transformers.

Nutzung:
    python length_analysis.py <input1.json|tsv> [<input2> ...] [--tokenizer MODEL_DIR]

Für jede Eingabedatei wird eine Ergebnisdatei
    <input-dateiname-ohne-endung>_length_distribution.json
im Unterordner "length_distribution" des Eingabedatei-Ordners geschrieben.

Als Modul (experiments/collect_results.py): analyze_texts(read_texts(path), tokenizer), bzw.
measure_lengths + analyze_lengths, wenn die Rohlängen zusätzlich für distance_to_reference
(TV-Distanz und Wasserstein-1 gegen eine Referenzverteilung, z.B. das Testset) gebraucht werden.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PERCENTILES = [50, 75, 90, 95, 99]
TV_BINS = 10  # Quantil-Bins der Referenz für distance_to_reference (Dezile)


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


def read_texts(input_path: Path) -> list[str]:
    """Texte einer .tsv (erste Spalte, Zeilen ohne Label übersprungen wie in
    distribution_analysis.read_labels) oder einer JSON-Liste mit "text"-Feld."""
    if input_path.suffix == ".tsv":
        with input_path.open(encoding="utf-8", newline="") as f:
            texts = [row[0] for row in csv.reader(f, delimiter="\t") if len(row) >= 2 and row[1].strip()]
    else:
        with input_path.open(encoding="utf-8") as f:
            texts = [entry["text"] for entry in json.load(f)]
    if not texts:
        raise ValueError(f"Keine Texte in {input_path} gefunden.")
    return texts


def measure_lengths(texts: list[str], tokenizer=None) -> dict[str, list[int]]:
    """Rohlängen pro Einheit (chars, words, tokens falls Tokenizer gegeben)."""
    lengths = {"chars": [len(t) for t in texts], "words": [len(t.split()) for t in texts]}
    if tokenizer is not None:
        lengths["tokens"] = [len(i) for i in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    return lengths


def analyze_lengths(lengths: dict[str, list[int]]) -> dict:
    return {unit: compute_distribution(values) for unit, values in lengths.items()}


def analyze_texts(texts: list[str], tokenizer=None) -> dict:
    return analyze_lengths(measure_lengths(texts, tokenizer))


def tv_quantile_bins(values, reference, n_bins: int = TV_BINS) -> float:
    """TV-Distanz (1/2 * sum |p_i - q_i|) der beiden Längenverteilungen über Bins an den Quantilen
    der Referenz (n_bins gleich besetzte Bins, bei Ties zusammengelegt), plus je ein Überlauf-Bin
    unterhalb des Minimums und oberhalb des Maximums der Referenz - Längen außerhalb des
    Referenzbereichs zählen damit voll als Abweichung. In [0, 1]."""
    values, reference = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    cuts = np.append(edges[:-1], np.nextafter(edges[-1], np.inf))  # letztes Bin schließt das Maximum ein
    # Index 0: < min, 1..len(cuts)-1: Quantil-Bins, len(cuts): > max
    p = np.bincount(np.searchsorted(cuts, values, side="right"), minlength=len(cuts) + 1) / values.size
    q = np.bincount(np.searchsorted(cuts, reference, side="right"), minlength=len(cuts) + 1) / reference.size
    return float(0.5 * np.abs(p - q).sum())


def wasserstein1(values, reference) -> float:
    """Wasserstein-1 / Earth Mover's Distance zweier empirischer 1D-Verteilungen = Integral über
    |F_values(x) - F_reference(x)| (wie scipy.stats.wasserstein_distance), in Einheiten der Länge."""
    a, b = np.sort(np.asarray(values, dtype=float)), np.sort(np.asarray(reference, dtype=float))
    support = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, support[:-1], side="right") / a.size
    cdf_b = np.searchsorted(b, support[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * np.diff(support)))


def distance_to_reference(values, reference, n_bins: int = TV_BINS) -> dict:
    """tv: tv_quantile_bins; w1: wasserstein1 (Längeneinheiten); w1_norm: w1 / Mittelwert der Referenz."""
    w1 = wasserstein1(values, reference)
    return {"tv": tv_quantile_bins(values, reference, n_bins), "w1": w1,
            "w1_norm": w1 / float(np.mean(reference))}


def analyze_file(input_path: Path, tokenizer=None) -> dict:
    return analyze_texts(read_texts(input_path), tokenizer)


def print_table(result: dict) -> None:
    units = [u for u in ("chars", "words", "tokens") if u in result]
    rows = ["count", "mean", "std", "min"] + [f"p{p}" for p in PERCENTILES] + ["max"]
    print(f"{'':6}" + "".join(f"{u:>15}" for u in units))
    for row in rows:
        print(f"{row:6}" + "".join(f"{result[u][row]:15.6f}" for u in units))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_files", nargs="+", help="Pfad(e) zu Eingabe-JSON/TSV-Dateien")
    parser.add_argument("--tokenizer", help="Modell-/Tokenizer-Verzeichnis für Token-Längen (optional)")
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for input_arg in args.input_files:
        input_path = Path(input_arg)
        if not input_path.exists():
            print(f"Datei nicht gefunden: {input_path}", file=sys.stderr)
            continue

        print(f"\n=== {input_path.name} ===")
        result = analyze_file(input_path, tokenizer)
        print_table(result)

        output_dir = input_path.parent / "length_distribution"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_length_distribution.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"-> geschrieben nach {output_path}")


if __name__ == "__main__":
    main()
