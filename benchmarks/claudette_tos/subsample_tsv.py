#!/usr/bin/env python3
"""Zieht n Beispiele ohne Zuruecklegen aus einer claudette_tos-tsv-Datei.

k Beispiele werden aus der Teilmenge aller Beispiele gezogen, deren
Ergebnisvektor mindestens einen Slot != N enthaelt (z.B. "LTD:N|TER:P|...").
Die restlichen (n - k) Beispiele werden ohne Zuruecklegen aus der
Gesamtmenge (exklusive bereits gezogener Zeilen) gezogen.

Die gezogenen Zeilen werden anschliessend aus der Eingabedatei entfernt
(die Eingabedatei wird also ueberschrieben und enthaelt nur noch die
nicht gezogenen Zeilen).

Usage:
    python subsample_tsv.py --input claudette_tos_val.tsv [--n 5] [--k 2] [--seed 42]
"""

import argparse
import csv
import os
import random
from pathlib import Path


def has_non_n_slot(label: str) -> bool:
    for slot in label.split("|"):
        if not slot:
            continue
        _, _, value = slot.partition(":")
        if value != "N":
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Pfad zur Eingabe-tsv-Datei")
    parser.add_argument("--n", type=int, default=5, help="Anzahl zu ziehender Beispiele (default: 5)")
    parser.add_argument("--k", type=int, default=2,
                         help="Anzahl der Beispiele aus dem Nicht-nur-N-Pool (default: 2)")
    parser.add_argument("--seed", type=int, default=None, help="Zufalls-Seed fuer Reproduzierbarkeit")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="Pfad zur Ausgabedatei (default: <input>_sample<n>.tsv im selben Verzeichnis)")
    args = parser.parse_args()

    if args.n < 1:
        parser.error("n muss >= 1 sein")
    if args.k < 0 or args.k > args.n:
        parser.error("k muss zwischen 0 und n liegen")

    rng = random.Random(args.seed)

    with args.input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = list(reader)

    n_total = len(rows)
    if args.n > n_total:
        parser.error(f"n ({args.n}) ist groesser als die Anzahl verfuegbarer Zeilen ({n_total})")

    positive_indices = [i for i, row in enumerate(rows) if len(row) > 1 and has_non_n_slot(row[1])]
    if len(positive_indices) < args.k:
        parser.error(f"Nicht genug Zeilen mit einem Slot != N gefunden ({len(positive_indices)} < k={args.k})")

    chosen_indices = set(rng.sample(positive_indices, args.k))

    remaining_needed = args.n - args.k
    pool = [i for i in range(n_total) if i not in chosen_indices]
    chosen_indices.update(rng.sample(pool, remaining_needed))

    selected_rows = [rows[i] for i in sorted(chosen_indices)]
    remaining_rows = [rows[i] for i in range(n_total) if i not in chosen_indices]

    output_path = args.output
    if output_path is None:
        output_path = args.input.with_name(f"{args.input.stem}_sample{args.n}{args.input.suffix}")

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(selected_rows)

    tmp_input_path = args.input.with_suffix(args.input.suffix + ".tmp")
    with tmp_input_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(remaining_rows)
    os.replace(tmp_input_path, args.input)

    print(f"{len(selected_rows)} Beispiele geschrieben nach {output_path}")
    print(f"{len(remaining_rows)} verbleibende Beispiele in {args.input} gespeichert "
          f"({len(selected_rows)} Zeilen entfernt)")


if __name__ == "__main__":
    main()
