#!/usr/bin/env python3
"""Zieht k Beispiele ohne Zuruecklegen aus einer claudette_tos-tsv-Datei.

Erlaubt sind nur k=5 und k=10. Davon werden 2 (bei k=5) bzw. 4 (bei k=10)
Beispiele aus der Teilmenge aller Beispiele gezogen, deren Ergebnisvektor
mindestens einen Slot != N enthaelt (z.B. "LTD: N|TER: P|...").
Die restlichen Beispiele werden ohne Zuruecklegen aus der
Gesamtmenge (exklusive bereits gezogener Zeilen) gezogen.

Die gezogenen Zeilen werden anschliessend aus der Eingabedatei entfernt
(die Eingabedatei wird also ueberschrieben und enthaelt nur noch die
nicht gezogenen Zeilen).

Usage:
    python subsample_tsv.py --input claudette_tos_val.tsv [--k 5|10] [--seed 42]
"""

import argparse
import csv
import os
import random
from pathlib import Path


# k -> Anzahl der Beispiele, die mindestens einen Slot != N haben muessen
NUM_POSITIVE_FOR_K = {5: 2, 10: 4}


def has_non_n_slot(label: str) -> bool:
    for slot in label.split("|"):
        slot = slot.strip()
        if not slot:
            continue
        _, _, value = slot.partition(":")
        if value.strip() != "N":
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Pfad zur Eingabe-tsv-Datei")
    parser.add_argument("--k", type=int, default=5, choices=sorted(NUM_POSITIVE_FOR_K),
                         help="Anzahl zu ziehender Beispiele: 5 (davon 2 mit Slot != N) "
                              "oder 10 (davon 4 mit Slot != N) (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="Zufalls-Seed fuer Reproduzierbarkeit")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="Pfad zur Ausgabedatei (default: <input>_sample<k>.tsv im selben Verzeichnis)")
    args = parser.parse_args()

    num_positive = NUM_POSITIVE_FOR_K[args.k]

    rng = random.Random(args.seed)

    with args.input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    n_total = len(rows)
    if args.k > n_total:
        parser.error(f"k ({args.k}) ist groesser als die Anzahl verfuegbarer Zeilen ({n_total})")

    positive_indices = [i for i, row in enumerate(rows) if len(row) > 1 and has_non_n_slot(row[1])]
    if len(positive_indices) < num_positive:
        parser.error(f"Nicht genug Zeilen mit einem Slot != N gefunden "
                     f"({len(positive_indices)} < {num_positive})")

    chosen_indices = set(rng.sample(positive_indices, num_positive))

    remaining_needed = args.k - num_positive
    pool = [i for i in range(n_total) if i not in chosen_indices]
    chosen_indices.update(rng.sample(pool, remaining_needed))

    selected_rows = [rows[i] for i in sorted(chosen_indices)]
    remaining_rows = [rows[i] for i in range(n_total) if i not in chosen_indices]

    output_path = args.output
    if output_path is None:
        output_path = args.input.with_name(f"{args.input.stem}_sample{args.k}{args.input.suffix}")

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(selected_rows)

    tmp_input_path = args.input.with_suffix(args.input.suffix + ".tmp")
    with tmp_input_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(remaining_rows)
    os.replace(tmp_input_path, args.input)

    print(f"{len(selected_rows)} Beispiele geschrieben nach {output_path}")
    print(f"{len(remaining_rows)} verbleibende Beispiele in {args.input} gespeichert "
          f"({len(selected_rows)} Zeilen entfernt)")


if __name__ == "__main__":
    main()
