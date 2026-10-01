#!/usr/bin/env python3
"""Zieht eine Seed-Gruppe aus einer cti_vsp-tsv-Datei (ohne Header, Spalten: Text, CVSS-Vektor).

--k 5:  5 Beispiele, alle mit unterschiedlichem Label.
--k 10: 10 Beispiele, jedes Label kommt hoechstens zweimal vor
        (also mindestens 5 unterschiedliche Label).

Gezogen wird per Rejection Sampling, d.h. gleichverteilt ueber alle
Teilmengen, die die Nebenbedingung erfuellen. Gezogen wird ohne Zuruecklegen:
Die gezogenen Zeilen werden anschliessend aus der Eingabedatei entfernt
(die Eingabedatei wird also ueberschrieben und enthaelt nur noch die
nicht gezogenen Zeilen).

Usage:
    python sample_seed_groups.py tsv_files/cti_vsp_no_test.tsv seed_groups/out.tsv --k 5 [--seed 42]
"""

import argparse
import csv
import os
import random
from collections import Counter
from pathlib import Path

# k -> maximale Anzahl Beispiele pro Label
MAX_PER_LABEL = {5: 1, 10: 2}
MAX_TRIES = 1_000_000


def is_valid(sample, max_per_label):
    return max(Counter(row[1] for row in sample).values()) <= max_per_label


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="Pfad zur Eingabe-tsv-Datei")
    parser.add_argument("--output", type=Path, help="Pfad zur Ausgabe-tsv-Datei")
    parser.add_argument("--k", type=int, required=True, choices=sorted(MAX_PER_LABEL),
                        help="Anzahl zu ziehender Beispiele (5 oder 10)")
    parser.add_argument("--seed", type=int, default=None, help="Zufalls-Seed fuer Reproduzierbarkeit")
    args = parser.parse_args()

    max_per_label = MAX_PER_LABEL[args.k]
    rng = random.Random(args.seed)

    with args.input.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.reader(f, delimiter="\t") if row]

    bad = [i for i, row in enumerate(rows, 1) if len(row) != 2]
    if bad:
        parser.error(f"Zeilen mit != 2 Spalten in {args.input}: {bad[:10]}")

    n_labels = len({row[1] for row in rows})
    if n_labels * max_per_label < args.k:
        parser.error(f"Nur {n_labels} unterschiedliche Label vorhanden, k={args.k} nicht erfuellbar")

    for _ in range(MAX_TRIES):
        chosen_indices = rng.sample(range(len(rows)), args.k)
        sample = [rows[i] for i in chosen_indices]
        if is_valid(sample, max_per_label):
            break
    else:
        raise RuntimeError(f"Keine gueltige Stichprobe nach {MAX_TRIES} Versuchen gefunden")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(sample)

    chosen = set(chosen_indices)
    remaining_rows = [row for i, row in enumerate(rows) if i not in chosen]
    tmp_input_path = args.input.with_suffix(args.input.suffix + ".tmp")
    with tmp_input_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(remaining_rows)
    os.replace(tmp_input_path, args.input)

    label_counts = Counter(row[1] for row in sample)
    print(f"{len(sample)} Beispiele mit {len(label_counts)} unterschiedlichen Labels geschrieben nach {args.output}")
    print(f"{len(remaining_rows)} verbleibende Beispiele in {args.input} gespeichert "
          f"({len(sample)} Zeilen entfernt)")


if __name__ == "__main__":
    main()
