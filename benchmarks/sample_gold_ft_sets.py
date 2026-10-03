#!/usr/bin/env python3
"""Zieht fuer jede Seed-Gruppe eines Benchmarks ein Gold-Fine-Tuning-Set aus einer tsv-Datei
(Spalten: Text, Label; eine optionale Header-Zeile "prompt<TAB>label" wird uebersprungen).

Fuer jede Seed-Gruppe {prefix}_seed_group_{xx}.tsv im Seed-Gruppen-Verzeichnis:
  1. Sampling-Seed ueber data_synthesis/shared/seed_derivation.py ableiten
     (Benchmark aus --benchmark, purpose "gold_sampling_seed").
  2. Die Seed-Beispiele aus dem Pool (= Eingabedatei) entfernen. Verglichen wird ueber den
     kanonisierten Text, d.h. auch Duplikate eines Seed-Textes landen nicht im Gold-Set.
  3. --n Beispiele ohne Zuruecklegen aus dem verbleibenden Pool ziehen.
  4. Seed-Beispiele + gezogene Beispiele nach {output_dir}/{prefix}_gold_ft_set_{xx}.tsv schreiben
     ({prefix} wird aus dem Namen der Seed-Gruppen-Datei uebernommen, z.B. "claudette").

Jede Seed-Gruppe zieht unabhaengig aus dem vollstaendigen Pool (nur ohne ihre eigenen
Seed-Beispiele). Die Eingabedatei wird nicht veraendert.

Usage:
    python sample_gold_ft_sets.py --benchmark cti_vsp --seed_groups_dir cti_vsp/seed_groups \
        --input cti_vsp/tsv_files/cti_vsp_holdout.tsv --n 100
    python sample_gold_ft_sets.py --benchmark toxicity_detection \
        --seed_groups_dir toxicity_detection/seed_groups \
        --input toxicity_detection/tsv_files/toxicity_holdout_set.tsv --n 100
    python sample_gold_ft_sets.py --benchmark claudette_tos --seed_groups_dir claudette_tos/seed_groups \
        --input claudette_tos/tsv_files/claudette_tos_val.tsv --n 100
"""

import argparse
import csv
import random
import re
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARKS_DIR.parent / "data_synthesis" / "shared"))
import seed_derivation  # noqa: E402

SEED_GROUP_PATTERN = re.compile(r"^(.+)_seed_group_(\d+)\.tsv$")
HEADER = ["prompt", "label"]


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.reader(f, delimiter="\t") if row]
    if rows and [cell.strip().lower() for cell in rows[0]] == HEADER:
        rows = rows[1:]
    bad = [i for i, row in enumerate(rows, 1) if len(row) != 2]
    if bad:
        raise ValueError(f"Zeilen mit != 2 Spalten in {path}: {bad[:10]}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", required=True, choices=seed_derivation.BENCHMARKS,
                        help="Benchmark-Name (geht in die Seed-Ableitung ein)")
    parser.add_argument("--seed_groups_dir", type=Path, required=True, help="Verzeichnis mit den Seed-Gruppen")
    parser.add_argument("--input", type=Path, required=True, help="Pfad zur Eingabe-tsv-Datei (Pool)")
    parser.add_argument("--n", type=int, required=True, help="Anzahl zu ziehender Beispiele pro Seed-Gruppe")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Ausgabeverzeichnis (default: {benchmarks}/{benchmark}/gold_ft_sets)")
    args = parser.parse_args()

    output_dir = args.output_dir or BENCHMARKS_DIR / args.benchmark / "gold_ft_sets"

    seed_files = sorted(p for p in args.seed_groups_dir.iterdir() if SEED_GROUP_PATTERN.match(p.name))
    if not seed_files:
        parser.error(f"Keine Seed-Gruppen (*_seed_group_XX.tsv) in {args.seed_groups_dir} gefunden")

    pool = read_rows(args.input)
    pool_texts = {seed_derivation.canonicalize_example(row[0]) for row in pool}
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed_file in seed_files:
        prefix, group_id = SEED_GROUP_PATTERN.match(seed_file.name).groups()
        seed_rows = read_rows(seed_file)
        seed = seed_derivation.derive_seed(args.benchmark, [row[0] for row in seed_rows],
                                           seed_derivation.PURPOSE_GOLD_SAMPLING)

        seed_texts = {seed_derivation.canonicalize_example(row[0]) for row in seed_rows}
        remaining = [row for row in pool if seed_derivation.canonicalize_example(row[0]) not in seed_texts]
        missing = seed_texts - pool_texts
        if missing:
            print(f"WARNUNG: {len(missing)} Seed-Beispiel(e) aus {seed_file.name} nicht in {args.input} gefunden")
        if args.n > len(remaining):
            parser.error(f"--n {args.n} groesser als verbleibender Pool ({len(remaining)}) fuer {seed_file.name}")

        sample = random.Random(seed).sample(remaining, args.n)

        output_path = output_dir / f"{prefix}_gold_ft_set_{group_id}.tsv"
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerows(seed_rows + sample)

        print(f"Seed-Gruppe {group_id}: seed={seed}, {len(pool) - len(remaining)} Zeilen entfernt, "
              f"{len(seed_rows)} Seed- + {len(sample)} gezogene Beispiele -> {output_path}")


if __name__ == "__main__":
    main()
