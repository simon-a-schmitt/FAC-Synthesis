"""Eine Seed-Group per Rejection Sampling aus einer TSV (ohne Header) ziehen.

Es werden wiederholt ``k`` Zeilen gleichverteilt ohne Zuruecklegen aus der
gesamten Input-TSV gezogen, bis die Ziehung genau die geforderte
Label-Verteilung hat:
  - k=5:  2 toxic / 3 safe
  - k=10: 4 toxic / 6 safe

Input und Output: TSV ohne Header mit zwei Spalten (prompt, label), wobei label
"Answer: safe" / "Answer: toxic" ist.

ACHTUNG: Die gezogenen Zeilen werden anschliessend aus der Input-TSV entfernt
(die Input-Datei wird ueberschrieben). Mehrere Aufrufe nacheinander liefern so
disjunkte Seed-Groups.
"""

import argparse
import random
from pathlib import Path

import pandas as pd

SAFE_LABEL = "Answer: safe"
TOXIC_LABEL = "Answer: toxic"

# k -> Anzahl toxic-Samples (Rest safe)
N_TOXIC_BY_K = {5: 2, 10: 4}


def load_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["prompt", "label"], dtype=str)
    df["label"] = df["label"].str.strip()

    unknown = set(df["label"]) - {SAFE_LABEL, TOXIC_LABEL}
    if unknown:
        raise ValueError(f"Unbekannte Labels in {path}: {sorted(unknown)}")
    return df


def detect_line_terminator(path: Path) -> str:
    """Zeilenende der Datei uebernehmen, damit die ueberschriebene Input-TSV ihr
    Format behaelt."""
    with path.open("rb") as f:
        return "\r\n" if b"\r\n" in f.read(1 << 16) else "\n"


def rejection_sample(
    df: pd.DataFrame, k: int, n_toxic: int, rng: random.Random, max_tries: int
) -> tuple[list[int], int]:
    """k Zeilen ohne Zuruecklegen ziehen, bis genau ``n_toxic`` davon toxic sind.
    Rueckgabe: (Zeilen-Indizes der Ziehung, Anzahl benoetigter Versuche)."""
    labels = df["label"].tolist()
    n_avail_toxic = labels.count(TOXIC_LABEL)
    if n_avail_toxic < n_toxic or len(labels) - n_avail_toxic < k - n_toxic:
        raise ValueError(
            f"Zu wenige Samples: benoetigt {n_toxic} toxic / {k - n_toxic} safe, "
            f"verfuegbar {n_avail_toxic} toxic / {len(labels) - n_avail_toxic} safe"
        )

    for attempt in range(1, max_tries + 1):
        idx = rng.sample(range(len(df)), k)
        if sum(labels[i] == TOXIC_LABEL for i in idx) == n_toxic:
            return idx, attempt
    raise RuntimeError(f"Keine gueltige Ziehung nach {max_tries} Versuchen")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="Input-TSV (ohne Header)")
    parser.add_argument("--output", type=Path, required=True, help="Output-TSV inkl. Dateiname")
    parser.add_argument("--k", type=int, required=True, choices=sorted(N_TOXIC_BY_K))
    parser.add_argument("--seed", type=int, default=None, help="Optionaler RNG-Seed")
    parser.add_argument("--max-tries", type=int, default=1_000_000)
    args = parser.parse_args()

    line_terminator = detect_line_terminator(args.input)
    df = load_tsv(args.input)
    n_toxic = N_TOXIC_BY_K[args.k]

    rng = random.Random(args.seed)
    idx, n_tries = rejection_sample(df, args.k, n_toxic, rng, args.max_tries)

    group = df.iloc[idx].reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    group.to_csv(args.output, sep="\t", header=False, index=False)

    # Gezogene Zeilen aus dem Pool entfernen und Input-TSV ueberschreiben
    # (erst nachdem die Seed-Group erfolgreich geschrieben wurde).
    remaining = df.drop(index=df.index[idx])
    remaining.to_csv(
        args.input, sep="\t", header=False, index=False, lineterminator=line_terminator
    )

    print(
        f"[seed-group] {args.output} ({args.k} Samples: {n_toxic} toxic / "
        f"{args.k - n_toxic} safe, {n_tries} Versuche)"
    )
    print(f"[seed-group] {args.input} ueberschrieben ({len(df)} -> {len(remaining)} Zeilen)")


if __name__ == "__main__":
    main()
