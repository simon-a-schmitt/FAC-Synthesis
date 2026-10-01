"""Toxicity-Benchmark (toxic-chat) aus den Roh-CSVs aufbereiten.

Analog zu toxicity_detection_data_curation.ipynb:
  1. Test-Set:   toxic-chat_annotation_test.csv, nur human_annotation == True.
  2. Anchor-Set: toxic-chat_annotation_train.csv (alle Zeilen), near-duplicates
                 entfernt -- intra-train (Jaccard >= 0.9, nur gleiches Label)
                 und gegenueber dem Test-Set (Jaccard >= 0.7, Label egal).
                 Danach nur Samples mit human_annotation == True behalten
                 (= der Pool, aus dem im Notebook die Seed-Groups gezogen werden).

Beide Datasets werden als TSV ohne Header mit den Spalten prompt und label nach
tsv_files/ geschrieben. Die (nicht-deterministische) Seed-Group-Ziehung ist
hier bewusst NICHT enthalten.
"""

import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

SAFE_LABEL = "Answer: safe"
TOXIC_LABEL = "Answer: toxic"
LABEL_MAP = {0: SAFE_LABEL, 1: TOXIC_LABEL}

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_whitespace(text: str) -> str:
    """Alle Whitespace-Folgen (inkl. Zeilenumbrueche und Tabs) zu einem einzelnen
    Leerzeichen zusammenfassen -> jedes Beispiel passt in genau eine TSV-Zeile."""
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Near-duplicate-Erkennung: Jaccard ueber Character-n-Gramm-Shingles
# ---------------------------------------------------------------------------


def canonicalize_for_match(text: str) -> str:
    """Aggressive Normalform *nur* fuer den Aehnlichkeitsvergleich: NFKC,
    lowercase, alles ausser [a-z0-9] -> Leerzeichen, Whitespace kollabieren."""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def char_shingles(canon: str, n: int = 4) -> set[str]:
    """Menge der Character-n-Gramme von ``canon``. Strings kuerzer als ``n``
    werden als ein einzelnes Shingle behandelt."""
    if len(canon) < n:
        return {canon} if canon else set()
    return {canon[i : i + n] for i in range(len(canon) - n + 1)}


def build_reference_index(
    reference_prompts: list[str], n: int = 4
) -> tuple[list[set[str]], dict[str, list[int]]]:
    """Shingle-Set je Referenz-Beispiel + invertierter Index (Shingle -> Indizes)."""
    ref_shingles: list[set[str]] = []
    inverted: dict[str, list[int]] = defaultdict(list)
    for idx, prompt in enumerate(reference_prompts):
        shingles = char_shingles(canonicalize_for_match(prompt), n)
        ref_shingles.append(shingles)
        for s in shingles:
            inverted[s].append(idx)
    return ref_shingles, inverted


def max_jaccard_to_reference(
    query_shingles: set[str],
    ref_shingles: list[set[str]],
    inverted: dict[str, list[int]],
    ref_labels: list[str] | None = None,
    require_label: str | None = None,
) -> tuple[float, int]:
    """Hoechste Jaccard-Aehnlichkeit zu einem Referenz-Beispiel mit mindestens
    einem gemeinsamen Shingle. Rueckgabe: (score, ref_index) bzw. (0.0, -1).
    Mit ``require_label`` werden nur Referenzen mit genau diesem Label betrachtet."""
    if not query_shingles:
        return 0.0, -1

    candidates: set[int] = set()
    for s in query_shingles:
        candidates.update(inverted.get(s, ()))

    best_score, best_idx = 0.0, -1
    q_len = len(query_shingles)
    for idx in candidates:
        if require_label is not None and ref_labels[idx] != require_label:
            continue
        ref = ref_shingles[idx]
        inter = len(query_shingles & ref)
        if inter == 0:
            continue
        score = inter / (q_len + len(ref) - inter)
        if score > best_score:
            best_score, best_idx = score, idx
    return best_score, best_idx


# ---------------------------------------------------------------------------
# Test-Set
# ---------------------------------------------------------------------------


def build_test_dataset(path: Path) -> list[dict]:
    """toxic-chat_annotation_test.csv -> Liste von {"prompt", "label"}.
    Nur Beispiele mit ``human_annotation == True``."""
    df = pd.read_csv(path)
    human = df[df["human_annotation"] == True]  # noqa: E712

    dataset = [
        {
            "prompt": normalize_whitespace(str(row["user_input"])),
            "label": LABEL_MAP[row["toxicity"]],
        }
        for _, row in human.iterrows()
    ]

    n_safe = sum(1 for e in dataset if e["label"] == SAFE_LABEL)
    print(f"[test] human_annotation == True: {len(human)} von {len(df)}")
    print(f"[test] {len(dataset)} Beispiele ({n_safe} safe / {len(dataset) - n_safe} toxic)")
    return dataset


# ---------------------------------------------------------------------------
# Anchor-Set
# ---------------------------------------------------------------------------


def build_anchor_dataset(
    path: Path,
    test_dataset: list[dict],
    intra_threshold: float = 0.9,
    test_threshold: float = 0.7,
    char_ngram: int = 4,
) -> list[dict]:
    """toxic-chat_annotation_train.csv -> Liste von {"prompt", "label"}.

    Die near-duplicate-Filterung laeuft wie im Notebook ueber ALLE train-Zeilen
    (kein Annotation-Filter). Erst danach wird auf human_annotation == True
    gefiltert -- das ist genau der Pool, aus dem das Notebook die Seed-Groups
    zieht (draw_seed_groups_from_anchor_v2)."""
    df = pd.read_csv(path)

    raw: list[dict] = []
    for _, row in df.iterrows():
        label = LABEL_MAP.get(row["toxicity"])
        if label is None:
            continue
        raw.append(
            {
                "prompt": normalize_whitespace(str(row["user_input"])),
                "label": label,
                "human_annotation": bool(row["human_annotation"]),
            }
        )

    # (1) intra-train: in CSV-Reihenfolge gegen inkrementell wachsenden Index der
    #     bereits behaltenen Beispiele pruefen; nur gleiches Label zaehlt.
    train_kept: list[dict] = []
    tr_shingles: list[set[str]] = []
    tr_labels: list[str] = []
    tr_inverted: dict[str, list[int]] = defaultdict(list)
    for ex in raw:
        q_shingles = char_shingles(canonicalize_for_match(ex["prompt"]), char_ngram)
        score, ref_idx = max_jaccard_to_reference(
            q_shingles, tr_shingles, tr_inverted, tr_labels, ex["label"]
        )
        if ref_idx >= 0 and score >= intra_threshold:
            continue
        idx = len(train_kept)
        train_kept.append(ex)
        tr_shingles.append(q_shingles)
        tr_labels.append(ex["label"])
        for s in q_shingles:
            tr_inverted[s].append(idx)
    n_intra = len(raw) - len(train_kept)

    # (2) vs. Test-Set: Leakage-Filter, Label egal.
    ref_shingles, inverted = build_reference_index(
        [e["prompt"] for e in test_dataset], char_ngram
    )
    kept: list[dict] = []
    for ex in train_kept:
        q_shingles = char_shingles(canonicalize_for_match(ex["prompt"]), char_ngram)
        score, _ = max_jaccard_to_reference(q_shingles, ref_shingles, inverted)
        if score < test_threshold:
            kept.append(ex)
    n_test = len(train_kept) - len(kept)

    # (3) Nur human-annotierte Samples behalten (Seed-Pool aus dem Notebook).
    n_dedup = len(kept)
    kept = [ex for ex in kept if ex["human_annotation"]]

    n_safe = sum(1 for e in kept if e["label"] == SAFE_LABEL)
    print(f"[anchor] train-Zeilen gesamt: {len(df)}")
    print(f"[anchor] entfernt intra-train (threshold={intra_threshold}, gleiches Label): {n_intra}")
    print(f"[anchor] entfernt vs. Test-Set (threshold={test_threshold}, Label egal):   {n_test}")
    print(f"[anchor] human_annotation == True: {len(kept)} von {n_dedup}")
    print(f"[anchor] {len(kept)} Beispiele ({n_safe} safe / {len(kept) - n_safe} toxic)")
    return kept


def write_tsv(dataset: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dataset, columns=["prompt", "label"]).to_csv(
        path, sep="\t", header=False, index=False
    )
    print(f"-> {path} ({len(dataset)} Zeilen)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=SCRIPT_DIR / "raw_csv_files")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "tsv_files")
    parser.add_argument("--holdout-name", default="toxicity_holdout_set.tsv")
    parser.add_argument("--test-name", default="toxicity_test_set.tsv")
    parser.add_argument("--intra-threshold", type=float, default=0.9)
    parser.add_argument("--test-threshold", type=float, default=0.7)
    parser.add_argument("--char-ngram", type=int, default=4)
    args = parser.parse_args()

    test_dataset = build_test_dataset(args.raw_dir / "toxic-chat_annotation_test.csv")
    anchor_dataset = build_anchor_dataset(
        args.raw_dir / "toxic-chat_annotation_train.csv",
        test_dataset,
        intra_threshold=args.intra_threshold,
        test_threshold=args.test_threshold,
        char_ngram=args.char_ngram,
    )

    write_tsv(test_dataset, args.output_dir / args.test_name)
    write_tsv(anchor_dataset, args.output_dir / args.holdout_name)


if __name__ == "__main__":
    main()
