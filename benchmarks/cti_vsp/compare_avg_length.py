#!/usr/bin/env python3
"""Compare the average entry length (in characters) across several TSV datasets.

Each dataset is a TSV file with two columns: an entry text and a CVSS vector.
Some entries contain a full instruction/prompt wrapped around the actual CVE
description (e.g. cti_vsp_ft_200.tsv). To make the comparison fair, entries
are normalized: if the marker "CVE Description: " is present, only the text
after it is counted; otherwise the whole entry is used as-is.
"""

MARKER = "CVE Description: "

DATASETS = [
    "cti_vsp_bb_deepseek_02.tsv",
    "cti_vsp_bb_deepseek_03_random200.tsv",
    "cti_vsp_bb_deepseek_03_top200.tsv",
    "cti_vsp_fg_deepseek.tsv",
    "cti_vsp_ft_200.tsv",
]


def normalize(entry: str) -> str:
    if MARKER in entry:
        entry = entry.split(MARKER, 1)[1]
    return entry.strip()


def load_entries(path: str) -> list[str]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            text = line.split("\t", 1)[0]
            entries.append(normalize(text))
    return entries


def main() -> None:
    results = []
    for name in DATASETS:
        entries = load_entries(name)
        lengths = [len(e) for e in entries]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        results.append((name, len(entries), avg_len))

    name_width = max(len(name) for name, _, _ in results)
    print(f"{'Datensatz':<{name_width}}  {'Einträge':>8}  {'Ø Länge (Zeichen)':>18}")
    print("-" * (name_width + 8 + 18 + 4))
    for name, count, avg_len in results:
        print(f"{name:<{name_width}}  {count:>8}  {avg_len:>18.2f}")


if __name__ == "__main__":
    main()
