#!/usr/bin/env python3
"""Compare the average entry length (in characters) across several TSV datasets.

Each dataset is a TSV file with two columns: an entry text and a CVSS vector.
Some entries contain a full instruction/prompt wrapped around the actual CVE
description (e.g. cti_vsp_ft_200.tsv). To make the comparison fair, entries
are normalized: if the marker "CVE Description: " is present, only the text
after it is counted; otherwise the whole entry is used as-is.
"""

import statistics

MARKER = "CVE Description: "

PERCENTILES = list(range(10, 100, 100))

DATASETS = [
    "cti_vsp_bb_deepseek_length_adj.tsv",
    "cti_vsp_bb_deepseek_length_adj_02.tsv",
    "cti_vsp_bb_deepseek_06_plain_vanilla.tsv",
    "cti_vsp_benchmark_seed_groups_03.tsv",
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
    rows = []
    for name in DATASETS:
        entries = load_entries(name)
        lengths = [len(e) for e in entries]
        word_counts = [len(e.split()) for e in entries]

        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        median_len = statistics.median(lengths) if lengths else 0.0
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0.0
        median_words = statistics.median(word_counts) if word_counts else 0.0
        min_words = min(word_counts) if word_counts else 0
        max_words = max(word_counts) if word_counts else 0
        std_words = statistics.pstdev(word_counts) if word_counts else 0.0
        pct_words = (
            statistics.quantiles(word_counts, n=10)
            if len(word_counts) >= 2
            else [0.0] * len(PERCENTILES)
        )

        rows.append(
            (
                name,
                len(entries),
                avg_len,
                median_len,
                min_len,
                max_len,
                avg_words,
                median_words,
                min_words,
                max_words,
                std_words,
                pct_words,
            )
        )

    headers = [
        "Datensatz",
        "Einträge",
        "Ø Zeichen",
        "Median Zeichen",
        "Min Zeichen",
        "Max Zeichen",
        "Ø Wörter",
        "Median Wörter",
        "Min Wörter",
        "Max Wörter",
        "σ Wörter",
    ]
    headers += [f"P{p} Wörter" for p in PERCENTILES]

    name_width = max(len(name) for name, *_ in rows)
    col_width = max(max(len(h) for h in headers[1:]), 8)

    header_line = f"{headers[0]:<{name_width}}"
    for h in headers[1:]:
        header_line += f"  {h:>{col_width}}"
    print(header_line)
    print("-" * (name_width + len(headers[1:]) * (col_width + 2)))

    for (
        name,
        count,
        avg_len,
        median_len,
        min_len,
        max_len,
        avg_words,
        median_words,
        min_words,
        max_words,
        std_words,
        pct_words,
    ) in rows:
        line = f"{name:<{name_width}}"
        line += f"  {count:>{col_width}}"
        line += f"  {avg_len:>{col_width}.2f}"
        line += f"  {median_len:>{col_width}.2f}"
        line += f"  {min_len:>{col_width}}"
        line += f"  {max_len:>{col_width}}"
        line += f"  {avg_words:>{col_width}.2f}"
        line += f"  {median_words:>{col_width}.2f}"
        line += f"  {min_words:>{col_width}}"
        line += f"  {max_words:>{col_width}}"
        line += f"  {std_words:>{col_width}.2f}"
        for v in pct_words:
            line += f"  {v:>{col_width}.2f}"
        print(line)


if __name__ == "__main__":
    main()
