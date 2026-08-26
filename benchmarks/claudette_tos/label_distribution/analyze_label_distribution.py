"""Analyze the distribution of claudette_tos clause labels (last column) in a benchmark TSV.

Usage:
    python analyze_label_distribution.py <input.tsv> [output_prefix]

Writes:
    <output_prefix>_labels.tsv   - full label counts/proportions, most common first
    <output_prefix>_summary.json - full-label + per-category distribution, normalized
                                    so two summary JSONs can be diffed to spot
                                    distribution shifts at a glance.

If output_prefix is omitted, it defaults to the input file's stem.
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

CATEGORY_ORDER = ["LTD", "TER", "CH", "CR", "USE", "LAW", "J", "ARB"]
LABEL_RE = re.compile(r"(?:" + "|".join(f"{cat}:[YN]\\|" for cat in CATEGORY_ORDER) + r"){8}")


def parse_label(label):
    categories = {}
    for part in label.split("|"):
        if not part:
            continue
        name, value = part.split(":")
        categories[name] = value
    return categories


def read_labels(input_path):
    labels = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if not row:
                continue
            match = LABEL_RE.search(row[-1])
            if not match:
                continue
            labels.append(match.group())
    return labels


def counter_with_proportions(counter, total):
    return {
        label: {"count": count, "proportion": round(count / total, 6)}
        for label, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    }


def build_summary(labels, source_path):
    total = len(labels)
    label_counts = Counter(labels)

    category_counters = {category: Counter() for category in CATEGORY_ORDER}
    for label in labels:
        categories = parse_label(label)
        for category in CATEGORY_ORDER:
            category_counters[category][categories[category]] += 1

    summary = {
        "source_file": str(source_path),
        "total_instances": total,
        "num_unique_labels": len(label_counts),
        "full_label_distribution": counter_with_proportions(label_counts, total) if total else {},
        "category_distribution": {
            category: counter_with_proportions(category_counters[category], total) if total else {}
            for category in CATEGORY_ORDER
        },
    }
    return summary, label_counts


def write_labels_tsv(label_counts, total, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["label", "count", "proportion"])
        for label, count in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow([label, count, round(count / total, 6) if total else 0.0])


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <input.tsv> [output_prefix]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / input_path.stem

    labels = read_labels(input_path)
    summary, label_counts = build_summary(labels, input_path)

    labels_tsv_path = output_prefix.with_name(output_prefix.name + "_labels.tsv")
    summary_json_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    write_labels_tsv(label_counts, summary["total_instances"], labels_tsv_path)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Analyzed {summary['total_instances']} instances ({summary['num_unique_labels']} unique labels)")
    print(f"-> {labels_tsv_path}")
    print(f"-> {summary_json_path}")


if __name__ == "__main__":
    main()
