"""Analyze the distribution of CVSS vector labels (column 2) in a benchmark TSV.

Usage:
    python analyze_label_distribution.py <input.tsv> [output_prefix]

Writes:
    <output_prefix>_vectors.tsv  - full CVSS vector counts/proportions, most common first
    <output_prefix>_summary.json - full-vector + per-metric distribution, normalized
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

VECTOR_RE = re.compile(r"CVSS:3\.1(?:/[A-Z]{1,2}:[A-Z])+")
METRIC_ORDER = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]


def parse_vector(vector):
    metrics = {}
    for part in vector.split("/")[1:]:
        name, value = part.split(":")
        metrics[name] = value
    return metrics


def read_labels(input_path):
    labels = []
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if not row:
                continue
            match = VECTOR_RE.search(row[-1])
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
    vector_counts = Counter(labels)

    metric_counters = {metric: Counter() for metric in METRIC_ORDER}
    for label in labels:
        metrics = parse_vector(label)
        for metric in METRIC_ORDER:
            metric_counters[metric][metrics[metric]] += 1

    summary = {
        "source_file": str(source_path),
        "total_instances": total,
        "num_unique_vectors": len(vector_counts),
        "full_vector_distribution": counter_with_proportions(vector_counts, total) if total else {},
        "metric_distribution": {
            metric: counter_with_proportions(metric_counters[metric], total) if total else {}
            for metric in METRIC_ORDER
        },
    }
    return summary, vector_counts


def write_vectors_tsv(vector_counts, total, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["vector", "count", "proportion"])
        for vector, count in sorted(vector_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            writer.writerow([vector, count, round(count / total, 6) if total else 0.0])


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <input.tsv> [output_prefix]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_prefix = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / input_path.stem

    labels = read_labels(input_path)
    summary, vector_counts = build_summary(labels, input_path)

    vectors_tsv_path = output_prefix.with_name(output_prefix.name + "_vectors.tsv")
    summary_json_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    write_vectors_tsv(vector_counts, summary["total_instances"], vectors_tsv_path)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Analyzed {summary['total_instances']} instances ({summary['num_unique_vectors']} unique vectors)")
    print(f"-> {vectors_tsv_path}")
    print(f"-> {summary_json_path}")


if __name__ == "__main__":
    main()
