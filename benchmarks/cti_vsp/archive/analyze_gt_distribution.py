import csv
import json
import re
from collections import Counter

INPUT_PATH = "cti_vsp_benchmark_test_500.tsv"
OUTPUT_PATH = "cti_vsp_gt_metric_distribution.json"

VECTOR_RE = re.compile(r"CVSS:3\.1(?:/[A-Z]{1,2}:[A-Z])+")
METRIC_ORDER = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]


def parse_vector(vector):
    metrics = {}
    for part in vector.split("/")[1:]:
        name, value = part.split(":")
        metrics[name] = value
    return metrics


def analyze_gt_distribution():
    counters = {metric: Counter() for metric in METRIC_ORDER}
    total = 0

    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        for row in reader:
            if not row:
                continue
            match = VECTOR_RE.search(row[-1])
            if not match:
                continue
            metrics = parse_vector(match.group())
            for name in METRIC_ORDER:
                counters[name][metrics[name]] += 1
            total += 1

    distribution = {
        metric: dict(sorted(counters[metric].items(), key=lambda kv: -kv[1]))
        for metric in METRIC_ORDER
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"total_instances": total, "metric_distribution": distribution}, f, indent=2, ensure_ascii=False)

    print(f"Analyzed {total} instances -> {OUTPUT_PATH}")


def main():
    analyze_gt_distribution()


if __name__ == "__main__":
    main()
