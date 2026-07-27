import csv
from collections import Counter

FILE_PATH = "cti-rcm.tsv"

def get_class_distribution():

    with open(FILE_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        rows = list(reader)

    gt_index = header.index("GT")
    labels = [row[gt_index] for row in rows]
    distribution = Counter(labels)

    for label, count in distribution.most_common():
        print(f"{label}\t{count}")

def main():
    get_class_distribution()


if __name__ == "__main__":
    main()