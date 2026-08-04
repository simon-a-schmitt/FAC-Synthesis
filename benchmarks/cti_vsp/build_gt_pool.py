import csv
import random

INPUT_PATH = "cti-vsp.tsv"
OUTPUT_PATH = "cti_vsp_gt_pool.txt"


def build_gt_pool():
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        pool = [row["GT"] for row in reader]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(pool) + "\n")

    print(f"Wrote {len(pool)} GT values -> {OUTPUT_PATH}")
    return pool


def load_gt_pool(path=OUTPUT_PATH):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def sample_gt(k=1, path=OUTPUT_PATH):
    return random.choices(load_gt_pool(path), k=k)


def main():
    build_gt_pool()


if __name__ == "__main__":
    main()
