import csv

INPUT_PATH = "cti-vsp.tsv"
OUTPUT_PATH = "cti_vsp_benchmark.tsv"
COLUMNS = ["Prompt", "GT"]


def build_benchmark():
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        rows = list(reader)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in COLUMNS})


def main():
    build_benchmark()


if __name__ == "__main__":
    main()
