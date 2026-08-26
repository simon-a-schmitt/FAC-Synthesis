import argparse
import csv
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.run_claudette_benchmark import CLAUDETTE_SYSTEM_PROMPT


def transform_llamafactory(sample_tsv, output_json):

    with open(sample_tsv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    # The CLAUDETTE-TOS TSVs have a "prompt\tlabel" header row; drop it if present
    # (mirrors the header detection in benchmark_play_ground/data_loader.py).
    if rows and rows[0][0].strip().lower() == "prompt":
        rows = rows[1:]

    records = []
    for row in rows:
        prompt, label = row[0], row[1]
        prompt = prompt.replace("\\n", "\n")
        records.append({
            "system": CLAUDETTE_SYSTEM_PROMPT,
            "instruction": prompt.strip(),
            "input": "",
            "output": label.strip()
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Transforms data into llamafactory format (instruction, input, output)"
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input TSV file (original format).")
    parser.add_argument("--output", "-o", required=True, help="Path to the output TSV file (llamafactory format).")
    return parser.parse_args()


def main():
    args = parse_args()
    transform_llamafactory(args.input, args.output)


if __name__ == "__main__":
    main()
