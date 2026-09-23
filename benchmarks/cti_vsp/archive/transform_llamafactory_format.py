import argparse
import csv
import random
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.prompt_builder import extract_cve_description_block
from benchmark_play_ground.run_cti_vsp_benchmark import CTI_VSP_SYSTEM_PROMPT

CVE_DESCRIPTION_MARKER = "CVE Description: "


def transform_llamafactory(sample_tsv, output_json, contains_instruction=False):

    records = []
    with open(sample_tsv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            prompt, label = row[0], row[1]
            if contains_instruction:
                idx = prompt.find(CVE_DESCRIPTION_MARKER)
                if idx != -1:
                    prompt = prompt[idx + len(CVE_DESCRIPTION_MARKER):]
            records.append({
                "system": CTI_VSP_SYSTEM_PROMPT,
                "instruction": "CVE Description: " + prompt,
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
    parser.add_argument(
        "--contains-instruction",
        action="store_true",
        help="Set if the prompt column already contains a baked-in instruction preamble; "
             "everything up to and including 'CVE Description: ' is stripped before use.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    transform_llamafactory(args.input, args.output, contains_instruction=args.contains_instruction)

   


if __name__ == "__main__":
    main()
