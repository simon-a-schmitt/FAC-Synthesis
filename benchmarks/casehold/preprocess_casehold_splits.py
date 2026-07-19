import argparse
import csv
import re

import pandas as pd

INSTRUCTION_PROMPT = (
    "Identify the single correct legal holding statement from options A-E to fill the "
    "<HOLDING> placeholder in the citation and output only the corresponding letter."
)

LABEL_MAP = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    "0": "A", "1": "B", "2": "C", "3": "D", "4": "E",
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
}


def sanitize_field(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.replace('"', "'")


def build_casehold_prompt(row):
    prompt_parts = [INSTRUCTION_PROMPT, sanitize_field(row["Citing Text"])]
    for letter, index in zip("ABCDE", range(5)):
        prompt_parts.append(f"{letter}. {sanitize_field(row[f'Holding Statement {index}'])}")
    return " ".join(prompt_parts)


def map_casehold_label(label):
    return LABEL_MAP.get(label, LABEL_MAP.get(str(label), label))


def preprocess_casehold(input_file, output_file):
    casehold_df = pd.read_csv(input_file, sep=",", index_col=0).reset_index(drop=True)
    casehold_df.columns = [
        "Citing Text",
        "Holding Statement 0", "Holding Statement 1", "Holding Statement 2",
        "Holding Statement 3", "Holding Statement 4",
        "Holding Statement 0 Similarity. w. GT Holding",
        "Holding Statement 1 Similarity. w. GT Holding",
        "Holding Statement 2 Similarity. w. GT Holding",
        "Holding Statement 3 Similarity. w. GT Holding",
        "Holding Statement 4 Similarity. w. GT Holding",
        "GT Holding Label",
    ]

    casehold_export_df = pd.DataFrame({
        "prompt": casehold_df.apply(build_casehold_prompt, axis=1),
        "gt holding label": casehold_df["GT Holding Label"].apply(map_casehold_label),
    })

    casehold_export_df.to_csv(output_file, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    return casehold_export_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess a raw CaseHOLD split CSV into a prompt/label TSV."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file (e.g. train.csv).")
    parser.add_argument("--output", "-o", required=True, help="Path to the output TSV file (e.g. casehold.tsv).")
    return parser.parse_args()


def main():
    args = parse_args()
    preprocess_casehold(args.input, args.output)


if __name__ == "__main__":
    main()
