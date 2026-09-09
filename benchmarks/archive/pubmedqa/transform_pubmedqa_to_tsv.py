import argparse
import csv

import pandas as pd

PROMPT_TEMPLATE = (
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer the question with yes, no, or maybe.\n"
    "Answer:\n"
)


def build_pubmedqa_prompt(row):
    contexts = row["context"]["contexts"]
    context = " ".join(str(c) for c in contexts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=row["question"])
    # Escape newlines/tabs so each record stays on a single physical TSV line.
    prompt = prompt.replace("\t", " ").replace("\r\n", "\n").replace("\n", "\\n")
    return prompt


def preprocess_pubmedqa(input_file, output_file):
    pubmedqa_df = pd.read_parquet(input_file)

    pubmedqa_export_df = pd.DataFrame({
        "prompt": pubmedqa_df.apply(build_pubmedqa_prompt, axis=1),
        "gt label": pubmedqa_df["final_decision"],
    })

    pubmedqa_export_df.to_csv(output_file, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    return pubmedqa_export_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform the PubMedQA parquet file into a prompt/label TSV."
    )
    parser.add_argument("--input", "-i", default="pubmedqa.parquet", help="Path to the input parquet file.")
    parser.add_argument("--output", "-o", default="pubmedqa.tsv", help="Path to the output TSV file.")
    return parser.parse_args()


def main():
    args = parse_args()
    preprocess_pubmedqa(args.input, args.output)


if __name__ == "__main__":
    main()
