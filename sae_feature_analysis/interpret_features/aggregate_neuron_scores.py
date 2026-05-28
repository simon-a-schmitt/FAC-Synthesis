from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = {"NeuronID", "TextID", "Score", "Span"}


def load_full_table(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", encoding="utf-8")
    missing_columns = EXPECTED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {input_path}: {sorted(missing_columns)}")
    return df


def aggregate_neuron_scores(df: pd.DataFrame) -> pd.DataFrame:
    total_text_count = int(df["TextID"].nunique())

    grouped = (
        df.groupby("NeuronID", as_index=False)
        .agg(
            score_sum=("Score", "sum"),
            texts_with_feature=("TextID", "nunique"),
        )
        .sort_values("NeuronID")
    )

    grouped["total_texts"] = total_text_count
    grouped["avg_score_total_texts"] = grouped["score_sum"] / total_text_count
    grouped["avg_score_feature_texts"] = grouped["score_sum"] / grouped["texts_with_feature"]

    return grouped[
        [
            "NeuronID",
            "total_texts",
            "texts_with_feature",
            "score_sum",
            "avg_score_total_texts",
            "avg_score_feature_texts",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate full.tsv by NeuronID and compute two score averages."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="xxx/threshold_1.0",
        help="Folder that contains full.tsv",
    )
    parser.add_argument(
        "--output",
        default="neuron_scores.tsv",
        help="Output TSV filename written into the same folder as full.tsv.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    folder = script_dir / args.folder
    input_path = folder / "full.tsv"
    output_path = folder / args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find {input_path}")

    df = load_full_table(input_path)
    result = aggregate_neuron_scores(df)
    result.to_csv(output_path, sep="\t", index=False)

    print(f"Wrote {len(result)} rows to {output_path}")
    print(f"Total unique texts: {int(df['TextID'].nunique())}")


if __name__ == "__main__":
    main()