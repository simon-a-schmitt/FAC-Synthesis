import argparse
import csv
import random
import sys

csv.field_size_limit(sys.maxsize)


def subsample_tsv(input_file, output_file, n, seed=None, remove_from_input=True):
    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        header = next(reader)
        rows = list(reader)

    if n > len(rows):
        raise ValueError(f"n={n} is larger than the number of available rows ({len(rows)}).")

    if seed is not None:
        random.seed(seed)

    sampled_indices = set(random.sample(range(len(rows)), n))
    sampled_rows = [row for i, row in enumerate(rows) if i in sampled_indices]
    remaining_rows = [row for i, row in enumerate(rows) if i not in sampled_indices]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writerow(header)
        writer.writerows(sampled_rows)

    if remove_from_input:
        with open(input_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
            writer.writerow(header)
            writer.writerows(remaining_rows)

    return len(sampled_rows), len(remaining_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly draw n rows (without replacement) from a TSV file. "
            "Sampled rows are written to the output TSV and, by default, removed from the input TSV "
            "(use --keep-in-input to leave the input file unchanged)."
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input TSV file (sampled rows are removed from here).")
    parser.add_argument("--output", "-o", required=True, help="Path to the output TSV file the sampled rows are written to.")
    parser.add_argument("--n", "-n", type=int, required=True, help="Number of rows to sample.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument(
        "--keep-in-input",
        action="store_true",
        help="Do not remove the sampled rows from the input TSV (default: sampled rows are removed).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_sampled, n_remaining = subsample_tsv(
        args.input, args.output, args.n, args.seed, remove_from_input=not args.keep_in_input
    )
    print(f"{n_sampled} Zeilen zufaellig gezogen und nach {args.output} geschrieben.")
    if args.keep_in_input:
        print(f"{n_remaining} Zeilen verbleiben in {args.input} (Samples wurden nicht entfernt).")
    else:
        print(f"{n_remaining} Zeilen verbleiben in {args.input}.")


if __name__ == "__main__":
    main()
