import argparse
import csv
import random
import sys

csv.field_size_limit(sys.maxsize)

# Predefined CVSS v3.1 vectors that must appear exactly once in the drawn
# seed group, keyed by group size n. The n=10 set is a superset of the n=5
# set plus five additional vectors.
TARGET_VECTORS = {
    5: [
        "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "CVSS:3.1/AV:N/AC:L/PR:L/UI:R/S:C/C:L/I:L/A:N",
        "CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
        "CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H",
        "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
    ],
    10: [
        "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "CVSS:3.1/AV:N/AC:L/PR:L/UI:R/S:C/C:L/I:L/A:N",
        "CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H",
        "CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:N/I:N/A:H",
        "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H",
        "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
        "CVSS:3.1/AV:N/AC:L/PR:H/UI:N/S:U/C:H/I:H/A:H",
        "CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H",
        "CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N",
        "CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:N/I:N/A:H",
    ],
}


def stratified_sample_tsv(input_file, output_file, n, seed=None):
    target_vectors = TARGET_VECTORS[n]

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        rows = list(reader)

    indices_by_gt = {}
    for i, row in enumerate(rows):
        indices_by_gt.setdefault(row[-1], []).append(i)

    if seed is not None:
        random.seed(seed)

    sampled_indices = []
    for vector in target_vectors:
        candidates = [i for i in indices_by_gt.get(vector, []) if i not in sampled_indices]
        if not candidates:
            raise ValueError(f"No remaining row with GT={vector} found in {input_file}.")
        sampled_indices.append(random.choice(candidates))

    sampled_by_index = {i: rows[i] for i in sampled_indices}
    sampled_rows = [sampled_by_index[i] for i in sampled_indices]
    remaining_rows = [row for i, row in enumerate(rows) if i not in sampled_indices]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writerows(sampled_rows)

    with open(input_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writerows(remaining_rows)

    return len(sampled_rows), len(remaining_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Draw a stratified seed group from a CTI-VSP TSV pool: exactly one row is "
            "picked per predefined CVSS v3.1 vector. Sampled rows are written to the "
            "output TSV and removed from the input TSV."
        )
    )
    parser.add_argument("--input", "-i", default="cti_vsp_benchmark.tsv", help="Path to the input pool TSV file (sampled rows are removed from here).")
    parser.add_argument("--output", "-o", required=True, help="Path to the output TSV file the sampled seed group is written to.")
    parser.add_argument("--n", "-n", type=int, required=True, choices=sorted(TARGET_VECTORS), help="Seed group size; selects the predefined set of target CVSS vectors.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    n_sampled, n_remaining = stratified_sample_tsv(args.input, args.output, args.n, args.seed)
    print(f"{n_sampled} stratifizierte Zeilen gezogen und nach {args.output} geschrieben.")
    print(f"{n_remaining} Zeilen verbleiben in {args.input}.")


if __name__ == "__main__":
    main()
