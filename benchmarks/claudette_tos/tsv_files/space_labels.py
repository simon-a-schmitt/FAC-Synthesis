#!/usr/bin/env python3
"""Transform the label column of claudette_tos TSV files.

Each row is `text<TAB>label`, where the label looks like

    LTD:N|TER:N|CH:N|CR:N|USE:N|LAW:N|J:N|ARB:N|

This script rewrites the label so that every slot has a space after the
colon and the trailing pipe is removed:

    LTD: N|TER: N|CH: N|CR: N|USE: N|LAW: N|J: N|ARB: N

The text column is left untouched.

Usage:
    python space_labels.py FILE [FILE ...]              # writes <name>_spaced.tsv
    python space_labels.py FILE --suffix _foo           # custom suffix
    python space_labels.py FILE -o OUT.tsv              # explicit output (single file)
    python space_labels.py FILE --in-place              # overwrite input
"""

import argparse
import re
import sys
from pathlib import Path

# Matches a slot key immediately followed by a colon, e.g. "LTD:" or "ARB:".
SLOT_KEY_COLON = re.compile(r"([A-Za-z0-9_]+):")


def transform_label(label: str) -> str:
    """Insert a space after each slot colon and drop the trailing pipe."""
    label = label.strip()
    label = SLOT_KEY_COLON.sub(r"\1: ", label)
    return label.rstrip("|").rstrip()


def transform_file(src: Path, dst: Path) -> int:
    rows = 0
    with src.open("r", encoding="utf-8", newline="") as fh_in, \
            dst.open("w", encoding="utf-8", newline="") as fh_out:
        for line in fh_in:
            line = line.rstrip("\n")
            if not line:
                fh_out.write("\n")
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"{src}: expected 2 tab-separated columns, got {len(parts)}: {line!r}"
                )
            text, label = parts
            fh_out.write(f"{text}\t{transform_label(label)}\n")
            rows += 1
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="input TSV file(s)")
    parser.add_argument("-o", "--output", type=Path,
                        help="explicit output path (only valid with a single input file)")
    parser.add_argument("--suffix", default="_spaced",
                        help="suffix added before .tsv for the output name (default: _spaced)")
    parser.add_argument("--in-place", action="store_true",
                        help="overwrite the input files")
    args = parser.parse_args(argv)

    if args.output and len(args.files) != 1:
        parser.error("-o/--output requires exactly one input file")
    if args.output and args.in_place:
        parser.error("-o/--output and --in-place are mutually exclusive")

    for src in args.files:
        if not src.is_file():
            print(f"skip (not a file): {src}", file=sys.stderr)
            continue
        if args.in_place:
            dst = src
            tmp = src.with_suffix(src.suffix + ".tmp")
            rows = transform_file(src, tmp)
            tmp.replace(dst)
        else:
            if args.output:
                dst = args.output
            else:
                dst = src.with_name(f"{src.stem}{args.suffix}{src.suffix}")
            rows = transform_file(src, dst)
        print(f"{src}  ->  {dst}  ({rows} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
