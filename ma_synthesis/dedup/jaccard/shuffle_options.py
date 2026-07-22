"""Shuffle answer options A-E within each prompt of blackbox_sorted.tsv.

Reads a TSV file with one example per line (prompt<TAB>gt_label, no header),
parses out the five "A. ... B. ... C. ... D. ... E. ..." answer options,
shuffles which option text sits behind which letter, reassembles the prompt,
and updates gt_label to the letter the originally-correct option now has.
"""

import random
import re
from pathlib import Path

IN_PATH = Path("blackbox_sorted.tsv")
OUT_PATH = Path("blackbox_sorted_shuffled.tsv")

LETTERS = ["A", "B", "C", "D", "E"]

# Lazy quantifiers so each group stops at the next literal "<letter>. " marker.
# Whitespace around each marker must be non-empty (\s+) so citation fragments
# like "§253B.09" (letter glued to digits, no surrounding space) aren't
# mistaken for an "B. " option marker.
OPTIONS_RE = re.compile(
    r"^A\.\s+(?P<A>.*?)\s+B\.\s+(?P<B>.*?)\s+C\.\s+(?P<C>.*?)\s+D\.\s+(?P<D>.*?)\s+E\.\s+(?P<E>.*)$",
    re.DOTALL,
)


def split_prompt(prompt: str) -> tuple[str, dict[str, str]]:
    """Split a prompt into (prefix incl. trailing separator, {letter: option_text})."""
    sep_idx = prompt.rindex("\\n\\n")
    prefix = prompt[: sep_idx + len("\\n\\n")]
    options_block = prompt[sep_idx + len("\\n\\n") :]

    match = OPTIONS_RE.match(options_block)
    if match is None:
        raise ValueError(f"could not parse options block: {options_block[:200]!r}")

    options = {letter: match.group(letter).strip() for letter in LETTERS}
    return prefix, options


def shuffle_prompt(prompt: str, label: str) -> tuple[str, str]:
    prefix, options = split_prompt(prompt)

    shuffled_letters = LETTERS[:]
    random.shuffle(shuffled_letters)

    new_label = None
    new_option_texts = []
    for new_letter, old_letter in zip(LETTERS, shuffled_letters):
        new_option_texts.append(f"{new_letter}. {options[old_letter]}")
        if old_letter == label:
            new_label = new_letter

    if new_label is None:
        raise ValueError(f"gt_label {label!r} not among parsed options {sorted(options)}")

    new_prompt = prefix + " ".join(new_option_texts)
    return new_prompt, new_label


def main() -> None:
    n = 0
    failed = 0
    with IN_PATH.open("r", encoding="utf-8") as f_in, OUT_PATH.open(
        "w", encoding="utf-8", newline="\n"
    ) as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            prompt, label = line.split("\t")

            try:
                prompt, label = shuffle_prompt(prompt, label)
            except ValueError as exc:
                failed += 1
                print(f"[warn] line {n + 1}: {exc}; keeping original")

            f_out.write(f"{prompt}\t{label}\n")
            n += 1

    print(f"{n} Zeilen nach {OUT_PATH} geschrieben ({failed} nicht parsbar/unveraendert).")


if __name__ == "__main__":
    main()
