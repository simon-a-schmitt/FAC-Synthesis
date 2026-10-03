"""Text cleaning shared by all generation arms (blackbox/, feature_guided/, hybrid/): seed texts
are loaded identically everywhere via load_seed_examples(); feature_guided/ additionally uses
clean_text / to_single_line / assert_clean for feature annotations and spans."""

from __future__ import annotations

import csv
import re
from pathlib import Path

# Mojibake repair: text whose UTF-8 bytes were once mis-decoded as cp1252/latin-1 (e.g. "Letâ€™s"
# instead of "Let’s", "Â\xa0" instead of a no-break space). Every such mis-decoded character maps
# back to exactly one byte, so runs of those characters are re-encoded to bytes, and every valid
# UTF-8 multi-byte sequence inside them is decoded again. Bytes that do not form valid UTF-8 (e.g.
# a genuine "é" or "’") are left as they were, so correctly encoded text passes through unchanged.
_CP1252_TO_BYTE = {}
for _byte in range(0x80, 0xA0):
    try:
        _CP1252_TO_BYTE[bytes([_byte]).decode("cp1252")] = _byte
    except UnicodeDecodeError:
        pass  # undefined in cp1252 - such bytes show up as the raw C1 control character instead
_MOJIBAKE_RUN_RE = re.compile("[\x80-\xff" + re.escape("".join(_CP1252_TO_BYTE)) + "]+")
_UTF8_SEQUENCE_RE = re.compile(rb"[\xc2-\xdf][\x80-\xbf]|[\xe0-\xef][\x80-\xbf]{2}|[\xf0-\xf4][\x80-\xbf]{3}")


def _repair_mojibake_run(match: re.Match) -> str:
    run = match.group()
    raw = bytes(_CP1252_TO_BYTE.get(ch, ord(ch)) for ch in run)
    parts, pos = [], 0
    for seq in _UTF8_SEQUENCE_RE.finditer(raw):
        parts.append(run[pos:seq.start()])
        try:
            parts.append(seq.group().decode("utf-8"))
        except UnicodeDecodeError:
            parts.append(run[seq.start():seq.end()])
        pos = seq.end()
    parts.append(run[pos:])
    return "".join(parts)


def fix_mojibake(text: str) -> str:
    # A few rounds, in case text was mis-decoded more than once.
    for _ in range(3):
        fixed = _MOJIBAKE_RUN_RE.sub(_repair_mojibake_run, text)
        if fixed == text:
            break
        text = fixed
    return text


_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_NEWLINE_RUN_RE = re.compile(r"\s*\n\s*")
_SPACE_RUN_RE = re.compile(r"[ \t\r\f\v\u00a0\u202f]+")

TRUNCATION_MARKER = "… "
LINE_BREAK_MARKER = " / "


def clean_text(text: str) -> str:
    """Encoding cleanup shared by seeds, annotations and spans: mojibake repaired, zero-width
    characters dropped, no-break spaces turned into plain spaces. Backslashes are deliberately
    NOT unescaped here: in the feature spans they are genuine source text (code like
    'cout<<"x\\t"', Windows paths like 'T:\\Lab11data\\rain', LaTeX like '\\( e^{-x} \\)')."""
    text = fix_mojibake(text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return text.replace("\u00a0", " ").replace("\u202f", " ")


def to_single_line(text: str) -> str:
    """Line breaks become the visible " / " marker, other whitespace runs a single space."""
    text = _NEWLINE_RUN_RE.sub(LINE_BREAK_MARKER, text.strip())
    return _SPACE_RUN_RE.sub(" ", text).strip()


def assert_clean(text: str, where: str) -> None:
    """Final guard: no repairable mojibake left in text that goes into a prompt."""
    if fix_mojibake(text) != text:
        raise SystemExit(f"{where}: text still contains mojibake after cleaning: {text[:160]!r}")


# Seed-specific escape artifacts that must never survive into a prompt: CSV-doubled quotes,
# backslash-escaped quotes, literal "\n"/"\t" escape sequences, and raw line breaks / carriage
# returns (a seed is rendered as one line of the numbered SEEDS list).
_SEED_ARTIFACT_RE = re.compile(r'""|\\"|\\n|\\t|[\r\n]')


def clean_seed_text(text: str) -> str:
    """Unescapes one seed text as read by csv.reader: encoding cleanup, then both the literal "\\n"
    escape sequences the seed TSVs use for line breaks and any real line breaks become " / "."""
    return to_single_line(clean_text(text).replace("\\n", "\n"))


def load_seed_examples(seed_file: Path) -> list[str]:
    """Seed texts (first column), fully unescaped.

    Parsed with csv.reader rather than a plain split on tabs, because the seed TSVs are written
    CSV-style: a field containing quotes is wrapped in quotes with its inner quotes doubled
    ("...like ""I look around"" and..."). A plain split would pass those quote artifacts straight
    into the prompt. Fails fast if any escape artifact is still left after unescaping.
    """
    examples = []
    with open(seed_file, "r", encoding="utf-8", newline="") as f:
        for line_no, row in enumerate(csv.reader(f, delimiter="\t"), start=1):
            text = clean_seed_text(row[0]) if row else ""
            if not text:
                continue
            artifact = _SEED_ARTIFACT_RE.search(text)
            if artifact:
                raise SystemExit(
                    f"{seed_file}:{line_no}: seed still contains the escape artifact {artifact.group()!r} "
                    f"after unescaping: {text[:120]!r}"
                )
            assert_clean(text, f"{seed_file}:{line_no}")
            examples.append(text)
    if not examples:
        raise SystemExit(f"No seed examples found in {seed_file}")
    return examples
