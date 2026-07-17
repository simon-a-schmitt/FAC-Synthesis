import json
from pathlib import Path

INPUT_PATH = Path(__file__).parent / "output" / "blackbox.jsonl"
OUTPUT_PATH = Path(__file__).parent / "output" / "blackbox_sorted.jsonl"


def parse_entries(text):
    decoder = json.JSONDecoder()
    entries = []
    idx = 0
    length = len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        entry, end = decoder.raw_decode(text, idx)
        entries.append(entry)
        idx = end
    return entries


def main():
    text = INPUT_PATH.read_text(encoding="utf-8")
    entries = [e for e in parse_entries(text) if e.get("accept") is not False]

    entries.sort(
        key=lambda e: (
            e.get("seed_index_1"),
            e.get("seed_index_2"),
            e.get("transformation_operator"),
        )
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"{len(entries)} Einträge geschrieben nach {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
