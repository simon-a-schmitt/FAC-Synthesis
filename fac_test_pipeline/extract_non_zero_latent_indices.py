import argparse
import json
from typing import Any, Iterable, List


def iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a dictionary on line {line_number}, got {type(payload).__name__}")
            yield payload


def non_zero_entries(values: list[Any]) -> List[tuple[int, Any]]:
    return [(index, value) for index, value in enumerate(values) if value != 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a JSONL file and print the non-zero index/value pairs in each latent array."
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="results_prompts_gen_casehold.jsonl",
        help="Path to the JSONL file to read.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for record in iter_jsonl(args.input_jsonl):
        latent = record.get("latent")
        if not isinstance(latent, list):
            raise ValueError("Each dictionary must contain a 'latent' field with a list value.")

        entries = non_zero_entries(latent)
        print(entries)


if __name__ == "__main__":
    main()
