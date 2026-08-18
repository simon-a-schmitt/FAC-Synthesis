"""
Draw a random subsample of pool items from the input JSON and write them to
output, in the same format as the input file.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_INPUT_JSON = os.path.join(
    _SCRIPT_DIR, "input", "cti_vsp_bb_pool_deepseek_1025.json"
)


def load_input_records(input_json: str) -> List[Dict[str, Any]]:
    with open(input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of pool items in {input_json}")
    return payload


def write_records_json(records: List[Dict[str, Any]], output_json: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a random subsample of pool items from an input JSON file."
    )
    parser.add_argument("--input-json", type=str, default=_DEFAULT_INPUT_JSON)
    parser.add_argument("--n", type=int, default=200, help="Number of items to sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Destination JSON. Defaults to "
        "feature_coverage_filter/output/<input-json-stem>_random<n>_seed<seed>.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input_json):
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    records = load_input_records(args.input_json)
    print(f"Loaded {len(records)} records from {args.input_json}")

    n = min(args.n, len(records))
    rng = random.Random(args.seed)
    sampled = rng.sample(records, n)

    output_json = args.output_json
    if not output_json:
        stem = os.path.splitext(os.path.basename(args.input_json))[0]
        output_json = os.path.join(
            _SCRIPT_DIR, "output", f"{stem}_random{n}_seed{args.seed}.json"
        )

    write_records_json(sampled, output_json)
    print(f"Wrote {n} randomly sampled records to {output_json}")


if __name__ == "__main__":
    main()
