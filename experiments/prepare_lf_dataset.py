"""Builds the LLaMA-Factory dataset of one fine-tuning run (gold/bb/fg/hybrid) under
experiments/runs/<run_id>/lf_data/ and checks it with a hard gate.

  source TSV   gold -> GOLD_TSV (benchmarks/<bench>/gold_ft_sets/), bb/fg/hybrid -> LABEL_TSV
  dataset      data_processing/build_llamafactory_dataset.py --tsv <source> --output LF_DATASET_JSON
  registry     LF_DATA_DIR/dataset_info.json with exactly one entry (name = run_id)

Gate (any violation is an error, nothing is moved into place):
  - number of examples == n_total
  - no empty output
  - the first n_examples instructions are the seed texts of the run's seed group, in seed-file
    order (seeds come first in both gold sets and labeled TSVs). Compared after
    shared.text_cleaning.clean_seed_text, since generation arms store the cleaned seed text;
    the "CVE Description: " prefix of cti_vsp is stripped first.
  - no example is longer than FT_CUTOFF_LEN tokens (LLaMA-Factory would silently cut it):
    counted like LLaMA-Factory's llama3 template encodes an SFT example - BOS + system slot +
    user slot as source, assistant slot as target, every element tokenized on its own with the
    base model's tokenizer (src/llamafactory/data/template.py; truncation happens when
    len(source) + len(target) > cutoff_len, data/processor/supervised.py).

Idempotent: if dataset + dataset_info.json exist and pass the gate, nothing is done.
--check only reports that state (exit 0 = prepared, 1 = not prepared) and changes nothing.

Usage:
    python experiments/prepare_lf_dataset.py <run_id> [--check]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from runs import PROJECT_DIR, load_config, parse_run_id, resolve

sys.path.insert(0, str(PROJECT_DIR / "data_synthesis"))
from shared import seed_derivation  # noqa: E402
from shared.text_cleaning import clean_seed_text  # noqa: E402

BUILD_SCRIPT = PROJECT_DIR / "data_processing" / "build_llamafactory_dataset.py"
CTI_VSP_PREFIX = "CVE Description: "
COLUMNS = {"system": "system", "prompt": "instruction", "query": "input", "response": "output"}


def dataset_info(r: dict) -> dict:
    return {r["lf_dataset_name"]: {"file_name": Path(r["lf_dataset_json"]).name, "columns": COLUMNS}}


# LLaMA-Factory template "llama3" (format_prefix = BOS, format_system, format_user, format_assistant).
LLAMA3_SLOTS = {
    "system": "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
    "user": "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "assistant": "{}<|eot_id|>",
}
_TOKENIZERS: dict = {}


def ft_token_lengths(data: list[dict], r: dict) -> list[int]:
    """Token length of every example as LLaMA-Factory builds it for SFT (source + target)."""
    if r["ft_template"] != "llama3":
        raise SystemExit(f"error: token-length gate only implements the llama3 template, got {r['ft_template']!r}.")
    if r["base_model"] not in _TOKENIZERS:
        from transformers import AutoTokenizer
        _TOKENIZERS[r["base_model"]] = AutoTokenizer.from_pretrained(r["base_model"])
    tok = _TOKENIZERS[r["base_model"]]

    def enc(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False)) if text else 0

    lengths = []
    for ex in data:
        # LLaMA-Factory's alpaca converter joins instruction and input with "\n".
        prompt = "\n".join(part for part in (ex.get("instruction", ""), ex.get("input", "")) if part)
        n = 1 + (enc(LLAMA3_SLOTS["system"].format(ex["system"])) if ex.get("system") else 0)  # 1 = BOS
        n += enc(LLAMA3_SLOTS["user"].format(prompt)) + enc(LLAMA3_SLOTS["assistant"].format(ex["output"]))
        lengths.append(n)
    return lengths


def normalize_seed(text: str, bench: str) -> str:
    if bench == "cti_vsp" and text.startswith(CTI_VSP_PREFIX):
        text = text[len(CTI_VSP_PREFIX):]
    return clean_seed_text(seed_derivation.canonicalize_example(text))


def gate(dataset_path: Path, r: dict, n_total: int) -> list[str]:
    """Violations of the gate (empty list = passed)."""
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    errors = []
    if len(data) != n_total:
        errors.append(f"{len(data)} examples, expected n_total={n_total}")
    empty = [i for i, ex in enumerate(data) if not str(ex.get("output", "")).strip()]
    if empty:
        errors.append(f"{len(empty)} example(s) with empty output (indices {empty[:10]})")
    seeds = [normalize_seed(t, r["bench"]) for t in seed_derivation.load_seed_texts(Path(r["seed_file"]))]
    head = [normalize_seed(ex.get("instruction", ""), r["bench"]) for ex in data[:len(seeds)]]
    mismatch = [i for i, (a, b) in enumerate(zip(seeds, head)) if a != b] + list(range(len(head), len(seeds)))
    if mismatch:
        errors.append(f"first {len(seeds)} instructions do not match the seed texts of {r['seed_file']} "
                      f"(mismatch at seed index {mismatch})")
    lengths = ft_token_lengths(data, r)
    too_long = [i for i, n in enumerate(lengths) if n > r["ft_cutoff_len"]]
    if too_long:
        errors.append(f"{len(too_long)} example(s) longer than ft_cutoff_len={r['ft_cutoff_len']} tokens "
                      f"(indices {too_long[:20]}{', ...' if len(too_long) > 20 else ''}; max {max(lengths)})")
    return errors


def is_prepared(r: dict, n_total: int) -> bool:
    dataset_path, info_path = Path(r["lf_dataset_json"]), Path(r["lf_dataset_info_json"])
    if not (dataset_path.is_file() and info_path.is_file()):
        return False
    if json.loads(info_path.read_text(encoding="utf-8")) != dataset_info(r):
        print(f"[prepare] {info_path} does not match the expected entry.")
        return False
    errors = gate(dataset_path, r, n_total)
    for e in errors:
        print(f"[prepare] existing dataset fails the gate: {e}")
    return not errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_id")
    parser.add_argument("--check", action="store_true",
                        help="Only check: exit 0 if the dataset is prepared and passes the gate, else 1.")
    args = parser.parse_args()

    config = load_config()
    try:
        r = resolve(parse_run_id(args.run_id), config)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    if r["arm"] == "gold":
        source = Path(r["gold_tsv"])
    elif "label_tsv" in r:
        source = Path(r["label_tsv"])
    else:
        raise SystemExit(f"error: arm {r['arm']!r} of {args.run_id} has no fine-tuning dataset.")
    n_total = config["global"]["n_total"]
    dataset_path, info_path = Path(r["lf_dataset_json"]), Path(r["lf_dataset_info_json"])

    if is_prepared(r, n_total):
        print(f"[prepare] {args.run_id}: dataset present and gate passed, nothing to do ({dataset_path}).")
        return
    if args.check:
        print(f"[prepare] {args.run_id}: not prepared ({dataset_path}).")
        sys.exit(1)
    if not source.is_file():
        raise SystemExit(f"error: source TSV {source} does not exist.")

    # Build next to the target and only move it into place once the gate has passed, so a failed
    # build never leaves a dataset behind that looks valid to the job script.
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dataset_path.with_name(dataset_path.name + ".tmp")
    print(f"[prepare] {args.run_id}: building {dataset_path} from {source}")
    subprocess.run([sys.executable, str(BUILD_SCRIPT), "--tsv", str(source), "--output", str(tmp_path)], check=True)

    errors = gate(tmp_path, r, n_total)
    if errors:
        tmp_path.unlink()
        raise SystemExit("error: gate failed for " + args.run_id + ":\n  " + "\n  ".join(errors))
    os.replace(tmp_path, dataset_path)
    info_path.write_text(json.dumps(dataset_info(r), indent=2) + "\n", encoding="utf-8")
    print(f"[prepare] {args.run_id}: gate passed ({n_total} examples, {r['seed_n_examples']} seeds first); "
          f"wrote {dataset_path} and {info_path}")


if __name__ == "__main__":
    main()
