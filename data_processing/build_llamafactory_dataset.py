"""Convert a labeled blackbox TSV (prompt<TAB>label) into a LLaMA-Factory-style JSON
dataset.

The task a TSV belongs to is inferred from the shape of its label column, not from the
file path, so any headerless "<text>\\t<label>" TSV produced by run_labeling.py can be
pointed at directly:

    - "Answer: safe" / "Answer: toxic"                              -> toxicity_detection
    - "CVSS:3.1/AV:.../AC:.../.../A:..."                             -> cti_vsp
    - "LTD: ...|TER: ...|CH: ...|CR: ...|USE: ...|LAW: ...|J: ...|ARB: ..." -> claudette_tos

For all three tasks, the system prompt is loaded from each domain's
data_synthesis/prompts/<domain>/labeling.py (SYSTEM_PROMPT), the same module
run_labeling.py uses to obtain model labels in the first place. For cti_vsp, the
instruction is additionally prefixed with "CVE Description: ".

Writes <tsv-stem>.json to data_processing/output/ (or to --output), as a JSON list of
    {"system": ..., "instruction": ..., "input": "", "output": ...}
entries.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

TOXICITY_LABEL_RE = re.compile(r"^Answer:\s*(safe|toxic)\s*$", re.IGNORECASE)
CTI_VSP_LABEL_RE = re.compile(r"^CVSS:3\.1/AV:", re.IGNORECASE)
CLAUDETTE_TOS_LABEL_RE = re.compile(r"^LTD:.*\|TER:.*\|CH:.*\|CR:.*\|USE:.*\|LAW:.*\|J:.*\|ARB:", re.IGNORECASE)


def detect_task(label: str) -> str:
    if TOXICITY_LABEL_RE.match(label):
        return "toxicity_detection"
    if CTI_VSP_LABEL_RE.match(label):
        return "cti_vsp"
    if CLAUDETTE_TOS_LABEL_RE.match(label):
        return "claudette_tos"
    raise SystemExit(f"Could not detect task for label: {label!r}")


def load_rows(tsv_path: Path) -> list[tuple[str, str]]:
    """Reads (text, label) pairs from a headerless TSV. Rows with a blank label - e.g. a
    call that failed or didn't parse, per run_labeling.py's convention - are skipped and
    counted, since they carry no ground truth to train on.
    """
    rows: list[tuple[str, str]] = []
    n_skipped_blank = 0
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2 or not row[0].strip():
                continue
            label = row[1].strip()
            if not label:
                n_skipped_blank += 1
                continue
            rows.append((row[0], label))
    if not rows:
        raise SystemExit(f"No labeled rows found in {tsv_path}.")
    if n_skipped_blank:
        print(f"Skipped {n_skipped_blank} row(s) with a blank label.")
    return rows


def load_system_prompt(domain: str) -> str:
    prompt_path = BASE_DIR.parent / "data_synthesis" / "prompts" / domain / "labeling.py"
    if not prompt_path.exists():
        raise SystemExit(f"Expected prompt module at {prompt_path}, but it does not exist.")
    spec = importlib.util.spec_from_file_location(f"labeling_prompt_{domain}", prompt_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "SYSTEM_PROMPT"):
        raise SystemExit(f"{prompt_path} is missing required attribute 'SYSTEM_PROMPT'.")
    return module.SYSTEM_PROMPT


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a LLaMA-Factory-style JSON dataset from a labeled blackbox TSV."
    )
    parser.add_argument("--tsv", type=Path, required=True, help="Path to the labeled TSV file (prompt<TAB>label).")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"Output JSON path (default: {OUTPUT_DIR}/<tsv-stem>.json).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    tsv_path = args.tsv
    if not tsv_path.exists():
        raise SystemExit(f"--tsv '{tsv_path}' does not exist.")

    rows = load_rows(tsv_path)
    tasks = {detect_task(label) for _, label in rows}
    if len(tasks) > 1:
        raise SystemExit(f"{tsv_path} contains rows for multiple tasks: {sorted(tasks)}. Expected a single task.")
    task = tasks.pop()
    print(f"Detected task: {task} ({len(rows)} row(s))")

    system_prompt = load_system_prompt(task)
    if task == "cti_vsp":
        dataset = [
            {"system": system_prompt, "instruction": f"CVE Description: {text}", "input": "", "output": label}
            for text, label in rows
        ]
    else:
        dataset = [
            {"system": system_prompt, "instruction": text, "input": "", "output": label}
            for text, label in rows
        ]

    out_path = args.output or OUTPUT_DIR / f"{tsv_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(dataset)} example(s) to {out_path}")


if __name__ == "__main__":
    main()
