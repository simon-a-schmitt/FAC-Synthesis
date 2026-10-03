"""Shared logic of run_feature_coverage_reference.py and run_feature_coverage.py.

Both scripts must measure feature activity in exactly the same way, and exactly the way the
generation arms check seeds, blackbox examples and candidates: every sample goes through
data_synthesis/shared/feature_guidance.sae_forward() - the domain's classification prompt (system =
SYSTEM_PROMPT from data_synthesis/prompts/<path>/labeling.py, user = USER_PROMPT_PREFIX + sample),
Llama-3.1-8B + the SAE - and a feature counts as active at threshold t if its RAW (not
p95-normalised) activation exceeds t on at least one content token (= user-turn tokens, special
tokens and the USER_PROMPT_PREFIX excluded) of at least one sample.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_SYNTHESIS_DIR = PROJECT_DIR / "data_synthesis"
sys.path.insert(0, str(DATA_SYNTHESIS_DIR))

# The SAE check is imported, not copied, so the coverage measurement can never drift from generation.
from shared import feature_guidance as fgd  # noqa: E402
from shared.benchmarks import DOMAINS, feature_scores_path  # noqa: E402,F401

CRITERION = "raw SAE activation > t on at least one content token of at least one sample"


def threshold_key(t: float) -> str:
    return f"{t:g}"


def domain_output_dir(path: str) -> Path:
    return SCRIPT_DIR / path


def load_tsv_texts(tsv_path: Path) -> list[str]:
    """Sample texts (first column) of a headerless <text>\\t<label> TSV."""
    texts = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and row[0].strip():
                texts.append(row[0])
    if not texts:
        raise SystemExit(f"No samples found in {tsv_path}.")
    return texts


def compute_peak_activations(texts: list[str], sae_ctx):
    """Float tensor [n_features]: max raw SAE activation of every feature over the content tokens of
    all samples (fgd.sae_forward per sample)."""
    import tqdm

    peaks = None
    n_without_content = 0
    for text in tqdm.tqdm(texts, desc="SAE forward passes"):
        forward = fgd.sae_forward(text, sae_ctx)
        if peaks is None:
            peaks = sae_ctx.fs.tc.zeros(forward.content_features.shape[1], dtype=sae_ctx.fs.tc.float32)
        if not forward.content_positions:
            n_without_content += 1
            continue
        peaks = sae_ctx.fs.tc.maximum(peaks, forward.content_features.max(dim=0).values)

    if n_without_content:
        print(f"[warn] {n_without_content} sample(s) had no content tokens and were skipped.", file=sys.stderr)
    return peaks


def active_ids(peaks, t: float) -> list[int]:
    return sorted(int(i) for i in (peaks > t).nonzero(as_tuple=True)[0].tolist())


def add_common_args(parser: argparse.ArgumentParser, multiple_inputs: bool = False) -> None:
    parser.add_argument("--path", type=str, required=True, choices=DOMAINS, help="Domain.")
    if multiple_inputs:
        parser.add_argument(
            "--input-tsv", type=Path, nargs="+", required=True,
            help="One or more headerless <text>\\t<label> TSVs, each processed independently.",
        )
    else:
        parser.add_argument("--input-tsv", type=Path, required=True, help="Headerless <text>\\t<label> TSV.")
    parser.add_argument(
        "--thresholds", type=float, nargs="+", required=True,
        help="Raw SAE activation thresholds t; a feature is active if its activation exceeds t on at least one content token.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Only use the first N samples (0 = all).")
    parser.add_argument("--model-name", type=str, required=True, help="Local Llama-3.1-8B-Instruct directory.")
    parser.add_argument("--sae-ckpt-path", type=str, required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--hf-cache-dir", type=str, default=os.environ.get("TRANSFORMERS_CACHE", ""))


@dataclass
class Runner:
    sae_ctx: Any
    user_prefix: str
    prompt_path: Path


def load_runner(args: argparse.Namespace) -> Runner:
    """Loads the domain's labeling prompt and Llama + SAE once, for any number of TSVs."""
    args.domain = args.path
    sae_ctx = fgd.load_sae_context(args)
    prompt_path = DATA_SYNTHESIS_DIR / "prompts" / args.path / "labeling.py"
    return Runner(sae_ctx=sae_ctx, user_prefix=sae_ctx.prompt.user_prefix, prompt_path=prompt_path)


def run_sae_on_tsv(
    args: argparse.Namespace, input_tsv: Path, runner: Runner
) -> tuple[list[str], dict[str, list[int]], dict]:
    """Runs the samples of `input_tsv` through Llama + SAE and returns
    (texts, {threshold_key: active feature ids}, metadata)."""
    if not input_tsv.is_file():
        raise SystemExit(f"Input TSV not found: {input_tsv}")
    texts = load_tsv_texts(input_tsv)
    if args.max_samples > 0:
        texts = texts[: args.max_samples]
    print(f"Loaded {len(texts)} sample(s) from {input_tsv}")

    peaks = compute_peak_activations(texts, runner.sae_ctx)

    active = {threshold_key(t): active_ids(peaks, t) for t in args.thresholds}
    metadata = {
        "path": args.path,
        "input_tsv": str(input_tsv.resolve()),
        "n_samples": len(texts),
        "system_prompt_source": str(runner.prompt_path),
        "user_prompt_prefix": runner.user_prefix,
        "sae_model": os.path.abspath(args.model_name),
        "sae_ckpt_path": os.path.abspath(args.sae_ckpt_path),
        "sae_layer": args.layer,
        "n_sae_features": int(peaks.shape[0]),
        "criterion": CRITERION,
    }
    return texts, active, metadata
