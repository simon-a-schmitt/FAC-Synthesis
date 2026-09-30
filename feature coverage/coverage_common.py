"""Shared logic of run_feature_coverage_reference.py and run_feature_coverage.py.

Both scripts must measure feature activity in exactly the same way, and exactly the way the
feature-guided generation branch verifies candidates (data_synthesis/feature_guided/run_generation.py):
every sample is wrapped in the domain's classification prompt (system = SYSTEM_PROMPT from
data_synthesis/labeling/<path>/prompt/prompt_step_2.py, user = sample verbatim, prefixed with the
prompt module's optional USER_PROMPT_PREFIX as in run_labeling.py), run through Llama-3.1-8B + the SAE,
and a feature counts as active at threshold t if its RAW (not p95-normalised) activation exceeds t on at
least one content token (= user-turn tokens, special tokens and the USER_PROMPT_PREFIX excluded) of at
least one sample.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_SYNTHESIS_DIR = PROJECT_DIR / "data_synthesis"
LABELING_DIR = DATA_SYNTHESIS_DIR / "labeling"
FEATURE_SCORES_DIR = DATA_SYNTHESIS_DIR / "data" / "feature_scores"
FEATURE_GUIDED_SCRIPT = DATA_SYNTHESIS_DIR / "feature_guided" / "run_generation.py"

DOMAINS = ("cti_vsp", "claudette_tos", "toxicity_detection")
CRITERION = "raw SAE activation > t on at least one content token of at least one sample"


def _load_feature_guided_module():
    spec = importlib.util.spec_from_file_location("fc_feature_guided_run_generation", FEATURE_GUIDED_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Must be registered before exec: its @dataclass definitions look themselves up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# SAE loading and prompt-module loading are imported, not copied, so the coverage measurement can
# never drift from the feature-guided verification.
fg = _load_feature_guided_module()


def threshold_key(t: float) -> str:
    return f"{t:g}"


def domain_output_dir(path: str) -> Path:
    return SCRIPT_DIR / path


def load_prompt_config(path: str) -> tuple[str, str, Path]:
    """Returns (SYSTEM_PROMPT, USER_PROMPT_PREFIX or "", prompt module path) of the domain's
    labeling prompt."""
    prompt_path = LABELING_DIR / path / "prompt" / "prompt_step_2.py"
    module = fg._load_module(prompt_path, f"fc_prompt_step_2_{path}", ("SYSTEM_PROMPT",))
    return module.SYSTEM_PROMPT, getattr(module, "USER_PROMPT_PREFIX", ""), prompt_path


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


def compute_peak_activations(texts: list[str], sae_ctx, user_prefix: str = ""):
    """Float tensor [n_features]: max raw SAE activation of every feature over the content tokens of
    all samples. Per sample identical to compute_target_activation() in feature_guided/run_generation.py,
    but for all features at once."""
    import tqdm

    fs = sae_ctx.fs
    tc = fs.tc
    model, collector = sae_ctx.model, sae_ctx.collector
    tokenizer = model._tokenizer  # noqa: SLF001
    special_ids = fs._special_token_id_set(tokenizer)
    opening_phrase = user_prefix.strip()

    peaks = None
    n_without_content = 0
    for text in tqdm.tqdm(texts, desc="SAE forward passes"):
        user_text = f"{user_prefix}{text}"
        with tc.no_grad():
            collector.cache = None
            _, token_ids, tokens = fs._encode_prompt_tokens(user_text, model, system=sae_ctx.system_prompt)
            try:
                model.get_activates(user_text, system=sae_ctx.system_prompt)
            except RuntimeError:
                pass  # early_stop raises right after the hooked layer
            if collector.cache is None:
                raise RuntimeError("Collector cache is empty. Hook may not be mounted correctly.")

            hidden_seq = collector.cache.to(tc.float32)[0]
            if hidden_seq.shape[0] != len(tokens):
                seq_len = min(hidden_seq.shape[0], len(tokens))
                hidden_seq = hidden_seq[:seq_len]
                token_ids = token_ids[:seq_len]
                tokens = tokens[:seq_len]
            sparse_features = sae_ctx.sae.encode(hidden_seq).detach().cpu()

        if peaks is None:
            peaks = tc.zeros(sparse_features.shape[1], dtype=tc.float32)
        content_start = fs._find_user_content_start(token_ids, tokens, tokenizer)
        if opening_phrase:
            content_start = fs._advance_past_opening_phrase(token_ids, content_start, tokenizer, opening_phrase)
        content_positions = [
            idx for idx, token_id in enumerate(token_ids) if idx >= content_start and token_id not in special_ids
        ]
        if not content_positions:
            n_without_content += 1
            continue
        peaks = tc.maximum(peaks, sparse_features[content_positions].max(dim=0).values)

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
    system_prompt, user_prefix, prompt_path = load_prompt_config(args.path)
    print(f"Loading Llama + SAE (prompt: {prompt_path})...")
    return Runner(sae_ctx=fg.load_sae_context(args, system_prompt), user_prefix=user_prefix, prompt_path=prompt_path)


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

    peaks = compute_peak_activations(texts, runner.sae_ctx, runner.user_prefix)

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
