"""
Compute, for every SAE feature, the maximum normalised (peak) magnitude
observed anywhere across a corpus of CLAUDETTE-TOS prompts.

Input
-----
A headerless TSV (feature_coverage/claudette_tos/input) with exactly two
tab-separated columns per line, as produced e.g. by
input/claudette_tos_test.tsv:

    <ToS sentence>\\t<LTD:Y|TER:N|...|ARB:N| unfairness-type vector string>

Only the first column (the sentence) is used.

Unlike the CVSS/CTI-VSP TSVs consumed by
feature_coverage/cti_vsp/run_feature_coverage.py, these sentences do NOT
already contain the classification instruction, so there is nothing to
extract via a normalize_cvss_prompt()-style split. Instead the fixed
CLAUDETTE-TOS instruction (CLAUDETTE_SYSTEM_PROMPT from claudette_prompt.py)
is added here as a SYSTEM turn, with the sentence run verbatim (via
claudette_prompt.build_claudette_user_content()) as the USER turn - matching
the genuine two-turn chat structure used by
feature_coverage_filter/claudette_tos/run_feature_coverage_filter.py and
benchmark_play_ground/run_claudette_benchmark.py ("plain" mode).

Processing
----------
Every prompt is run through Llama-3.1-8B-Instruct + the SAE hook exactly as
in run_fac_test_pipeline_feature_stats.py, giving per-feature
(activation_density, peak_magnitude, log_odds) for that prompt.
peak_magnitude is already the p95-baseline-normalised activation magnitude,
unless --raw-magnitude is passed, in which case the p95 normalisation is
skipped and the raw (unnormalised) SAE activation magnitudes are used
instead. The user turn is the ToS sentence verbatim, so there is no opening
phrase to skip past - the whole user turn is treated as content tokens (no
--opening-phrase option, unlike the CVSS pipelines).

Output
------
For every feature that was active on at least one prompt, the maximum
peak_magnitude observed across the whole corpus is written to a TSV in
feature_coverage/claudette_tos/output: columns feature_id,
max_normalized_magnitude (or max_raw_magnitude with --raw-magnitude),
n_prompts_active.
"""

import argparse
import collections
import csv
import os
import sys
from typing import Any, Dict, List, Tuple

import torch as tc
import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import run_fac_test_pipeline_feature_stats as fs  # noqa: E402
import claudette_prompt as clp  # noqa: E402

_DEFAULT_INPUT_TSV = os.path.join(_SCRIPT_DIR, "input", "claudette_tos_test.tsv")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_prompts_tsv(tsv_path: str) -> List[str]:
    """Read the ToS sentences (first column) of a headerless
    <sentence>\\t<unfairness_vector> TSV."""
    sentences: List[str] = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{tsv_path}:{line_no}: expected 2 tab-separated columns "
                    f"(sentence, unfairness_vector), got {len(row)}"
                )
            sentence = row[0].strip()
            if sentence:
                sentences.append(sentence)
    if not sentences:
        raise ValueError(f"No prompts found in {tsv_path}")
    return sentences


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_max_magnitude_per_feature(
    records: List[Dict[str, Any]],
) -> Dict[int, Tuple[float, int]]:
    """{feature_id: (max_peak_magnitude_over_corpus, n_prompts_active)}."""
    max_mag: Dict[int, float] = {}
    n_active: Dict[int, int] = collections.defaultdict(int)
    for rec in records:
        for fid_str, (_density, peak, _log_odds) in rec.get("features", {}).items():
            fid = int(fid_str)
            n_active[fid] += 1
            if peak > max_mag.get(fid, float("-inf")):
                max_mag[fid] = peak
    return {fid: (max_mag[fid], n_active[fid]) for fid in max_mag}


def write_coverage_tsv(
    output_tsv: str,
    stats: Dict[int, Tuple[float, int]],
    raw_magnitude: bool = False,
) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_tsv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    magnitude_col = "max_raw_magnitude" if raw_magnitude else "max_normalized_magnitude"
    with open(output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["feature_id", magnitude_col, "n_prompts_active"])
        for fid in sorted(stats):
            max_mag, n = stats[fid]
            writer.writerow([fid, round(max_mag, 6), n])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FAC feature-coverage pipeline: run every CLAUDETTE-TOS sentence "
            "in a TSV through Llama-3.1-8B-Instruct + the SAE, and record "
            "the maximum p95-normalised magnitude reached by each feature "
            "across the whole corpus."
        )
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sae-ckpt-path", type=str, default="")

    parser.add_argument(
        "--input-tsv",
        type=str,
        default=_DEFAULT_INPUT_TSV,
        help=(
            "Headerless TSV with two tab-separated columns per line: "
            "<ToS sentence>\\t<unfairness-type vector string>. Only the "
            "sentence column is used."
        ),
    )
    parser.add_argument("--max-prompts", type=int, default=0, help="0 = all")
    parser.add_argument("--hf-cache-dir", type=str, default=fs._default_cache_dir())

    parser.add_argument("--baseline-tsv", type=str, default=fs._DEFAULT_BASELINE_TSV)

    parser.add_argument(
        "--raw-magnitude",
        action="store_true",
        help=(
            "Skip the p95-baseline normalisation and extract the raw "
            "(unnormalised) SAE activation magnitudes instead."
        ),
    )

    parser.add_argument(
        "--output-tsv",
        type=str,
        default="",
        help="Destination TSV. Defaults to "
        "feature_coverage/claudette_tos/output/<input-tsv-stem>_max_magnitude.tsv.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir
    os.makedirs(args.hf_cache_dir, exist_ok=True)

    if args.device == "cuda":
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    model_path = os.path.abspath(args.model_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Local model directory not found: {model_path}")

    if not args.baseline_tsv or not os.path.isfile(args.baseline_tsv):
        raise FileNotFoundError(f"Baseline TSV not found: {args.baseline_tsv}")

    if not os.path.isfile(args.input_tsv):
        raise FileNotFoundError(f"Input TSV not found: {args.input_tsv}")

    output_tsv = args.output_tsv
    if not output_tsv:
        stem = os.path.splitext(os.path.basename(args.input_tsv))[0]
        output_tsv = os.path.join(_SCRIPT_DIR, "output", f"{stem}_max_magnitude.tsv")

    sentences = load_prompts_tsv(args.input_tsv)
    if args.max_prompts > 0:
        sentences = sentences[: args.max_prompts]
    print(f"Loaded {len(sentences)} prompts from {args.input_tsv}")

    sae_ckpt = fs.resolve_sae_checkpoint(local_path=args.sae_ckpt_path or None)
    baseline_full = fs.load_baseline_full(args.baseline_tsv)
    print(f"Loaded baseline for {len(baseline_full)} features from {args.baseline_tsv}")

    model = fs.UnifiedGenerator(
        model_path,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.hf_cache_dir,
        local_files_only=True,
        strict_local_paths=True,
    )
    collector = fs.Collector(args.layer)
    fs.mount_function(model._model, "llama", args.layer, collector)
    collector.early_stop = True

    sae = fs.TopKSAE.from_disk(sae_ckpt, device=args.device)
    sae.topk = fs.TOP_K
    sae.eval()

    # -----------------------------------------------------------------------
    # Step 1: build the CLAUDETTE-TOS classification prompt for every
    # sentence and extract per-prompt feature stats (identical to
    # feature-stats pipeline). The user turn is the ToS sentence verbatim,
    # so there is no opening phrase to skip past - the whole user turn is
    # content.
    # -----------------------------------------------------------------------
    records: List[Dict[str, Any]] = []
    with tc.no_grad():
        for idx, sentence in enumerate(tqdm.tqdm(sentences, desc="Processing prompts")):
            user_content = clp.build_claudette_user_content(sentence)
            record = fs.compute_feature_stats_for_prompt(
                user_content, model, collector, sae, baseline_full,
                system=clp.CLAUDETTE_SYSTEM_PROMPT,
                raw_magnitude=args.raw_magnitude,
            )
            record["index"] = idx
            records.append(record)

    # -----------------------------------------------------------------------
    # Step 2: per-feature maximum normalised magnitude over the whole corpus
    # -----------------------------------------------------------------------
    stats = compute_max_magnitude_per_feature(records)
    print(f"{len(stats)} features active across {len(records)} prompts.")

    write_coverage_tsv(output_tsv, stats, raw_magnitude=args.raw_magnitude)
    print(f"Wrote max-magnitude coverage for {len(stats)} features to {output_tsv}")


if __name__ == "__main__":
    main()
