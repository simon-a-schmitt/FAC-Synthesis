"""
Build a pool_id -> active-feature mapping for a CLAUDETTE-TOS prompt pool.

Input
-----
A headerless TSV file (feature_coverage_filter/claudette_tos/input) with
exactly two tab-separated columns per line, as produced e.g. by
input/claudette_tos_train.tsv:

    <ToS sentence>\t<LTD:Y|TER:N|...|ARB:N| unfairness-type vector string>

There is no id column, so each line's 1-based line number is used as its
pool_id.

Processing
----------
Each row's sentence is run, unmodified, as the USER turn of the CLAUDETTE-TOS
classification prompt (SYSTEM turn = CLAUDETTE_SYSTEM_PROMPT), through
Llama-3.1-8B-Instruct + the SAE hook exactly as in
run_fac_test_pipeline_feature_stats.py. This is completely analogous to
feature_coverage_filter/cti_vsp/run_feature_coverage_filter.py, with prompt
construction delegated to claudette_prompt.py (the CLAUDETTE counterpart of
cvss_prompt.py) instead of cvss_prompt.py, so the same processing logic
(SYSTEM/USER prompt structure, model + SAE call, feature-stats extraction) is
shared with benchmark_play_ground/run_claudette_benchmark.py's "plain" mode.

Unlike the CVSS pipeline, the CLAUDETTE user turn has no fixed opening phrase
preceding the content (the user turn *is* the ToS sentence verbatim), so
every token of the user turn is treated as a content token - no
--opening-phrase is used.

For every feature that fired on those content tokens, the maximum
p95-baseline-normalised activation magnitude (peak_magnitude) on that prompt
is recorded, unless --raw-magnitude is passed, in which case the p95
normalisation is skipped and the raw (unnormalised) SAE activation
magnitudes are used instead.

Output
------
A JSONL file (feature_coverage_filter/claudette_tos/output) with one line per
input row:

    {"pool_id": 3, "n_content_tokens": 41, "features": {"1234": 0.8123, ...}}

`pool_id` is the 1-based line number of the row in the input TSV, so this
file can later be used to filter/join back against the original input file.
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List

import torch as tc
import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import run_fac_test_pipeline_feature_stats as fs  # noqa: E402
import claudette_prompt as clp  # noqa: E402

_DEFAULT_INPUT_TSV = os.path.join(_SCRIPT_DIR, "input", "claudette_tos_train.tsv")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_pool_items(input_tsv: str) -> List[Dict[str, Any]]:
    """Read pool items (pool_id + text) from a headerless <sentence>\\t<vector> TSV."""
    items: List[Dict[str, Any]] = []
    with open(input_tsv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{input_tsv}:{line_no}: expected 2 tab-separated columns "
                    f"(sentence, unfairness_vector), got {len(row)}"
                )
            text = row[0]
            if not text.strip():
                continue
            items.append({"pool_id": line_no, "text": text})

    if not items:
        raise ValueError(f"No valid pool rows (sentence + unfairness_vector) found in {input_tsv}")
    return items


# ---------------------------------------------------------------------------
# Output building
# ---------------------------------------------------------------------------

def build_pool_records(
    items: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """{pool_id, n_content_tokens, features: {feature_id: peak_magnitude}}."""
    out: List[Dict[str, Any]] = []
    for item, rec in zip(items, records):
        features = {
            fid_str: peak
            for fid_str, (_density, peak, _log_odds) in rec.get("features", {}).items()
        }
        out.append({
            "pool_id": item["pool_id"],
            "n_content_tokens": rec.get("n_content_tokens", 0),
            "features": features,
        })
    return out


def write_pool_records_jsonl(records: List[Dict[str, Any]], output_jsonl: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FAC feature-coverage-filter pipeline: build the CLAUDETTE-TOS "
            "classification prompt for every pool item, run it through "
            "Llama-3.1-8B-Instruct + the SAE, and record the active features and "
            "their max p95-normalised magnitude, keyed by pool_id."
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
            "<ToS sentence>\\t<unfairness-type vector string>. Each line's "
            "1-based line number is used as its pool_id."
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
        "--output-jsonl",
        type=str,
        default="",
        help="Destination JSONL. Defaults to "
        "feature_coverage_filter/claudette_tos/output/<input-tsv-stem>_feature_magnitudes.jsonl.",
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

    output_jsonl = args.output_jsonl
    if not output_jsonl:
        stem = os.path.splitext(os.path.basename(args.input_tsv))[0]
        output_jsonl = os.path.join(_SCRIPT_DIR, "output", f"{stem}_feature_magnitudes.jsonl")

    items = load_pool_items(args.input_tsv)
    if args.max_prompts > 0:
        items = items[: args.max_prompts]
    print(f"Loaded {len(items)} pool items from {args.input_tsv}")

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
    # Step 1: build the CLAUDETTE-TOS classification prompt for every pool
    # item and extract per-prompt feature stats (identical to feature-stats
    # pipeline). The user turn is the ToS sentence verbatim, so there is no
    # opening phrase to skip past - the whole user turn is content.
    # -----------------------------------------------------------------------
    records: List[Dict[str, Any]] = []
    with tc.no_grad():
        for item in tqdm.tqdm(items, desc="Processing prompts"):
            user_content = clp.build_claudette_user_content(item["text"])
            record = fs.compute_feature_stats_for_prompt(
                user_content, model, collector, sae, baseline_full,
                system=clp.CLAUDETTE_SYSTEM_PROMPT,
                raw_magnitude=args.raw_magnitude,
            )
            records.append(record)

    # -----------------------------------------------------------------------
    # Step 2: build the pool_id -> active-feature mapping and write it
    # -----------------------------------------------------------------------
    pool_records = build_pool_records(items, records)
    write_pool_records_jsonl(pool_records, output_jsonl)
    n_with_features = sum(1 for r in pool_records if r["features"])
    print(
        f"Wrote {len(pool_records)} pool records ({n_with_features} with active "
        f"features) to {output_jsonl}"
    )


if __name__ == "__main__":
    main()
