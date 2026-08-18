"""
Build a pool_id -> active-feature mapping for a CVSS-vector prompt pool.

Input
-----
A JSON list of pool items (feature_coverage_filter/input), each shaped like:

    {
      "pool_id": 3,
      "type": "synthetic",
      "text": "The JNews WordPress theme through 10.0.1 does not validate ...",
      "seed_example_id": 3,
      "batch_id": null,
      "run_id": null,
      "added_at": "2026-08-11T14:17:57.175538+00:00"
    }

Processing
----------
Each item's "text" field is appended to a fixed CVSS-classification
instruction (ending in the literal phrase "CVE Description:") to form the
full prompt that is run through Llama-3.1-8B-Instruct + the SAE hook exactly
as in run_fac_test_pipeline_feature_stats.py. The opening-phrase mechanism
from that module is used so only tokens *after* "CVE Description:" (i.e. the
CVE description text itself, not the fixed instruction) are treated as
content tokens and considered for feature-activation stats.

For every feature that fired on those content tokens, the maximum
p95-baseline-normalised activation magnitude (peak_magnitude) on that prompt
is recorded.

Output
------
A JSONL file (feature_coverage_filter/output) with one line per input item:

    {"pool_id": 3, "n_content_tokens": 41, "features": {"1234": 0.8123, ...}}

`pool_id` matches the `pool_id` field in the input JSON, so this file can
later be used to filter/join back against the original input file.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch as tc
import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_SCRIPT_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import run_fac_test_pipeline_feature_stats as fs  # noqa: E402
import cvss_prompt as cp  # noqa: E402

_DEFAULT_INPUT_JSON = os.path.join(_SCRIPT_DIR, "input", "cti_vsp_bb_pool_deepseek_1025.json")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_pool_items(input_json: str) -> List[Dict[str, Any]]:
    """Read pool items (pool_id + text) from the input JSON list."""
    with open(input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of pool items in {input_json}")

    items: List[Dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        pool_id = entry.get("pool_id")
        text = entry.get("text")
        if pool_id is None or not isinstance(text, str) or not text.strip():
            continue
        items.append({"pool_id": pool_id, "text": text})

    if not items:
        raise ValueError(f"No valid pool items (pool_id + text) found in {input_json}")
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
            "FAC feature-coverage-filter pipeline: build the CVSS-classification "
            "prompt for every pool item, run it through Llama-3.1-8B-Instruct + the "
            "SAE, and record the active features and their max p95-normalised "
            "magnitude, keyed by pool_id."
        )
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sae-ckpt-path", type=str, default="")

    parser.add_argument("--input-json", type=str, default=_DEFAULT_INPUT_JSON)
    parser.add_argument("--max-prompts", type=int, default=0, help="0 = all")
    parser.add_argument("--hf-cache-dir", type=str, default=fs._default_cache_dir())

    parser.add_argument("--baseline-tsv", type=str, default=fs._DEFAULT_BASELINE_TSV)

    parser.add_argument(
        "--opening-phrase",
        type=str,
        default=cp.CVSS_OPENING_PHRASE,
        help=(
            "Only tokens after this phrase (excluding the phrase itself) within "
            "each prompt's content are considered for feature-activation stats. "
            f"Defaults to {cp.CVSS_OPENING_PHRASE!r}, i.e. the fixed instruction preceding "
            "the CVE description is excluded."
        ),
    )

    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="",
        help="Destination JSONL. Defaults to "
        "feature_coverage_filter/output/<input-json-stem>_feature_magnitudes.jsonl.",
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

    if not os.path.isfile(args.input_json):
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    output_jsonl = args.output_jsonl
    if not output_jsonl:
        stem = os.path.splitext(os.path.basename(args.input_json))[0]
        output_jsonl = os.path.join(_SCRIPT_DIR, "output", f"{stem}_feature_magnitudes.jsonl")

    items = load_pool_items(args.input_json)
    if args.max_prompts > 0:
        items = items[: args.max_prompts]
    print(f"Loaded {len(items)} pool items from {args.input_json}")

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
    # Step 1: build the CVSS-classification prompt for every pool item and
    # extract per-prompt feature stats (identical to feature-stats pipeline),
    # restricted to tokens after --opening-phrase ("CVE Description:").
    # -----------------------------------------------------------------------
    records: List[Dict[str, Any]] = []
    with tc.no_grad():
        for item in tqdm.tqdm(items, desc="Processing prompts"):
            user_content = cp.build_cvss_user_content(item["text"])
            record = fs.compute_feature_stats_for_prompt(
                user_content, model, collector, sae, baseline_full,
                opening_phrase=args.opening_phrase or None,
                system=cp.CVSS_SYSTEM_PROMPT,
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
