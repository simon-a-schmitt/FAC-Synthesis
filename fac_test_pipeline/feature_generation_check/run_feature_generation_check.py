"""
Check whether synthetically generated CVE-description examples actually
trigger their intended SAE feature, e.g. for input/cti_vsp_holdout_f_miss_gen_run_01.json:

    {
      "call_index": 0,
      "feature_id": 504,
      ...
      "generated_example": "In the CDI 1002MFR, the Modbus TCP register ...",
      ...
    }

Input
-----
A JSON list of items (feature_generation_check/input), each with at least a
"feature_id" (the SAE feature the example was generated to activate) and a
"generated_example" (the candidate CVE description text) field. All other
fields are ignored for processing but carried through into the output record
for traceability.

Processing
----------
Each item's "generated_example" is wrapped in the CVSS-classification prompt
template exactly as in cvss_prompt.py / feature_coverage / feature_coverage_filter:
CVSS_SYSTEM_PROMPT as the system turn, "CVE Description: " + generated_example
as the user turn. This two-turn prompt is run through Llama-3.1-8B-Instruct +
the SAE hook, restricted to content tokens after the "CVE Description:" phrase
(the fixed instruction tokens are excluded), reusing
run_synthetic_feature_activation_check.compute_feature_activation_for_description
- which already implements exactly this single-target-feature, content-token-only
extraction - unchanged.

For each item's target feature_id, this records the maximum p95-baseline-normalised
magnitude and the maximum raw (pre-normalisation) magnitude reached on that
prompt's content tokens.

Output
------
Three files, all under feature_generation_check/output, derived from the
input filename stem:

1. `<stem>_feature_magnitudes.jsonl`
   One line per input item: call_index, feature_id, seed_example_id,
   n_content_tokens, max_normalized_magnitude, max_raw_magnitude,
   above_threshold, generated_example.

2. `<stem>_above_threshold_<threshold>.tsv`
   The raw `generated_example` text of every item whose max_normalized_magnitude
   exceeds --threshold (default 1.5), one per line, nothing else.

3. `<stem>_untriggered_feature_ids.json`
   A JSON array of feature_ids for which NONE of their items' examples
   exceeded --threshold (i.e. still need a better generated example).
"""

import argparse
import collections
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

import run_synthetic_feature_activation_check as sfac  # noqa: E402
import cvss_prompt as cp  # noqa: E402

_DEFAULT_INPUT_JSON = os.path.join(_SCRIPT_DIR, "input", "cti_vsp_holdout_f_miss_gen_run_01.json")
_DEFAULT_THRESHOLD = 1.5


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_generation_items(input_json: str) -> List[Dict[str, Any]]:
    """Read items (feature_id + generated_example, plus passthrough fields)."""
    with open(input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of items in {input_json}")

    items: List[Dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        feature_id = entry.get("feature_id")
        generated_example = entry.get("generated_example")
        if feature_id is None or not isinstance(generated_example, str) or not generated_example.strip():
            continue
        items.append(entry)

    if not items:
        raise ValueError(
            f"No usable items (feature_id + generated_example) found in {input_json}"
        )
    return items


# ---------------------------------------------------------------------------
# Output building
# ---------------------------------------------------------------------------

def build_magnitude_records(
    items: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item, result in zip(items, results):
        max_norm = result["max_normalized_magnitude"]
        max_raw = result["activations"][0]["raw_magnitude"] if result["activations"] else 0.0
        records.append({
            "call_index": item.get("call_index"),
            "feature_id": int(item["feature_id"]),
            "seed_example_id": item.get("seed_example_id"),
            "n_content_tokens": result["n_content_tokens"],
            "max_normalized_magnitude": max_norm,
            "max_raw_magnitude": round(max_raw, 6),
            "above_threshold": max_norm > threshold,
            "generated_example": item["generated_example"],
        })
    return records


def write_magnitudes_jsonl(records: List[Dict[str, Any]], output_jsonl: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_above_threshold_tsv(records: List[Dict[str, Any]], output_tsv: str) -> None:
    """One `generated_example` per line for every record above threshold, nothing else."""
    out_dir = os.path.dirname(os.path.abspath(output_tsv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_tsv, "w", encoding="utf-8") as f:
        for rec in records:
            if not rec["above_threshold"]:
                continue
            # Guard the one-line-per-prompt invariant even if a description
            # ever contains an embedded newline/tab.
            line = " ".join(rec["generated_example"].split())
            f.write(line + "\n")


def write_untriggered_feature_ids_json(records: List[Dict[str, Any]], output_json: str) -> None:
    """feature_ids for which NO record among their items is above threshold."""
    triggered: set = set()
    all_fids: set = set()
    for rec in records:
        fid = rec["feature_id"]
        all_fids.add(fid)
        if rec["above_threshold"]:
            triggered.add(fid)
    untriggered = sorted(all_fids - triggered)

    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(untriggered, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FAC feature-generation-check pipeline: wrap each item's "
            "generated_example in the CVSS-classification prompt template, run "
            "it through Llama-3.1-8B-Instruct + the SAE, and record the "
            "max normalised / raw magnitude reached by that item's target "
            "feature_id on the CVE-description content tokens."
        )
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sae-ckpt-path", type=str, default="")

    parser.add_argument("--input-json", type=str, default=_DEFAULT_INPUT_JSON)
    parser.add_argument("--max-items", type=int, default=0, help="0 = all")
    parser.add_argument("--hf-cache-dir", type=str, default=sfac._default_cache_dir())

    parser.add_argument("--baseline-tsv", type=str, default=sfac._DEFAULT_BASELINE_TSV)

    parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        help="Normalised-magnitude threshold above which an example counts as "
        "having triggered its target feature (default: 1.5).",
    )

    parser.add_argument(
        "--output-jsonl", type=str, default="",
        help="Destination JSONL for per-item magnitudes. Defaults to "
        "feature_generation_check/output/<input-json-stem>_feature_magnitudes.jsonl.",
    )
    parser.add_argument(
        "--output-tsv", type=str, default="",
        help="Destination TSV for above-threshold generated_example texts. Defaults to "
        "feature_generation_check/output/<input-json-stem>_above_threshold_<threshold>.tsv.",
    )
    parser.add_argument(
        "--output-untriggered-json", type=str, default="",
        help="Destination JSON array of feature_ids never triggered above threshold. "
        "Defaults to feature_generation_check/output/<input-json-stem>_untriggered_feature_ids.json.",
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

    stem = os.path.splitext(os.path.basename(args.input_json))[0]
    output_jsonl = args.output_jsonl or os.path.join(
        _SCRIPT_DIR, "output", f"{stem}_feature_magnitudes.jsonl"
    )
    output_tsv = args.output_tsv or os.path.join(
        _SCRIPT_DIR, "output", f"{stem}_above_threshold_{args.threshold}.tsv"
    )
    output_untriggered_json = args.output_untriggered_json or os.path.join(
        _SCRIPT_DIR, "output", f"{stem}_untriggered_feature_ids.json"
    )

    items = load_generation_items(args.input_json)
    if args.max_items > 0:
        items = items[: args.max_items]
    print(f"Loaded {len(items)} items from {args.input_json}")

    sae_ckpt = sfac.resolve_sae_checkpoint(local_path=args.sae_ckpt_path or None)
    baseline_full = sfac.load_baseline_full(args.baseline_tsv)
    print(f"Loaded baseline for {len(baseline_full)} features from {args.baseline_tsv}")

    model = sfac.UnifiedGenerator(
        model_path,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.hf_cache_dir,
        local_files_only=True,
        strict_local_paths=True,
    )
    collector = sfac.Collector(args.layer)
    sfac.mount_function(model._model, "llama", args.layer, collector)
    collector.early_stop = True

    sae = sfac.TopKSAE.from_disk(sae_ckpt, device=args.device)
    sae.topk = sfac.TOP_K
    sae.eval()

    # -----------------------------------------------------------------------
    # Step 1: for every item, run its CVSS-wrapped generated_example through
    # the model + SAE and record how strongly its target feature_id fired on
    # the CVE-description content tokens (opening-phrase-restricted).
    # -----------------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    with tc.no_grad():
        for item in tqdm.tqdm(items, desc="Processing generated examples"):
            feature_id = int(item["feature_id"])
            result = sfac.compute_feature_activation_for_description(
                item["generated_example"], feature_id, model, collector, sae, baseline_full,
                system=cp.CVSS_SYSTEM_PROMPT,
                opening_phrase=cp.CVSS_OPENING_PHRASE,
            )
            results.append(result)

    # -----------------------------------------------------------------------
    # Step 2: build & write the three output files.
    # -----------------------------------------------------------------------
    records = build_magnitude_records(items, results, args.threshold)
    write_magnitudes_jsonl(records, output_jsonl)
    print(f"Wrote {len(records)} per-item magnitude records to {output_jsonl}")

    write_above_threshold_tsv(records, output_tsv)
    n_above = sum(1 for r in records if r["above_threshold"])
    print(f"Wrote {n_above} above-threshold (>{args.threshold}) generated examples to {output_tsv}")

    write_untriggered_feature_ids_json(records, output_untriggered_json)
    n_fids = len(set(r["feature_id"] for r in records))
    n_untriggered = n_fids - len(set(r["feature_id"] for r in records if r["above_threshold"]))
    print(
        f"Wrote {n_untriggered}/{n_fids} still-untriggered feature_ids to "
        f"{output_untriggered_json}"
    )


if __name__ == "__main__":
    main()
