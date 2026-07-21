"""
Build a dense prompt x feature statistics matrix for the seed examples.

Per prompt and feature this computes the same three values as
run_fac_test_pipeline_feature_stats.py:

    (activation_density, peak_magnitude, log_odds)

Features are then filtered exactly like in the feature-stats pipeline:

  1. Drop features with mean_activation_density > 0.5 (too ubiquitous).
  2. Drop features with max_peak_magnitude < 0.3 (never strongly active).

The remaining feature ids (i.e. every feature still active on at least one
prompt after filtering) become the column axis; every input prompt is a row.
The result is written as a dense [n_prompts, n_features] matrix per stat, so
that a single cell, or the max/min over one row/column, can be looked up
without scanning the whole thing. See feature_matrix_io.py for the reader.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch as tc
import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import run_fac_test_pipeline_feature_stats as fs  # noqa: E402

_DEFAULT_PROMPTS_JSON = os.path.join(_SCRIPT_DIR, "input", "prompts_seed_casehold.json")


def build_feature_matrix(
    records: List[Dict[str, Any]],
    kept_feature_ids: List[int],
) -> Dict[str, np.ndarray]:
    """Dense [n_prompts, n_features] arrays per stat, NaN where a feature was
    not active on that prompt."""
    n_prompts = len(records)
    n_features = len(kept_feature_ids)
    col_of = {fid: j for j, fid in enumerate(kept_feature_ids)}

    matrices = {
        stat: np.full((n_prompts, n_features), np.nan, dtype=np.float32)
        for stat in ("activation_density", "peak_magnitude", "log_odds")
    }

    for row, rec in enumerate(records):
        for fid_str, (density, peak, log_odds) in rec.get("features", {}).items():
            fid = int(fid_str)
            col = col_of.get(fid)
            if col is None:
                continue
            matrices["activation_density"][row, col] = density
            matrices["peak_magnitude"][row, col] = peak
            matrices["log_odds"][row, col] = log_odds

    return matrices


def write_feature_matrix(
    output_dir: str,
    matrices: Dict[str, np.ndarray],
    prompt_ids: List[int],
    feature_ids: List[int],
    meta_extra: Dict[str, Any],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for stat, arr in matrices.items():
        np.save(os.path.join(output_dir, f"{stat}.npy"), arr)
    np.save(os.path.join(output_dir, "prompt_ids.npy"), np.asarray(prompt_ids, dtype=np.int64))
    np.save(os.path.join(output_dir, "feature_ids.npy"), np.asarray(feature_ids, dtype=np.int64))

    meta = {
        "n_prompts": len(prompt_ids),
        "n_features": len(feature_ids),
        "stats": ["activation_density", "peak_magnitude", "log_odds"],
        "fill_value": "NaN (feature not active on that prompt)",
        **meta_extra,
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FAC feature-matrix pipeline: compute (activation_density, peak_magnitude, "
            "log_odds) per SAE feature per seed prompt, filter features, and write the "
            "result as a dense prompt x feature matrix for fast lookups."
        )
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sae-ckpt-path", type=str, default="")

    parser.add_argument("--prompts-json", type=str, default=_DEFAULT_PROMPTS_JSON)
    parser.add_argument("--max-prompts", type=int, default=0, help="0 = all")
    parser.add_argument("--hf-cache-dir", type=str, default=fs._default_cache_dir())

    parser.add_argument("--baseline-tsv", type=str, default=fs._DEFAULT_BASELINE_TSV)
    parser.add_argument("--max-mean-density", type=float, default=0.5)
    parser.add_argument("--min-max-peak", type=float, default=0.3)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for the matrix files. Defaults to "
        "output/feature_matrix_<prompts-json-stem>.",
    )
    return parser.parse_args()


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

    prompts = fs.load_prompts(args.prompts_json)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    output_dir = args.output_dir
    if not output_dir:
        stem = os.path.splitext(os.path.basename(args.prompts_json))[0]
        output_dir = os.path.join(_SCRIPT_DIR, "output", f"feature_matrix_{stem}")

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
    # Step 1: extract per-example feature stats (identical to feature-stats pipeline)
    # -----------------------------------------------------------------------
    records: List[Dict[str, Any]] = []
    with tc.no_grad():
        for idx, prompt in enumerate(tqdm.tqdm(prompts, desc="Processing prompts")):
            record = fs.compute_feature_stats_for_prompt(prompt, model, collector, sae, baseline_full)
            record["index"] = idx
            records.append(record)

    # -----------------------------------------------------------------------
    # Step 2: aggregate and filter features (same rule as feature-stats pipeline)
    # -----------------------------------------------------------------------
    n_examples = len(records)
    agg = fs.aggregate_feature_stats(records)
    kept = fs.filter_features(
        agg, max_mean_density=args.max_mean_density, min_max_peak=args.min_max_peak
    )
    print(
        f"Aggregated {len(agg)} features across {n_examples} examples; "
        f"{len(kept)} passed filters (mean_density <= {args.max_mean_density}, "
        f"max_peak >= {args.min_max_peak})."
    )

    # -----------------------------------------------------------------------
    # Step 3: build the dense prompt x feature matrix and write it
    # -----------------------------------------------------------------------
    feature_ids = sorted(kept)
    prompt_ids = [rec["index"] for rec in records]

    matrices = build_feature_matrix(records, feature_ids)
    write_feature_matrix(
        output_dir=output_dir,
        matrices=matrices,
        prompt_ids=prompt_ids,
        feature_ids=feature_ids,
        meta_extra={
            "prompts_json": os.path.abspath(args.prompts_json),
            "baseline_tsv": os.path.abspath(args.baseline_tsv),
            "sae_ckpt_path": os.path.abspath(sae_ckpt),
            "layer": args.layer,
            "max_mean_density": args.max_mean_density,
            "min_max_peak": args.min_max_peak,
        },
    )
    print(f"Wrote {n_examples} x {len(feature_ids)} feature matrix to {output_dir}")


if __name__ == "__main__":
    main()
