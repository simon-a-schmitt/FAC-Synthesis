"""
Compare SAE feature activations of blackbox-generated CTI/VSP prompts
(cti_vsp_bb_deepseek.json) against the seed-derived feature classification
(output/feature_classification_cti_vsp_seed_group_01.jsonl).

Seed feature sets
------------------
From the classification JSONL:
  - template_features (single entry, already sorted by min_log_odds desc):
    top 10 feature IDs -> the "seed template" set (used for every example).
  - subdomain_features (one entry per seed example, already sorted by
    log_odds_on_example desc): top 10 feature IDs per seed example_index ->
    a *per-seed* "seed subdomain" set, looked up per example via its
    seed_example_id field.

Per-example processing
-----------------------
Every record in cti_vsp_bb_deepseek.json ("seed" and "synthetic") is turned
into a full prompt by prepending the common prompt prefix shared by all
entries in input/prompts_cti_vsp_seed_group_01.json (everything up to and
including "CVE Description: ") to the record's "text". The prompt is run
through Llama-3.1-8B + the SAE hook exactly as in
run_fac_test_pipeline_blackbox_features.py (identical to
run_fac_test_pipeline_feature_stats.py) to obtain, per feature,
(activation_density, peak_magnitude, log_odds). Features are filtered by the
same thresholds (activation_density <= max_density, peak_magnitude >=
min_peak) before ranking by log_odds, exactly as in
run_fac_test_pipeline_blackbox_features.py.

Two-stage comparison:
  1. Take the top 30 (by log_odds) surviving features. Any that are in the
     seed template set are recorded (IDs only) as "template_features_found"
     and removed from consideration.
  2. From the remaining features (still ordered by log_odds), take the top
     20. Any that are in this example's per-seed subdomain set are recorded
     (IDs only) as "subdomain_features_found". Everything else in this top
     20 is a "new_concept" feature: recorded with feature_id, stats,
     top_tokens and descriptions (as in the seed JSONL).

Resumable: pool_id values already present in --output-jsonl are skipped on
start, so the script can be re-run across multiple SLURM jobs / runs to work
through the batch incrementally. Each result is appended and flushed
immediately after processing.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import torch as tc
import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_SCRIPT_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import run_fac_test_pipeline_blackbox_features as bbf  # noqa: E402


SEED_TOP_N_TEMPLATE = 10
SEED_TOP_N_SUBDOMAIN = 10
STAGE1_TOP_N = 30
STAGE2_TOP_N = 20

_DEFAULT_CLASSIFICATION_JSONL = os.path.join(
    _PIPELINE_DIR, "output", "feature_classification_cti_vsp_seed_group_01.jsonl"
)
_DEFAULT_BB_JSON = os.path.join(_SCRIPT_DIR, "cti_vsp_bb_deepseek.json")
_DEFAULT_SEED_PROMPTS_JSON = os.path.join(_PIPELINE_DIR, "input", "prompts_cti_vsp_seed_group_01.json")
_DEFAULT_OUTPUT_JSONL = os.path.join(_SCRIPT_DIR, "output", "cti_vsp_bb_deepseek_feature_comparison.jsonl")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_seed_feature_sets(
    classification_jsonl: str,
    top_n_template: int,
    top_n_subdomain: int,
) -> Tuple[Set[int], Dict[int, Set[int]]]:
    """Return (template_ids, subdomain_ids_by_seed_index)."""
    template_ids: Set[int] = set()
    subdomain_ids_by_seed: Dict[int, Set[int]] = {}

    for rec in load_jsonl(classification_jsonl):
        rec_type = rec.get("type")
        if rec_type == "template_features":
            top_features = rec.get("features", [])[:top_n_template]
            template_ids = {int(f["feature_id"]) for f in top_features}
        elif rec_type == "subdomain_features":
            seed_idx = rec.get("example_index")
            top_features = rec.get("features", [])[:top_n_subdomain]
            subdomain_ids_by_seed[int(seed_idx)] = {int(f["feature_id"]) for f in top_features}

    return template_ids, subdomain_ids_by_seed


def load_bb_examples(bb_json_path: str) -> List[Dict[str, Any]]:
    with open(bb_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {bb_json_path}, got {type(data)}")
    for rec in data:
        if not isinstance(rec.get("text"), str):
            raise ValueError(f"BB record missing 'text' string field: {rec}")
        if "seed_example_id" not in rec:
            raise ValueError(f"BB record missing 'seed_example_id' field: {rec}")
    if not data:
        raise ValueError(f"No records found in {bb_json_path}")
    return data


def derive_prompt_prefix(seed_prompts_json: str) -> str:
    """The prompt prefix shared by every entry in the seed-group prompts file
    (everything up to and including e.g. "CVE Description: "), computed as
    the longest common prefix so the bb example's raw "text" can be appended
    directly to reconstruct the same kind of prompt the seed features were
    derived from."""
    with open(seed_prompts_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [rec["prompt"] for rec in data]
    prefix = os.path.commonprefix(prompts)
    if len(prefix) < 50:
        raise ValueError(
            f"Common prompt prefix derived from {seed_prompts_json} is suspiciously short "
            f"({len(prefix)} chars): {prefix!r}"
        )
    return prefix


def load_processed_pool_ids(output_jsonl: str) -> Set[int]:
    processed: Set[int] = set()
    if not os.path.isfile(output_jsonl):
        return processed
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pool_id = rec.get("pool_id")
            if isinstance(pool_id, int):
                processed.add(pool_id)
    return processed


# ---------------------------------------------------------------------------
# Per-example comparison
# ---------------------------------------------------------------------------

def compare_example_features(
    stats_record: Dict[str, Any],
    template_ids: Set[int],
    subdomain_ids_for_seed: Set[int],
    descriptions_all: Dict[int, List[str]],
    max_density: float,
    min_peak: float,
) -> Dict[str, Any]:
    stage1_candidates = bbf.filter_and_rank_top_features(
        stats_record, max_density=max_density, min_peak=min_peak, top_n=STAGE1_TOP_N
    )

    template_hits = [fid for fid, _, _, _ in stage1_candidates if fid in template_ids]
    remaining = [cand for cand in stage1_candidates if cand[0] not in template_ids]
    stage2_candidates = remaining[:STAGE2_TOP_N]

    subdomain_hits = [fid for fid, _, _, _ in stage2_candidates if fid in subdomain_ids_for_seed]

    top_tokens_map = stats_record.get("top_tokens", {})
    new_concept_features = [
        {
            "feature_id": fid,
            "activation_density": density,
            "peak_magnitude": peak,
            "log_odds": log_odds,
            "top_tokens": top_tokens_map.get(str(fid), []),
            "descriptions": descriptions_all.get(fid, []),
        }
        for fid, density, peak, log_odds in stage2_candidates
        if fid not in subdomain_ids_for_seed
    ]

    return {
        "template_features_found": template_hits,
        "subdomain_features_found": subdomain_hits,
        "new_concept_features": new_concept_features,
    }


def build_output_record(
    example_idx: int,
    bb_rec: Dict[str, Any],
    prompt: str,
    stats_record: Dict[str, Any],
    comparison: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "example_index": example_idx,
        "pool_id": bb_rec.get("pool_id"),
        "type": bb_rec.get("type"),
        "seed_example_id": bb_rec.get("seed_example_id"),
        "batch_id": bb_rec.get("batch_id"),
        "run_id": bb_rec.get("run_id"),
        "text": bb_rec.get("text"),
        "prompt": prompt,
        "n_content_tokens": stats_record.get("n_content_tokens", 0),
        "template_features_found": comparison["template_features_found"],
        "subdomain_features_found": comparison["subdomain_features_found"],
        "new_concept_features": comparison["new_concept_features"],
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each prompt in cti_vsp_bb_deepseek.json, compute per-feature "
            "(activation_density, peak_magnitude, log_odds) via Llama-3.1-8B + SAE, "
            "then compare the top features against the seed template/subdomain "
            "feature sets from feature_classification_cti_vsp_seed_group_01.jsonl. "
            "Resumable across runs."
        )
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    parser.add_argument("--model-name", type=str, default=bbf._DEFAULT_MODEL_PATH)  # noqa: SLF001
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sae-ckpt-path", type=str, default=bbf._DEFAULT_SAE_PATH)  # noqa: SLF001

    parser.add_argument("--bb-json", type=str, default=_DEFAULT_BB_JSON)
    parser.add_argument("--classification-jsonl", type=str, default=_DEFAULT_CLASSIFICATION_JSONL)
    parser.add_argument("--seed-prompts-json", type=str, default=_DEFAULT_SEED_PROMPTS_JSON)
    parser.add_argument("--max-examples", type=int, default=0, help="0 = all")
    parser.add_argument("--hf-cache-dir", type=str, default=bbf._default_cache_dir())  # noqa: SLF001

    parser.add_argument("--baseline-tsv", type=str, default=bbf._DEFAULT_BASELINE_TSV)  # noqa: SLF001
    parser.add_argument(
        "--feature-descriptions-tsv",
        type=str,
        default=bbf._DEFAULT_FEAT_DESC_TSV,  # noqa: SLF001
        help="TSV with columns FeatureID and Words. Used to annotate new_concept features.",
    )

    parser.add_argument("--max-density", type=float, default=0.5, help="Filter: exclude if activation_density > this.")
    parser.add_argument("--min-peak", type=float, default=0.3, help="Filter: exclude if peak_magnitude < this.")
    parser.add_argument("--seed-top-n-template", type=int, default=SEED_TOP_N_TEMPLATE)
    parser.add_argument("--seed-top-n-subdomain", type=int, default=SEED_TOP_N_SUBDOMAIN)
    parser.add_argument("--stage1-top-n", type=int, default=STAGE1_TOP_N)
    parser.add_argument("--stage2-top-n", type=int, default=STAGE2_TOP_N)

    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=_DEFAULT_OUTPUT_JSONL,
        help="Destination JSONL (one line per example). Existing pool_id values are skipped (resume).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    global STAGE1_TOP_N, STAGE2_TOP_N
    STAGE1_TOP_N = args.stage1_top_n
    STAGE2_TOP_N = args.stage2_top_n

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
    if not os.path.isfile(args.classification_jsonl):
        raise FileNotFoundError(f"Classification JSONL not found: {args.classification_jsonl}")
    if not os.path.isfile(args.seed_prompts_json):
        raise FileNotFoundError(f"Seed prompts JSON not found: {args.seed_prompts_json}")

    examples = load_bb_examples(args.bb_json)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    prompt_prefix = derive_prompt_prefix(args.seed_prompts_json)
    print(f"Derived prompt prefix ({len(prompt_prefix)} chars) from {args.seed_prompts_json}")

    template_ids, subdomain_ids_by_seed = load_seed_feature_sets(
        args.classification_jsonl,
        top_n_template=args.seed_top_n_template,
        top_n_subdomain=args.seed_top_n_subdomain,
    )
    print(
        f"Loaded {len(template_ids)} seed template feature IDs and "
        f"subdomain sets for {len(subdomain_ids_by_seed)} seed examples "
        f"(top {args.seed_top_n_template}/{args.seed_top_n_subdomain})."
    )

    out_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    processed = load_processed_pool_ids(args.output_jsonl)
    remaining = [
        (idx, rec) for idx, rec in enumerate(examples) if rec.get("pool_id") not in processed
    ]
    print(
        f"{len(examples)} total examples in {args.bb_json}; "
        f"{len(processed)} already in {args.output_jsonl}; {len(remaining)} remaining."
    )
    if not remaining:
        print("Nothing to do.")
        return

    sae_ckpt = bbf.resolve_sae_checkpoint(local_path=args.sae_ckpt_path or None)
    baseline_full = bbf.load_baseline_full(args.baseline_tsv)
    print(f"Loaded baseline for {len(baseline_full)} features from {args.baseline_tsv}")

    descriptions_all: Dict[int, List[str]] = {}
    if args.feature_descriptions_tsv:
        if not os.path.isfile(args.feature_descriptions_tsv):
            raise FileNotFoundError(
                f"Feature descriptions TSV not found: {args.feature_descriptions_tsv}"
            )
        descriptions_all = bbf.load_feature_descriptions_all(args.feature_descriptions_tsv)
        print(f"Loaded descriptions for {len(descriptions_all)} features from {args.feature_descriptions_tsv}")

    model = bbf.UnifiedGenerator(
        model_path,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.hf_cache_dir,
        local_files_only=True,
        strict_local_paths=True,
    )
    collector = bbf.Collector(args.layer)
    bbf.mount_function(model._model, "llama", args.layer, collector)  # noqa: SLF001
    collector.early_stop = True

    sae = bbf.TopKSAE.from_disk(sae_ckpt, device=args.device)
    sae.topk = bbf.TOP_K
    sae.eval()

    with tc.no_grad(), open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for idx, bb_rec in tqdm.tqdm(remaining, desc="Processing examples"):
            prompt = prompt_prefix + bb_rec["text"]
            stats_record = bbf.compute_feature_stats_for_prompt(prompt, model, collector, sae, baseline_full)

            seed_example_id = bb_rec.get("seed_example_id")
            subdomain_ids_for_seed = subdomain_ids_by_seed.get(int(seed_example_id), set())

            comparison = compare_example_features(
                stats_record,
                template_ids=template_ids,
                subdomain_ids_for_seed=subdomain_ids_for_seed,
                descriptions_all=descriptions_all,
                max_density=args.max_density,
                min_peak=args.min_peak,
            )
            out_rec = build_output_record(idx, bb_rec, prompt, stats_record, comparison)
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"Wrote {len(remaining)} new records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
