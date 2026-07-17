"""
FAC blackbox pipeline: process input/prompts_gen_blackbox_casehold.jsonl through
Llama-3.1-8B, extract SAE feature activations via hook, and for each example write
the top-20 features by log-odds (with top_tokens and text-span descriptions) to a
JSONL in the output directory.

Per-example, per-feature (activation_density, peak_magnitude, log_odds) are computed
exactly as in run_fac_test_pipeline_feature_stats.py. On that single example, features
with activation_density > 0.5 OR peak_magnitude < 0.3 are filtered out; among the
remaining features the top 20 by log_odds are kept.

Resumable: the input JSONL is processed in order (0-based example_index). On start,
already-written example_index values are read back from --output-jsonl and skipped,
so the script can be re-run (e.g. after a SLURM time limit) to continue where it
left off. Each result is appended and flushed immediately after processing.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch as tc
import tqdm


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAE_PRETRAIN_DIR = os.path.join(ROOT_DIR, "sae_pretrain")
if SAE_PRETRAIN_DIR not in sys.path:
    sys.path.insert(0, SAE_PRETRAIN_DIR)

from autoencoder import TopKSAE  # noqa: E402
from generator_uni import UnifiedGenerator  # noqa: E402
from llm_surgery_uni import mount_function, switch_mode  # noqa: E402


tc.manual_seed(42)

TOP_K = 20
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BASELINE_TSV = os.path.join(_SCRIPT_DIR, "feature_activation_baseline_agg.tsv")
_DEFAULT_INPUT_JSONL = os.path.join(_SCRIPT_DIR, "input", "prompts_gen_blackbox_casehold.jsonl")
_DEFAULT_OUTPUT_JSONL = os.path.join(_SCRIPT_DIR, "output", "features_gen_blackbox.jsonl")

# WS_PATH is set by start_llama.sh in the SLURM jobs; fall back to deriving it
# from this file's location (fac_test_pipeline/../../.. == workspace root) so
# the script also works outside a SLURM job.
_DEFAULT_WS_PATH = os.environ.get("WS_PATH") or os.path.dirname(os.path.dirname(ROOT_DIR))
_DEFAULT_MODEL_PATH = os.path.join(_DEFAULT_WS_PATH, "models", "llama-3.1-8b")
_DEFAULT_SAE_PATH = os.path.join(_DEFAULT_WS_PATH, "models", "sae_llama_l16", "TopK7_l16_h4096_epoch3.pth")
_DEFAULT_FEAT_DESC_TSV = os.path.join(
    _DEFAULT_WS_PATH,
    "code", "FAC-Synthesis", "sae_feature_analysis", "interpret_features",
    "xxx", "threshold_1.0", "threshold_1.0.tsv",
)


# ---------------------------------------------------------------------------
# Hook collector
# ---------------------------------------------------------------------------

class Collector:
    def __init__(self, layer: int):
        self.layer = layer
        switch_mode(self, "monitor")
        self.early_stop = False
        self.cache = None

    def monitor(self, x):
        self.cache = x


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _default_cache_dir() -> str:
    return os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))


def load_baseline_full(tsv_path: str) -> Dict[int, Dict[str, float]]:
    baseline: Dict[int, Dict[str, float]] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                fid = int(row["feature_id"])
                baseline[fid] = {
                    "p95": float(row["p95"]),
                    "activation_count": float(row["activation_count"]),
                    "total_tokens": float(row["total_tokens"]),
                }
            except (KeyError, ValueError):
                continue
    return baseline


def _parse_word_spans(words_text: str) -> List[str]:
    parts = re.split(r"(?:^|\n)Span \d+:\s*", words_text)
    return [p.strip() for p in parts if p.strip()]


def load_feature_descriptions_all(tsv_path: str) -> Dict[int, List[str]]:
    """Load text-span descriptions for every feature in the TSV (columns FeatureID, Words)."""
    descriptions: Dict[int, List[str]] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                fid = int(row["FeatureID"])
            except (KeyError, ValueError):
                continue
            words = row.get("Words", "").strip()
            if words:
                descriptions[fid] = _parse_word_spans(words)
    return descriptions


def load_input_examples(jsonl_path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec.get("final_question"), str):
                raise ValueError(f"Input record missing 'final_question' string field: {rec}")
            examples.append(rec)
    if not examples:
        raise ValueError(f"No examples found in {jsonl_path}")
    return examples


def load_processed_indices(output_jsonl: str) -> Set[int]:
    """Read example_index values already written to --output-jsonl (resume support)."""
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
            idx = rec.get("example_index")
            if isinstance(idx, int):
                processed.add(idx)
    return processed


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _encode_prompt_tokens(prompt: str, model: UnifiedGenerator) -> Tuple[tc.Tensor, List[int], List[str]]:
    messages = model.build_messages(user_text=prompt)
    enc = model._tokenizer.apply_chat_template(  # noqa: SLF001
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    if isinstance(enc, dict):
        input_ids = enc["input_ids"]
    else:
        input_ids = enc

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    input_ids = input_ids.to(model._device)
    token_ids = input_ids[0].tolist()
    tokens = model._tokenizer.convert_ids_to_tokens(token_ids)  # noqa: SLF001
    return input_ids, token_ids, tokens


def _special_token_id_set(tokenizer) -> set:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    special_ids.update(range(128000, 128256))
    return special_ids


def _is_newline_only_token(token: str) -> bool:
    stripped = token.replace("Ċ", "").replace("\n", "")
    return len(stripped) == 0 and len(token) > 0


def _find_user_content_start(token_ids: Sequence[int], tokens: Sequence[str], tokenizer) -> int:
    marker_tokens = ("<|start_header_id|>", "user", "<|end_header_id|>")
    marker_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in marker_tokens]

    user_header_end = -1
    if all(isinstance(tok_id, int) and tok_id >= 0 for tok_id in marker_ids):
        marker_len = len(marker_ids)
        for idx in range(0, len(token_ids) - marker_len + 1):
            if list(token_ids[idx : idx + marker_len]) == marker_ids:
                user_header_end = idx + marker_len
                break

    if user_header_end < 0:
        marker_len = len(marker_tokens)
        for idx in range(0, len(tokens) - marker_len + 1):
            if tuple(tokens[idx : idx + marker_len]) == marker_tokens:
                user_header_end = idx + marker_len
                break

    if user_header_end < 0:
        return 0

    content_start = user_header_end
    while content_start < len(tokens) and _is_newline_only_token(tokens[content_start]):
        content_start += 1
    return content_start


# ---------------------------------------------------------------------------
# Log-odds computation (identical to run_fac_test_pipeline_feature_stats.py)
# ---------------------------------------------------------------------------

def _compute_log_odds_se(a_seed: float, N_seed: float, a_base: float, N_base: float) -> float:
    lor = (
        math.log((a_seed + 0.5) / (N_seed - a_seed + 0.5))
        - math.log((a_base + 0.5) / (N_base - a_base + 0.5))
    )
    se = math.sqrt(
        1.0 / (a_seed + 0.5)
        + 1.0 / (N_seed - a_seed + 0.5)
        + 1.0 / (a_base + 0.5)
        + 1.0 / (N_base - a_base + 0.5)
    )
    return lor / se if se > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Per-example extraction (identical to run_fac_test_pipeline_feature_stats.py)
# ---------------------------------------------------------------------------

def compute_feature_stats_for_prompt(
    prompt: str,
    model: UnifiedGenerator,
    collector: Collector,
    sae: TopKSAE,
    baseline_full: Dict[int, Dict[str, float]],
) -> Dict[str, Any]:
    """Return per-feature (activation_density, peak_magnitude, log_odds) for one prompt."""
    collector.cache = None
    _, token_ids, tokens = _encode_prompt_tokens(prompt, model)

    try:
        model.get_activates(prompt)
    except RuntimeError:
        pass

    if collector.cache is None:
        raise RuntimeError("Collector cache is empty. Hook may not be mounted correctly.")

    hidden = collector.cache.to(tc.float32)
    if hidden.dim() != 3:
        raise RuntimeError(f"Expected hidden states [batch, seq, hidden], got {tuple(hidden.shape)}")

    hidden_seq = hidden[0]
    if hidden_seq.shape[0] != len(tokens):
        seq_len = min(hidden_seq.shape[0], len(tokens))
        hidden_seq = hidden_seq[:seq_len]
        token_ids = token_ids[:seq_len]
        tokens = tokens[:seq_len]

    sparse_features = sae.encode(hidden_seq).detach().cpu()

    special_ids = _special_token_id_set(model._tokenizer)  # noqa: SLF001
    content_start = _find_user_content_start(token_ids, tokens, model._tokenizer)  # noqa: SLF001
    token_mask = tc.tensor(
        [idx >= content_start and token_id not in special_ids for idx, token_id in enumerate(token_ids)],
        device=sparse_features.device,
    )

    content_token_positions = token_mask.nonzero(as_tuple=False).squeeze(1).tolist()

    content_features = sparse_features[token_mask]
    if content_features.numel() == 0:
        return {"prompt": prompt, "n_content_tokens": 0, "features": {}, "top_tokens": {}}

    N_seed = int(content_features.shape[0])

    active_feature_indices = tc.unique(
        (content_features > 0).nonzero(as_tuple=False)[:, 1]
    )
    active_feature_indices, _ = tc.sort(active_feature_indices)

    baseline_fids = set(baseline_full.keys())
    keep_mask = tc.tensor(
        [int(fid) in baseline_fids for fid in active_feature_indices.tolist()],
        dtype=tc.bool,
        device=active_feature_indices.device,
    )
    active_feature_indices = active_feature_indices[keep_mask]

    if active_feature_indices.numel() == 0:
        return {"prompt": prompt, "n_content_tokens": N_seed, "features": {}, "top_tokens": {}}

    fid_list = [int(f) for f in active_feature_indices.tolist()]

    raw_counts = (content_features[:, active_feature_indices] > 0).sum(dim=0)

    p95_vals = tc.tensor(
        [baseline_full[fid]["p95"] for fid in fid_list],
        dtype=tc.float32,
        device=content_features.device,
    ).clamp(min=1e-9)
    norm_features = content_features[:, active_feature_indices] / p95_vals.unsqueeze(0)
    peak_magnitudes = norm_features.max(dim=0).values

    _TOP_TOKEN_K = 10
    features: Dict[str, List[float]] = {}
    top_tokens: Dict[str, List] = {}
    for i, fid in enumerate(fid_list):
        activation_density = round(float(raw_counts[i].item()) / N_seed, 6)
        peak_magnitude = round(float(peak_magnitudes[i].item()), 6)
        bdata = baseline_full[fid]
        log_odds = round(
            _compute_log_odds_se(
                a_seed=float(raw_counts[i].item()),
                N_seed=float(N_seed),
                a_base=bdata["activation_count"],
                N_base=bdata["total_tokens"],
            ),
            4,
        )
        features[str(fid)] = [activation_density, peak_magnitude, log_odds]

        feat_col = norm_features[:, i]  # [N_content]
        k = min(_TOP_TOKEN_K, int((feat_col > 0).sum().item()))
        if k > 0:
            top_vals, top_local_idx = tc.topk(feat_col, k=k)
            top_tokens[str(fid)] = [
                [content_token_positions[local_idx], tokens[content_token_positions[local_idx]], round(val, 4)]
                for local_idx, val in zip(top_local_idx.tolist(), top_vals.tolist())
                if val > 0
            ]
        else:
            top_tokens[str(fid)] = []

    return {"prompt": prompt, "n_content_tokens": N_seed, "features": features, "top_tokens": top_tokens}


# ---------------------------------------------------------------------------
# Per-example filtering & ranking
# ---------------------------------------------------------------------------

def filter_and_rank_top_features(
    stats_record: Dict[str, Any],
    max_density: float = 0.5,
    min_peak: float = 0.3,
    top_n: int = 20,
) -> List[Tuple[int, float, float, float]]:
    """Keep features with activation_density <= max_density AND peak_magnitude >= min_peak
    on this example, then return the top_n by log_odds (descending) as
    (feature_id, activation_density, peak_magnitude, log_odds) tuples."""
    candidates = []
    for fid_str, (density, peak, log_odds) in stats_record.get("features", {}).items():
        if density > max_density or peak < min_peak:
            continue
        candidates.append((int(fid_str), density, peak, log_odds))
    candidates.sort(key=lambda x: x[3], reverse=True)
    return candidates[:top_n]


def build_output_record(
    example_idx: int,
    input_rec: Dict[str, Any],
    stats_record: Dict[str, Any],
    top_features: List[Tuple[int, float, float, float]],
    descriptions: Dict[int, List[str]],
) -> Dict[str, Any]:
    top_tokens_map = stats_record.get("top_tokens", {})
    features_out = [
        {
            "feature_id": fid,
            "activation_density": density,
            "peak_magnitude": peak,
            "log_odds": log_odds,
            "top_tokens": top_tokens_map.get(str(fid), []),
            "descriptions": descriptions.get(fid, []),
        }
        for fid, density, peak, log_odds in top_features
    ]
    return {
        "example_index": example_idx,
        "path": input_rec.get("path"),
        "seed_index_1": input_rec.get("seed_index_1"),
        "seed_index_2": input_rec.get("seed_index_2"),
        "transformation_operator": input_rec.get("transformation_operator"),
        "sample_index": input_rec.get("sample_index"),
        "final_question": input_rec.get("final_question"),
        "final_label": input_rec.get("final_label"),
        "accept": input_rec.get("accept"),
        "n_content_tokens": stats_record.get("n_content_tokens", 0),
        "features": features_out,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def resolve_sae_checkpoint(local_path: Optional[str]) -> str:
    if local_path:
        local_path = os.path.abspath(local_path)
        if os.path.exists(local_path):
            return local_path
        raise FileNotFoundError(f"Local SAE checkpoint not found: {local_path}")
    raise ValueError("--sae-ckpt-path is required.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FAC blackbox pipeline: for each example in a JSONL of generated prompts, "
            "compute per-feature (activation_density, peak_magnitude, log_odds) via the "
            "Llama-3.1-8B + SAE hook, filter, and write the top-20 features by log_odds "
            "(with top_tokens and descriptions) to an output JSONL. Resumable across runs."
        )
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    parser.add_argument("--model-name", type=str, default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sae-ckpt-path", type=str, default=_DEFAULT_SAE_PATH)

    parser.add_argument("--input-jsonl", type=str, default=_DEFAULT_INPUT_JSONL)
    parser.add_argument("--max-examples", type=int, default=0, help="0 = all")
    parser.add_argument("--hf-cache-dir", type=str, default=_default_cache_dir())

    parser.add_argument("--baseline-tsv", type=str, default=_DEFAULT_BASELINE_TSV)
    parser.add_argument(
        "--feature-descriptions-tsv",
        type=str,
        default=_DEFAULT_FEAT_DESC_TSV,
        help="TSV with columns FeatureID and Words. Used to annotate output.",
    )

    parser.add_argument("--max-density", type=float, default=0.5, help="Filter: exclude if activation_density > this.")
    parser.add_argument("--min-peak", type=float, default=0.3, help="Filter: exclude if peak_magnitude < this.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top features (by log_odds) to keep per example.")

    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=_DEFAULT_OUTPUT_JSONL,
        help="Destination JSONL (one line per example). Existing example_index values are skipped (resume).",
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

    examples = load_input_examples(args.input_jsonl)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    out_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    processed = load_processed_indices(args.output_jsonl)
    remaining = [(idx, ex) for idx, ex in enumerate(examples) if idx not in processed]
    print(
        f"{len(examples)} total examples in {args.input_jsonl}; "
        f"{len(processed)} already in {args.output_jsonl}; {len(remaining)} remaining."
    )
    if not remaining:
        print("Nothing to do.")
        return

    sae_ckpt = resolve_sae_checkpoint(local_path=args.sae_ckpt_path or None)
    baseline_full = load_baseline_full(args.baseline_tsv)
    print(f"Loaded baseline for {len(baseline_full)} features from {args.baseline_tsv}")

    descriptions_all: Dict[int, List[str]] = {}
    if args.feature_descriptions_tsv:
        if not os.path.isfile(args.feature_descriptions_tsv):
            raise FileNotFoundError(
                f"Feature descriptions TSV not found: {args.feature_descriptions_tsv}"
            )
        descriptions_all = load_feature_descriptions_all(args.feature_descriptions_tsv)
        print(f"Loaded descriptions for {len(descriptions_all)} features from {args.feature_descriptions_tsv}")

    model = UnifiedGenerator(
        model_path,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.hf_cache_dir,
        local_files_only=True,
        strict_local_paths=True,
    )
    collector = Collector(args.layer)
    mount_function(model._model, "llama", args.layer, collector)
    collector.early_stop = True

    sae = TopKSAE.from_disk(sae_ckpt, device=args.device)
    sae.topk = TOP_K
    sae.eval()

    with tc.no_grad(), open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for idx, input_rec in tqdm.tqdm(remaining, desc="Processing examples"):
            prompt = input_rec["final_question"]
            stats_record = compute_feature_stats_for_prompt(prompt, model, collector, sae, baseline_full)
            top_features = filter_and_rank_top_features(
                stats_record, max_density=args.max_density, min_peak=args.min_peak, top_n=args.top_n
            )
            out_rec = build_output_record(idx, input_rec, stats_record, top_features, descriptions_all)
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"Wrote {len(remaining)} new records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
