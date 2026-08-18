"""
Compute per-example, per-feature statistics for every prompt:

    (activation_density, peak_magnitude, log_odds)

- activation_density : fraction of content tokens on which the feature fired
                       (raw activations > 0, before p95 normalisation)
- peak_magnitude     : maximum p95-normalised activation on this example
- log_odds           : smoothed log-odds ratio / Woolf SE

After collecting all examples the script aggregates stats per feature and:

  1. Filters out features with mean_activation_density > 0.5 (too ubiquitous).
  2. Filters out features with max_peak_magnitude < 0.3 (never strongly active).

On the remaining feature subset it classifies:

  template_features
      Features active on EVERY example. Rank by min(log_odds) across examples.
      Keep top 20.

  subdomain_features  (per example)
      Features where max(log_odds) >= 2 * second_max(log_odds) across examples
      (strongly dominant on one example). Per example keep the top 15 by
      max_log_odds among those that were active on that example.

Both categories are written — with feature IDs and descriptions — to a JSONL
in the output directory.

Flags
-----
--save-intermediate   Also write the raw per-example JSONL to --output-jsonl.
"""

import argparse
import collections
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


def load_feature_descriptions(tsv_path: str, feature_ids: Set[int]) -> Dict[int, List[str]]:
    descriptions: Dict[int, List[str]] = {fid: [] for fid in feature_ids}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                fid = int(row["FeatureID"])
            except (KeyError, ValueError):
                continue
            if fid not in feature_ids:
                continue
            words = row.get("Words", "").strip()
            if words:
                descriptions[fid] = _parse_word_spans(words)
    return descriptions


def load_prompts(prompts_json: str) -> List[str]:
    with open(prompts_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    prompts: List[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict):
                prompt = item.get("prompt")
                if isinstance(prompt, str):
                    prompts.append(prompt)
    elif isinstance(payload, dict):
        items = payload.get("prompts", [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict):
                    prompt = item.get("prompt")
                    if isinstance(prompt, str):
                        prompts.append(prompt)

    if len(prompts) == 0:
        raise ValueError("No prompts found in prompts JSON.")
    return prompts


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _encode_prompt_tokens(
    prompt: str, model: UnifiedGenerator, system: Optional[str] = None
) -> Tuple[tc.Tensor, List[int], List[str]]:
    messages = model.build_messages(user_text=prompt, system_text=system)
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


def _advance_past_opening_phrase(
    token_ids: Sequence[int],
    content_start: int,
    tokenizer,
    opening_phrase: str,
) -> int:
    """Return a token index >= content_start that skips past `opening_phrase`
    (the phrase itself excluded) within the decoded prompt content.

    Falls back to `content_start` (i.e. the whole content is used) if the
    phrase cannot be located in the decoded text.
    """
    remaining_ids = list(token_ids[content_start:])
    if not remaining_ids:
        return content_start

    content_text = tokenizer.decode(remaining_ids, skip_special_tokens=True)
    match_idx = content_text.find(opening_phrase)
    if match_idx < 0:
        print(
            f"Warning: opening phrase {opening_phrase!r} not found in prompt content; "
            "using full content instead."
        )
        return content_start
    phrase_end_offset = match_idx + len(opening_phrase)

    lo, hi = 0, len(remaining_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        decoded_len = len(tokenizer.decode(remaining_ids[:mid], skip_special_tokens=True))
        if decoded_len >= phrase_end_offset:
            hi = mid
        else:
            lo = mid + 1
    return content_start + lo


# ---------------------------------------------------------------------------
# Log-odds computation (identical to run_fac_test_pipeline_log_odds.py)
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
# Per-example extraction
# ---------------------------------------------------------------------------

def compute_feature_stats_for_prompt(
    prompt: str,
    model: UnifiedGenerator,
    collector: Collector,
    sae: TopKSAE,
    baseline_full: Dict[int, Dict[str, float]],
    opening_phrase: Optional[str] = None,
    system: Optional[str] = None,
    raw_magnitude: bool = False,
) -> Dict[str, Any]:
    """Return per-feature (activation_density, peak_magnitude, log_odds) for one prompt.

    `system`, if given, is prepended as a system turn ahead of `prompt` (the
    user turn) before tokenization/encoding. Default None reproduces the
    previous single-flat-user-turn behaviour exactly (non-CVSS callers of
    this function stay unaffected).

    `raw_magnitude`, if True, skips the p95-baseline normalisation and
    reports the raw SAE activation magnitudes instead (peak_magnitude and
    the per-token top-token values are then raw activations, not
    p95-normalised ones).
    """
    collector.cache = None
    _, token_ids, tokens = _encode_prompt_tokens(prompt, model, system=system)

    try:
        model.get_activates(prompt, system=system)
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
    if opening_phrase:
        content_start = _advance_past_opening_phrase(
            token_ids, content_start, model._tokenizer, opening_phrase  # noqa: SLF001
        )
    token_mask = tc.tensor(
        [idx >= content_start and token_id not in special_ids for idx, token_id in enumerate(token_ids)],
        device=sparse_features.device,
    )

    # Original positions of content tokens (for token index reporting).
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

    if raw_magnitude:
        norm_features = content_features[:, active_feature_indices]
    else:
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

        # Top tokens by normalised magnitude for this feature on this example.
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
# Aggregation & filtering
# ---------------------------------------------------------------------------

def aggregate_feature_stats(
    records: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Collect all per-example stats into per-feature lists."""
    # {fid: {"densities": [...], "peaks": [...], "log_odds": [...], "example_indices": [...]}}
    agg: Dict[int, Dict[str, list]] = collections.defaultdict(
        lambda: {"densities": [], "peaks": [], "log_odds": [], "example_indices": []}
    )
    for rec in records:
        idx = rec["index"]
        for fid_str, (density, peak, lo) in rec.get("features", {}).items():
            fid = int(fid_str)
            agg[fid]["densities"].append(density)
            agg[fid]["peaks"].append(peak)
            agg[fid]["log_odds"].append(lo)
            agg[fid]["example_indices"].append(idx)
    return dict(agg)


def filter_features(
    agg: Dict[int, Dict[str, Any]],
    max_mean_density: float = 0.5,
    min_max_peak: float = 0.3,
) -> Set[int]:
    """Return feature IDs that pass both filters."""
    kept: Set[int] = set()
    for fid, stats in agg.items():
        mean_density = sum(stats["densities"]) / len(stats["densities"])
        max_peak = max(stats["peaks"])
        if mean_density > max_mean_density:
            continue
        if max_peak < min_max_peak:
            continue
        kept.add(fid)
    return kept


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_template_features(
    agg: Dict[int, Dict[str, Any]],
    kept: Set[int],
    n_examples: int,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Top-N features active on every example, ranked by min log_odds."""
    candidates = []
    for fid in kept:
        stats = agg[fid]
        if len(stats["example_indices"]) < n_examples:
            continue
        # Verify presence on all example indices (0 .. n_examples-1).
        if set(stats["example_indices"]) != set(range(n_examples)):
            continue
        min_lo = min(stats["log_odds"])
        mean_density = round(sum(stats["densities"]) / len(stats["densities"]), 6)
        max_peak = round(max(stats["peaks"]), 6)
        candidates.append({
            "feature_id": fid,
            "min_log_odds": round(min_lo, 4),
            "mean_activation_density": mean_density,
            "max_peak_magnitude": max_peak,
            "log_odds_per_example": stats["log_odds"],
        })

    candidates.sort(key=lambda x: x["min_log_odds"], reverse=True)
    return candidates[:top_n]


def classify_subdomain_features(
    agg: Dict[int, Dict[str, Any]],
    kept: Set[int],
    records: List[Dict[str, Any]],
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Per-example: features that reach their global max log_odds on this example,
    and that max is >= 2 * second-highest log_odds (any other example).
    Keeps top-N by log_odds_on_example."""
    # Build global stats per feature: {fid: (max_lo, second_max_lo)}
    dominance: Dict[int, Tuple[float, float]] = {}
    for fid in kept:
        lo_vals = sorted(agg[fid]["log_odds"], reverse=True)
        max_lo = lo_vals[0]
        second_max_lo = lo_vals[1] if len(lo_vals) >= 2 else 0.0
        dominance[fid] = (max_lo, second_max_lo)

    n_examples = len(records)
    result = []
    for rec in records:
        idx = rec["index"]
        prompt = rec["prompt"]

        candidates = []
        for fid_str, (density, peak, lo) in rec.get("features", {}).items():
            fid = int(fid_str)
            if fid not in kept:
                continue
            max_lo, second_max_lo = dominance[fid]
            # Feature must achieve its global maximum ON this example.
            if abs(lo - max_lo) > 1e-6:
                continue
            # That maximum must be at least 2x the second-highest value.
            if max_lo < 2.0 * second_max_lo:
                continue
            # Build full log_odds array across all examples (None where not active).
            lo_per_example: List[Optional[float]] = [None] * n_examples
            for ex_idx, lo_val in zip(agg[fid]["example_indices"], agg[fid]["log_odds"]):
                lo_per_example[ex_idx] = round(lo_val, 4)
            candidates.append({
                "feature_id": fid,
                "max_log_odds": round(max_lo, 4),
                "log_odds_on_example": round(lo, 4),
                "log_odds_per_example": lo_per_example,
                "activation_density_on_example": round(density, 6),
                "peak_magnitude_on_example": round(peak, 6),
                "top_tokens": rec.get("top_tokens", {}).get(fid_str, []),
            })

        # Sort by log_odds on this example (equals max_log_odds for these features).
        candidates.sort(key=lambda x: x["log_odds_on_example"], reverse=True)
        result.append({
            "example_index": idx,
            "prompt": prompt,
            "features": candidates[:top_n],
        })

    return result


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


def write_jsonl(records: List[Dict[str, Any]], output_jsonl: str) -> None:
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
            "FAC feature-stats pipeline: compute (activation_density, peak_magnitude, "
            "log_odds) per SAE feature per prompt, then classify into template and "
            "subdomain features."
        )
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sae-ckpt-path", type=str, default="")

    parser.add_argument("--prompts-json", type=str, required=True)
    parser.add_argument("--max-prompts", type=int, default=0, help="0 = all")
    parser.add_argument("--hf-cache-dir", type=str, default=_default_cache_dir())

    parser.add_argument(
        "--baseline-tsv",
        type=str,
        default=_DEFAULT_BASELINE_TSV,
    )
    parser.add_argument(
        "--feature-descriptions-tsv",
        type=str,
        default="",
        help="TSV with columns FeatureID and Words. Used to annotate output.",
    )
    parser.add_argument(
        "--opening-phrase",
        type=str,
        default="",
        help=(
            "If set, only tokens after this phrase (excluding the phrase itself) "
            "within each prompt's content are considered for feature-activation "
            "stats. Useful to skip a fixed instruction that precedes the actual "
            "prompt content. The prompt is still processed as a whole; only the "
            "considered token range changes. If the phrase is not found in a "
            "prompt, the full content is used for that prompt."
        ),
    )

    # Intermediate per-example output (optional).
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Write per-example feature stats to --output-jsonl.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "output", "feature_stats_intermediate.jsonl"),
        help="Destination for intermediate per-example JSONL (only written with --save-intermediate).",
    )

    # Classification output.
    parser.add_argument(
        "--output-classification-jsonl",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "output", "feature_classification.jsonl"),
        help="Destination for template/subdomain feature classification.",
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

    prompts = load_prompts(args.prompts_json)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    sae_ckpt = resolve_sae_checkpoint(local_path=args.sae_ckpt_path or None)
    baseline_full = load_baseline_full(args.baseline_tsv)
    print(f"Loaded baseline for {len(baseline_full)} features from {args.baseline_tsv}")

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

    # -----------------------------------------------------------------------
    # Step 1: extract per-example feature stats
    # -----------------------------------------------------------------------
    records: List[Dict[str, Any]] = []
    with tc.no_grad():
        for idx, prompt in enumerate(tqdm.tqdm(prompts, desc="Processing prompts")):
            record = compute_feature_stats_for_prompt(
                prompt, model, collector, sae, baseline_full,
                opening_phrase=args.opening_phrase or None,
            )
            record["index"] = idx
            records.append(record)

    if args.save_intermediate:
        write_jsonl(records, args.output_jsonl)
        print(f"Wrote {len(records)} intermediate records to {args.output_jsonl}")

    # -----------------------------------------------------------------------
    # Step 2: aggregate and filter
    # -----------------------------------------------------------------------
    n_examples = len(records)
    agg = aggregate_feature_stats(records)
    kept = filter_features(agg, max_mean_density=0.5, min_max_peak=0.3)
    print(
        f"Aggregated {len(agg)} features across {n_examples} examples; "
        f"{len(kept)} passed filters (mean_density ≤ 0.5, max_peak ≥ 0.3)."
    )

    # -----------------------------------------------------------------------
    # Step 3: classify
    # -----------------------------------------------------------------------
    template_features = classify_template_features(agg, kept, n_examples, top_n=20)
    subdomain_results = classify_subdomain_features(agg, kept, records, top_n=15)

    print(f"template_features: {len(template_features)} features")
    print(
        f"subdomain_features: "
        + ", ".join(
            f"example {r['example_index']}: {len(r['features'])} features"
            for r in subdomain_results
        )
    )

    # -----------------------------------------------------------------------
    # Step 4: load descriptions for all relevant feature IDs
    # -----------------------------------------------------------------------
    relevant_fids: Set[int] = {f["feature_id"] for f in template_features}
    for r in subdomain_results:
        for f in r["features"]:
            relevant_fids.add(f["feature_id"])

    descriptions: Dict[int, List[str]] = {fid: [] for fid in relevant_fids}
    if args.feature_descriptions_tsv:
        if not os.path.isfile(args.feature_descriptions_tsv):
            raise FileNotFoundError(
                f"Feature descriptions TSV not found: {args.feature_descriptions_tsv}"
            )
        descriptions = load_feature_descriptions(args.feature_descriptions_tsv, relevant_fids)
        print(f"Loaded descriptions for {sum(1 for v in descriptions.values() if v)} features.")

    # -----------------------------------------------------------------------
    # Step 5: build and write classification JSONL
    # -----------------------------------------------------------------------
    output_records: List[Dict[str, Any]] = []

    # template_features record
    for f in template_features:
        fid = f["feature_id"]
        f["descriptions"] = descriptions.get(fid, [])
    output_records.append({
        "type": "template_features",
        "n_examples": n_examples,
        "features": template_features,
    })

    # subdomain_features records (one per example)
    for r in subdomain_results:
        for f in r["features"]:
            fid = f["feature_id"]
            f["descriptions"] = descriptions.get(fid, [])
        output_records.append({
            "type": "subdomain_features",
            "example_index": r["example_index"],
            "prompt": r["prompt"],
            "features": r["features"],
        })

    write_jsonl(output_records, args.output_classification_jsonl)
    print(f"Wrote classification to {args.output_classification_jsonl}")


if __name__ == "__main__":
    main()
