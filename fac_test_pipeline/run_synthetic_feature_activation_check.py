"""
FAC CVSS feature-check pipeline: for each {feature_id, descriptions} entry in an
input JSON (format: input/cti_vsp_fg_examples.json), wrap every description string
in the CVSS-classification prompt template, run it through Llama-3.1-8B + SAE
(identical hook/encode machinery as run_fac_test_pipeline_feature_stats.py), mask
out the template tokens, and check whether `feature_id` is active anywhere on the
remaining content (= the CVE description) tokens.

For every description this writes whether the feature is active, its strength
(p95-normalised magnitude) and at which token(s) it fired, to a JSONL in output/.

Resumable: (example_index, description_index) pairs already present in
--output-jsonl are skipped on restart; each record is appended and flushed
immediately after processing.
"""

import argparse
import csv
import json
import os
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
import cvss_prompt as cp  # noqa: E402


tc.manual_seed(42)

TOP_K = 20
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BASELINE_TSV = os.path.join(_SCRIPT_DIR, "feature_activation_baseline_agg.tsv")
_DEFAULT_INPUT_JSON = os.path.join(_SCRIPT_DIR, "input", "cti_vsp_fg_examples.json")
_DEFAULT_OUTPUT_JSONL = os.path.join(_SCRIPT_DIR, "output", "cvss_feature_check.jsonl")

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


def load_input_items(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of items in {json_path}")

    items: List[Dict[str, Any]] = []
    for rec in payload:
        if not isinstance(rec, dict):
            continue
        if "feature_id" not in rec or "descriptions" not in rec:
            continue
        if not isinstance(rec["descriptions"], list):
            continue
        items.append(rec)

    if not items:
        raise ValueError(
            f"No usable items found in {json_path}. "
            "Expected a list of dicts with 'feature_id' and 'descriptions' fields."
        )
    return items


def load_processed_keys(output_jsonl: str) -> Set[Tuple[int, int]]:
    """Read (example_index, description_index) pairs already written (resume support)."""
    processed: Set[Tuple[int, int]] = set()
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
            ex_idx = rec.get("example_index")
            d_idx = rec.get("description_index")
            if isinstance(ex_idx, int) and isinstance(d_idx, int):
                processed.add((ex_idx, d_idx))
    return processed


# ---------------------------------------------------------------------------
# Tokenisation helpers (identical to run_fac_test_pipeline_feature_stats.py)
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
    (the phrase itself excluded) within the decoded prompt content — i.e. this
    masks out the fixed template tokens so only the CVE-description tokens
    remain. Falls back to `content_start` if the phrase cannot be located.
    """
    remaining_ids = list(token_ids[content_start:])
    if not remaining_ids:
        return content_start

    content_text = tokenizer.decode(remaining_ids, skip_special_tokens=True)
    match_idx = content_text.find(opening_phrase)
    if match_idx < 0:
        print(
            "Warning: template phrase not found in prompt content; using full content instead."
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
# Per-description extraction
# ---------------------------------------------------------------------------

def compute_feature_activation_for_description(
    description: str,
    feature_id: int,
    model: UnifiedGenerator,
    collector: Collector,
    sae: TopKSAE,
    baseline_full: Dict[int, Dict[str, float]],
    system: str = cp.CVSS_SYSTEM_PROMPT,
    opening_phrase: str = cp.CVSS_OPENING_PHRASE,
) -> Dict[str, Any]:
    """Build the two-turn CVSS prompt (system=CVSS_SYSTEM_PROMPT, user="CVE
    Description: " + description) for `description`, run it through the
    model + SAE, mask out everything up to and including `opening_phrase`
    within the user turn, and report whether `feature_id` is active on the
    remaining (description) tokens, with per-token normalised magnitude."""
    user_content = cp.build_cvss_user_content(description)

    collector.cache = None
    _, token_ids, tokens = _encode_prompt_tokens(user_content, model, system=system)

    try:
        model.get_activates(user_content, system=system)
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
    # The system turn (the actual instruction) is already excluded, because
    # _find_user_content_start only returns positions from the user-turn
    # header onward. This just trims the residual "CVE Description:" label
    # itself from the user turn's content tokens.
    if opening_phrase:
        content_start = _advance_past_opening_phrase(
            token_ids, content_start, model._tokenizer, opening_phrase  # noqa: SLF001
        )
    token_mask = tc.tensor(
        [idx >= content_start and token_id not in special_ids for idx, token_id in enumerate(token_ids)],
        device=sparse_features.device,
    )
    content_token_positions = token_mask.nonzero(as_tuple=False).squeeze(1).tolist()

    content_features = sparse_features[token_mask]
    n_content_tokens = int(content_features.shape[0])

    empty_result = {
        "prompt_n_tokens": len(tokens),
        "content_start_token_index": content_start,
        "n_content_tokens": n_content_tokens,
        "active": False,
        "n_active_tokens": 0,
        "max_normalized_magnitude": 0.0,
        "normalized": feature_id in baseline_full,
        "activations": [],
    }

    if n_content_tokens == 0 or feature_id < 0 or feature_id >= sparse_features.shape[1]:
        return empty_result

    raw_col = content_features[:, feature_id]

    bdata = baseline_full.get(feature_id)
    if bdata is not None:
        p95 = max(bdata["p95"], 1e-9)
        norm_col = raw_col / p95
        normalized = True
    else:
        norm_col = raw_col
        normalized = False

    active_mask = raw_col > 0
    n_active = int(active_mask.sum().item())
    if n_active == 0:
        empty_result["normalized"] = normalized
        return empty_result

    active_local_idx = active_mask.nonzero(as_tuple=False).squeeze(1).tolist()
    activations: List[Dict[str, Any]] = []
    for local_idx in active_local_idx:
        pos = content_token_positions[local_idx]
        activations.append({
            "token_index": pos,
            "token": tokens[pos],
            "raw_magnitude": round(float(raw_col[local_idx].item()), 6),
            "normalized_magnitude": round(float(norm_col[local_idx].item()), 6),
        })
    activations.sort(key=lambda a: a["normalized_magnitude"], reverse=True)

    return {
        "prompt_n_tokens": len(tokens),
        "content_start_token_index": content_start,
        "n_content_tokens": n_content_tokens,
        "active": True,
        "n_active_tokens": n_active,
        "max_normalized_magnitude": activations[0]["normalized_magnitude"],
        "normalized": normalized,
        "activations": activations,
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
            "FAC CVSS feature-check pipeline: wrap each description of an "
            "{feature_id, descriptions} input item in the CVSS prompt template, "
            "run it through Llama-3.1-8B + SAE, mask out the template tokens, and "
            "report whether feature_id fires on the description tokens (and how "
            "strongly / on which tokens)."
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
    parser.add_argument("--hf-cache-dir", type=str, default=_default_cache_dir())

    parser.add_argument("--baseline-tsv", type=str, default=_DEFAULT_BASELINE_TSV)

    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=_DEFAULT_OUTPUT_JSONL,
        help="Destination JSONL (one line per description). Existing "
             "(example_index, description_index) pairs are skipped (resume).",
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

    items = load_input_items(args.input_json)
    if args.max_items > 0:
        items = items[: args.max_items]

    out_dir = os.path.dirname(os.path.abspath(args.output_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Flatten to (example_index, description_index, feature_id, description) work items.
    work: List[Tuple[int, int, int, str]] = []
    for ex_idx, item in enumerate(items):
        feature_id = int(item["feature_id"])
        for d_idx, desc in enumerate(item["descriptions"]):
            if isinstance(desc, str) and desc.strip():
                work.append((ex_idx, d_idx, feature_id, desc))

    processed = load_processed_keys(args.output_jsonl)
    remaining = [w for w in work if (w[0], w[1]) not in processed]
    print(
        f"{len(items)} items ({len(work)} descriptions) in {args.input_json}; "
        f"{len(processed)} already in {args.output_jsonl}; {len(remaining)} remaining."
    )
    if not remaining:
        print("Nothing to do.")
        return

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

    n_active = 0
    with tc.no_grad(), open(args.output_jsonl, "a", encoding="utf-8") as out_f:
        for ex_idx, d_idx, feature_id, desc in tqdm.tqdm(remaining, desc="Processing descriptions"):
            result = compute_feature_activation_for_description(
                desc, feature_id, model, collector, sae, baseline_full
            )
            if result["active"]:
                n_active += 1
            record = {
                "example_index": ex_idx,
                "description_index": d_idx,
                "feature_id": feature_id,
                "description": desc,
                **result,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    print(
        f"Wrote {len(remaining)} new records to {args.output_jsonl} "
        f"({n_active} active, {len(remaining) - n_active} inactive)."
    )


if __name__ == "__main__":
    main()
