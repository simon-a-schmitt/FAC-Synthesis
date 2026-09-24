"""
Determine which SAE features are "active" on a seed-group TSV, by running
every row through Llama-3.1-8B-Instruct + the SAE hook exactly as the
fac_test_pipeline/ scripts do, and checking whether each feature reaches a
RAW (unnormalised) SAE activation above --threshold on at least one content
token anywhere in the dataset.

Input
-----
A headerless two-column TSV (as produced by data_synthesis/data/seed_groups/):

    <text>\t<label>

The TASK is auto-detected per row from the label column's format:

    "Answer: toxic" / "Answer: safe"                         -> toxicity detection
    "CVSS:3.1/AV:.../AC:.../..."                              -> CTI-VSP (CVSS)
    "LTD: ?|TER: ?|CH: ?|CR: ?|USE: ?|LAW: ?|J: ?|ARB: ?"      -> CLAUDETTE-TOS

For each task the corresponding system-prompt / user-turn / content-token
convention is reproduced byte-for-byte from the benchmark configs in
benchmark_play_ground/ (run_toxicity_benchmark.py, run_cti_vsp_benchmark.py,
run_claudette_benchmark.py), via the fac_test_pipeline/*_prompt.py helper
modules that already do this for the other fac_test_pipeline/ scripts
(cvss_prompt.py, claudette_prompt.py, toxicity_prompt.py). A TSV may freely
mix rows from different tasks; each row is classified and prompted
independently.

Processing
----------
Every row is run through the model + SAE (same encoding / content-token
logic as run_fac_test_pipeline_feature_stats.py / feature_coverage/
run_feature_coverage.py), giving per-feature peak_magnitude: the maximum
RAW, unnormalised SAE activation reached on any content token of that row.
Content tokens are the user-turn tokens after any task-specific opening
phrase (only CTI-VSP has one: "CVE Description:"), excluding special
tokens. No baseline statistics are used anywhere in this script - every
feature the SAE fires on content tokens is considered, not just ones
present in a baseline TSV.

A feature is "active" iff its peak_magnitude, maximised over every row in the
dataset, exceeds --threshold.

Output
------
A JSON file containing the sorted list of active feature ids, written to
--output-json (default: output/<input-tsv-stem>_threshold_<threshold>_active_features.json
next to this script).
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch as tc
import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FAC_SYNTHESIS_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_FAC_TEST_PIPELINE_DIR = os.path.join(_FAC_SYNTHESIS_DIR, "fac_test_pipeline")
_WS_PATH = os.path.dirname(os.path.dirname(_FAC_SYNTHESIS_DIR))

if _FAC_TEST_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _FAC_TEST_PIPELINE_DIR)

import run_fac_test_pipeline_feature_stats as fs  # noqa: E402
import cvss_prompt as cp  # noqa: E402
import claudette_prompt as clp  # noqa: E402
import toxicity_prompt as tp  # noqa: E402

_DEFAULT_MODEL_PATH = os.path.join(_WS_PATH, "models", "llama-3.1-8b")
_DEFAULT_SAE_PATH = os.path.join(_WS_PATH, "models", "sae_llama_l16", "TopK7_l16_h4096_epoch3.pth")
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "output")


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------

TASK_TOXICITY = "toxicity_detection"
TASK_CTI_VSP = "cti_vsp"
TASK_CLAUDETTE = "claudette_tos"


def detect_task(label: str) -> Optional[str]:
    """Infer the task from a label-column value's format.

    "Answer: toxic" / "Answer: safe"                     -> toxicity detection
    "CVSS:3.1/AV:.../..."                                 -> CTI-VSP
    "LTD: ?|TER: ?|CH: ?|CR: ?|USE: ?|LAW: ?|J: ?|ARB: ?"  -> CLAUDETTE-TOS

    Returns None if the label matches none of the known formats.
    """
    stripped = label.strip()
    if stripped.lower().startswith("answer:"):
        return TASK_TOXICITY
    if stripped.startswith("CVSS:"):
        return TASK_CTI_VSP
    if "LTD:" in stripped and "ARB:" in stripped:
        return TASK_CLAUDETTE
    return None


def build_prompt_for_row(task: str, text: str) -> Tuple[str, str, Optional[str]]:
    """Return (system, user_content, opening_phrase) for one (task, text) row,
    reproducing the exact prompt structure of the corresponding
    benchmark_play_ground/run_*_benchmark.py config."""
    if task == TASK_TOXICITY:
        return tp.TOXICITY_SYSTEM_PROMPT, tp.build_toxicity_user_content(text), None
    if task == TASK_CTI_VSP:
        return cp.CVSS_SYSTEM_PROMPT, cp.build_cvss_user_content(text), cp.CVSS_OPENING_PHRASE
    if task == TASK_CLAUDETTE:
        return clp.CLAUDETTE_SYSTEM_PROMPT, clp.build_claudette_user_content(text), None
    raise ValueError(f"Unknown task: {task!r}")


# ---------------------------------------------------------------------------
# Per-row raw-activation extraction (no baseline statistics involved)
# ---------------------------------------------------------------------------

def compute_max_raw_activation_per_feature(
    user_content: str,
    model: "fs.UnifiedGenerator",
    collector: "fs.Collector",
    sae: "fs.TopKSAE",
    opening_phrase: Optional[str],
    system: Optional[str],
) -> Dict[int, float]:
    """{feature_id: max raw SAE activation over this row's content tokens}.

    Same tokenisation / content-token-masking logic as
    run_fac_test_pipeline_feature_stats.compute_feature_stats_for_prompt(),
    stripped down to just the raw peak activation and with no dependency on
    baseline statistics (no filtering to a baseline feature-id set)."""
    collector.cache = None
    _, token_ids, tokens = fs._encode_prompt_tokens(user_content, model, system=system)

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

    special_ids = fs._special_token_id_set(model._tokenizer)  # noqa: SLF001
    content_start = fs._find_user_content_start(token_ids, tokens, model._tokenizer)  # noqa: SLF001
    if opening_phrase:
        content_start = fs._advance_past_opening_phrase(
            token_ids, content_start, model._tokenizer, opening_phrase  # noqa: SLF001
        )
    token_mask = tc.tensor(
        [idx >= content_start and token_id not in special_ids for idx, token_id in enumerate(token_ids)],
    )

    content_features = sparse_features[token_mask]
    if content_features.numel() == 0:
        return {}

    active_feature_indices = tc.unique((content_features > 0).nonzero(as_tuple=False)[:, 1])
    if active_feature_indices.numel() == 0:
        return {}

    peak_magnitudes = content_features[:, active_feature_indices].max(dim=0).values
    return {
        int(fid): float(val)
        for fid, val in zip(active_feature_indices.tolist(), peak_magnitudes.tolist())
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_seed_group_tsv(tsv_path: str) -> List[Tuple[str, str]]:
    """Read a headerless <text>\\t<label> TSV (standard CSV quoting, i.e. a
    field is wrapped in double quotes and doubled internal quotes are
    unescaped if it contains a literal tab/newline/quote)."""
    rows: List[Tuple[str, str]] = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{tsv_path}:{line_no}: expected 2 tab-separated columns (text, label), "
                    f"got {len(row)}"
                )
            text, label = row[0].strip(), row[1].strip()
            if text:
                rows.append((text, label))
    if not rows:
        raise ValueError(f"No rows found in {tsv_path}")
    return rows


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Determine active SAE features (peak activation over --threshold on "
            "at least one content token) for a seed-group TSV, auto-detecting "
            "the task (toxicity / CTI-VSP / CLAUDETTE-TOS) per row from its "
            "label column."
        )
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    parser.add_argument("--model-name", type=str, default=_DEFAULT_MODEL_PATH)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--sae-ckpt-path", type=str, default=_DEFAULT_SAE_PATH)

    parser.add_argument("--input-tsv", type=str, required=True)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all")
    parser.add_argument("--hf-cache-dir", type=str, default=fs._default_cache_dir())

    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="A feature is active iff its peak activation on at least one content "
        "token, anywhere in the dataset, exceeds this value.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Destination JSON. Defaults to "
        "output/<input-tsv-stem>_threshold_<threshold>_active_features.json.",
    )
    return parser.parse_args()


def _format_threshold(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p").replace("-", "neg")


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

    if not os.path.isfile(args.input_tsv):
        raise FileNotFoundError(f"Input TSV not found: {args.input_tsv}")

    output_json = args.output_json
    if not output_json:
        stem = os.path.splitext(os.path.basename(args.input_tsv))[0]
        output_json = os.path.join(
            _DEFAULT_OUTPUT_DIR,
            f"{stem}_threshold_{_format_threshold(args.threshold)}_active_features.json",
        )

    rows = load_seed_group_tsv(args.input_tsv)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    print(f"Loaded {len(rows)} rows from {args.input_tsv}")

    task_counts: Dict[str, int] = {}
    prepared_rows: List[Tuple[str, str, Optional[str]]] = []
    for text, label in rows:
        task = detect_task(label)
        if task is None:
            raise ValueError(
                f"Could not determine task from label {label!r} (text: {text[:80]!r}...). "
                "Expected an 'Answer: toxic'/'Answer: safe' (toxicity), a 'CVSS:3.1/...' "
                "(CTI-VSP), or an 'LTD: ?|...|ARB: ?' (CLAUDETTE-TOS) label."
            )
        task_counts[task] = task_counts.get(task, 0) + 1
        system, user_content, opening_phrase = build_prompt_for_row(task, text)
        prepared_rows.append((system, user_content, opening_phrase))
    print(f"Task counts: {task_counts}")

    sae_ckpt = fs.resolve_sae_checkpoint(local_path=args.sae_ckpt_path or None)

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
    # Step 1: per-row raw peak activations.
    # -----------------------------------------------------------------------
    global_max_peak: Dict[int, float] = {}
    with tc.no_grad():
        for system, user_content, opening_phrase in tqdm.tqdm(prepared_rows, desc="Processing rows"):
            row_peaks = compute_max_raw_activation_per_feature(
                user_content, model, collector, sae,
                opening_phrase=opening_phrase,
                system=system,
            )
            for fid, peak in row_peaks.items():
                if peak > global_max_peak.get(fid, float("-inf")):
                    global_max_peak[fid] = peak

    # -----------------------------------------------------------------------
    # Step 2: threshold and write output.
    # -----------------------------------------------------------------------
    active_feature_ids = sorted(fid for fid, peak in global_max_peak.items() if peak > args.threshold)
    print(
        f"{len(active_feature_ids)} / {len(global_max_peak)} features active "
        f"(peak raw activation > {args.threshold}) across {len(rows)} rows."
    )

    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(active_feature_ids, f, indent=2)
    print(f"Wrote {len(active_feature_ids)} active feature ids to {output_json}")


if __name__ == "__main__":
    main()
