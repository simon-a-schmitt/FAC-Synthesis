#!/usr/bin/env python3
"""Constrained-slot CTI-VSP (CVSS v3.1 Base vector prediction) benchmark.

Conceptually the 8-slot, k-way-per-slot counterpart of run_claudette_benchmark.py:
each of the 8 CVSS base metrics (AV, AC, PR, UI, S, C, I, A) is one slot, scored
with benchmark_play_ground.slot_scoring (force the "CVSS:3.1/AV: " ... fragment
through the model, read one full-vocabulary log-softmax at the position fixed a
priori by construction, keep only that slot's valid candidate letters). No free
generation happens.

The stored/reported CVSS vector strings (predicted_vector, gt) stay in the old
compact format used throughout this repo, e.g. "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/
S:U/C:H/I:H/A:N" -- no space after each ":". That format cannot be used for the
constrained-decoding template itself, though: probing the tokenizer shows that
e.g. ":R" (UI:R) does not tokenize as a single token the way ":N" does, so
build_slot_plan() rejects a no-space template outright. The internal scoring
template therefore uses a space before each value ("AV: N", matching CLAUDETTE's
convention); only the human-facing/stored vector strings are rendered without
one, via build_cvss_vector().

Metrics
-------
  * exact_match_accuracy : fraction of examples where all 8 metrics match,
  * slot_accuracy        : fraction of (example, metric) pairs that match,
  * macro_f1             : mean of the 8 per-metric macro-F1 scores (each
                            per-metric macro-F1 itself an unweighted mean of
                            that metric's own per-class F1s),
  * micro_f1             : TP/FP/FN pooled over every (metric, class) pair,
  * mad                  : mean absolute difference between the predicted and
                            gold CVSS v3.1 base score.
Every one of these also gets a trivial-baseline counterpart: a constant
predictor that always outputs, for each slot, that slot's majority ground-truth
class -- computed dynamically from the records in --data-tsv, i.e. the same
evaluation set the real predictions are scored against. Per-slot support /
precision / recall / F1 (and their trivial-baseline counterparts) are reported
under "per_metric".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

from cvss import CVSS3
from cvss.exceptions import CVSSError

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.data_loader import load_cti_vsp_tsv, load_cti_vsp_metric_classes
from benchmark_play_ground.prompt_builder import extract_cve_description_block
from benchmark_play_ground.model_wrapper import LocalModel
from benchmark_play_ground.slot_scoring import (
    SlotPlan,
    build_slot_plan,
    format_example,
    make_fragments,
    score_slots,
)


CVSS_METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

# The full CVSS v3.1 Base metric domain (fixed by the spec, independent of what
# a particular --data-tsv happens to contain), used as the constrained-decoding
# candidate alphabet per slot.
CVSS_METRIC_VALUES = {
    "AV": ["N", "A", "L", "P"],
    "AC": ["L", "H"],
    "PR": ["N", "L", "H"],
    "UI": ["N", "R"],
    "S": ["U", "C"],
    "C": ["N", "L", "H"],
    "I": ["N", "L", "H"],
    "A": ["N", "L", "H"],
}

# Old, no-space format ("CVSS:3.1/AV:N/AC:L/..."), used only for the system
# prompt's format example and for parsing gt / few-shot labels. NOT usable as
# the constrained-decoding template -- see module docstring.
CVSS_DISPLAY_FRAGMENTS = make_fragments(CVSS_METRICS, sep="/", assign=":", prefix="CVSS:3.1/")

# Scoring template actually forced through the model. The space before each
# value is what makes every candidate letter a clean single token; see
# slot_scoring.make_fragments / build_slot_plan.
CVSS_SCORING_FRAGMENTS = make_fragments(CVSS_METRICS, sep="/", assign=": ", prefix="CVSS:3.1/")

CVSS_VECTOR_RE = re.compile(
    r"CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]"
)
CVSS_LOOSE_RE = re.compile(r"CVSS:3\.[01]/[A-Za-z:/]+")


CTI_VSP_SYSTEM_PROMPT = (
    "Analyze the CVE description and determine its CVSS v3.1 Base vector.\n"
    "Valid options for each metric:\n"
    "- Attack Vector (AV): N, A, L, P\n"
    "- Attack Complexity (AC): L, H\n"
    "- Privileges Required (PR): N, L, H\n"
    "- User Interaction (UI): N, R\n"
    "- Scope (S): U, C\n"
    "- Confidentiality (C): N, L, H\n"
    "- Integrity (I): N, L, H\n"
    "- Availability (A): N, L, H\n"
    "Answer with exactly one line in the following format, and nothing else:\n\n"
    # Derived from the same (no-space) fragments used to store/report vectors,
    # so the prompt's example and the reported format can never drift apart.
    f"{format_example(CVSS_DISPLAY_FRAGMENTS)}"
)


def extract_cvss_vector_from_text(text: str) -> str | None:
    match = CVSS_VECTOR_RE.search(text)
    return match.group(0) if match else None


def extract_cvss_metrics_from_text(text: str) -> dict:
    """Parse a "CVSS:3.1/AV:N/AC:L/..." vector string into a per-metric dict.

    Used for the ground-truth vector loaded verbatim from the TSV and for
    few-shot labels -- never for model predictions, which come straight off
    the SlotResult returned by slot_scoring.score_slots (see generate_batch).
    """
    metrics = {m: None for m in CVSS_METRICS}
    vector = extract_cvss_vector_from_text(text)
    if vector is None:
        loose = CVSS_LOOSE_RE.search(text)
        vector = loose.group(0) if loose else None
    if not vector:
        return metrics
    for segment in vector.split("/")[1:]:
        key, _, value = segment.partition(":")
        key = key.strip()
        if key in metrics and value:
            metrics[key] = value.strip()[0]
    return metrics


def metrics_match(predicted_metrics: dict, gt_metrics: dict) -> bool:
    return all(predicted_metrics.get(m) == gt_metrics.get(m) for m in CVSS_METRICS)


def build_cvss_vector(metrics: dict) -> str | None:
    """Reassemble a canonical, no-space 'CVSS:3.1/AV:.../.../A:...' string.

    Returns None if any of the 8 metrics is missing, since a partial vector
    cannot be scored.
    """
    if any(metrics.get(m) is None for m in CVSS_METRICS):
        return None
    return "CVSS:3.1/" + "/".join(f"{m}:{metrics[m]}" for m in CVSS_METRICS)


def cvss3_base_score(metrics: dict) -> float | None:
    vector = build_cvss_vector(metrics)
    if vector is None:
        return None
    try:
        score = CVSS3(vector).base_score
    except CVSSError:
        return None
    return float(score) if score is not None else None


def compute_majority_classes(records: list[dict]) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Per-metric majority ground-truth class over this run's --data-tsv records.

    Used both as the trivial per-slot baseline predictor and to order each
    slot's candidate alphabet for the plan's tie policy (majority class last,
    see slot_scoring.build_slot_plan), so an exact log-prob tie resolves to
    the majority class instead of inflating a minority one.
    """
    counts: dict[str, dict[str, int]] = {m: {} for m in CVSS_METRICS}
    for r in records:
        gt_metrics = extract_cvss_metrics_from_text(r.get("gt", ""))
        for m in CVSS_METRICS:
            v = gt_metrics.get(m)
            if v is not None:
                counts[m][v] = counts[m].get(v, 0) + 1
    majority = {}
    for m in CVSS_METRICS:
        if not counts[m]:
            raise SystemExit(f"No parsable ground-truth values for metric {m}; cannot compute majority baseline")
        majority[m] = max(counts[m].items(), key=lambda kv: kv[1])[0]
    return majority, counts


def _ordered_candidates(metric: str, majority_cls: str) -> list[str]:
    """Full CVSS domain for `metric`, with the majority class moved last."""
    values = CVSS_METRIC_VALUES[metric]
    rest = [v for v in values if v != majority_cls]
    return rest + [majority_cls]


def _micro_f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def evaluate_cti_vsp_predictions(
    results: list[dict], metric_classes: dict, majority_classes: dict
) -> dict:
    total = len(results)
    exact_match = sum(1 for r in results if r["correct"])
    exact_match_trivial = sum(
        1 for r in results
        if all(r["gt_metrics"].get(m) == majority_classes[m] for m in CVSS_METRICS)
    )

    slot_total = total * len(CVSS_METRICS)
    slot_correct = sum(
        1 for r in results for m in CVSS_METRICS
        if r["predicted_metrics"].get(m) == r["gt_metrics"].get(m)
    )
    slot_correct_trivial = sum(
        1 for r in results for m in CVSS_METRICS
        if majority_classes[m] == r["gt_metrics"].get(m)
    )

    per_metric = {}
    micro_tp = micro_fp = micro_fn = 0
    micro_tp_trivial = micro_fp_trivial = micro_fn_trivial = 0

    for metric in CVSS_METRICS:
        classes = metric_classes.get(metric, [])
        maj = majority_classes[metric]
        class_stats = {}
        macro_f1_sum = 0.0
        macro_f1_trivial_sum = 0.0
        n_supported = 0

        for cls in classes:
            tp = sum(
                1 for r in results
                if r["predicted_metrics"].get(metric) == cls and r["gt_metrics"].get(metric) == cls
            )
            fp = sum(
                1 for r in results
                if r["predicted_metrics"].get(metric) == cls and r["gt_metrics"].get(metric) != cls
            )
            fn = sum(
                1 for r in results
                if r["predicted_metrics"].get(metric) != cls and r["gt_metrics"].get(metric) == cls
            )
            support = sum(1 for r in results if r["gt_metrics"].get(metric) == cls)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            # Trivial baseline for this class: a constant predictor that always
            # outputs this metric's majority class `maj`. If cls == maj, it
            # "hits" every gt==cls example and "false-positives" every other
            # example; otherwise it never predicts cls at all.
            if cls == maj:
                t_tp, t_fp, t_fn = support, total - support, 0
            else:
                t_tp, t_fp, t_fn = 0, 0, support
            t_precision = t_tp / (t_tp + t_fp) if (t_tp + t_fp) else 0.0
            t_recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) else 0.0
            t_f1 = 2 * t_precision * t_recall / (t_precision + t_recall) if (t_precision + t_recall) else 0.0

            class_stats[cls] = {
                "support": support,
                "precision": precision, "recall": recall, "f1": f1,
                "precision_trivial": t_precision, "recall_trivial": t_recall, "f1_trivial": t_f1,
            }

            micro_tp += tp
            micro_fp += fp
            micro_fn += fn
            micro_tp_trivial += t_tp
            micro_fp_trivial += t_fp
            micro_fn_trivial += t_fn

            if support > 0:
                macro_f1_sum += f1
                macro_f1_trivial_sum += t_f1
                n_supported += 1

        per_metric[metric] = {
            "macro_f1": macro_f1_sum / n_supported if n_supported else 0.0,
            "macro_f1_trivial": macro_f1_trivial_sum / n_supported if n_supported else 0.0,
            "majority_class": maj,
            "classes": class_stats,
        }

    macro_f1 = sum(per_metric[m]["macro_f1"] for m in CVSS_METRICS) / len(CVSS_METRICS)
    macro_f1_trivial = sum(per_metric[m]["macro_f1_trivial"] for m in CVSS_METRICS) / len(CVSS_METRICS)
    micro_f1 = _micro_f1_from_counts(micro_tp, micro_fp, micro_fn)
    micro_f1_trivial = _micro_f1_from_counts(micro_tp_trivial, micro_fp_trivial, micro_fn_trivial)

    score_diffs = [
        abs(r["predicted_score"] - r["gt_score"])
        for r in results
        if r["predicted_score"] is not None and r["gt_score"] is not None
    ]
    n_scored = len(score_diffs)
    mad = sum(score_diffs) / n_scored if n_scored else None

    majority_vector = build_cvss_vector(majority_classes)
    majority_vector_score = cvss3_base_score(majority_classes)
    if majority_vector_score is not None:
        trivial_score_diffs = [
            abs(majority_vector_score - r["gt_score"]) for r in results if r["gt_score"] is not None
        ]
    else:
        trivial_score_diffs = []
    mad_trivial = sum(trivial_score_diffs) / len(trivial_score_diffs) if trivial_score_diffs else None

    return {
        "total": total,
        "exact_match": exact_match,
        "exact_match_accuracy": exact_match / total if total else 0.0,
        "exact_match_accuracy_trivial": exact_match_trivial / total if total else 0.0,
        "slot_correct": slot_correct,
        "slot_total": slot_total,
        "slot_accuracy": slot_correct / slot_total if slot_total else 0.0,
        "slot_accuracy_trivial": slot_correct_trivial / slot_total if slot_total else 0.0,
        "macro_f1": macro_f1,
        "macro_f1_trivial": macro_f1_trivial,
        "micro_f1": micro_f1,
        "micro_f1_trivial": micro_f1_trivial,
        "mad": mad,
        "mad_trivial": mad_trivial,
        "mad_n_scored": n_scored,
        "mad_n_unscored": total - n_scored,
        "majority_classes": majority_classes,
        "majority_vector": majority_vector,
        "majority_vector_score": majority_vector_score,
        "per_metric": per_metric,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Local model directory for Llama-3.1-8b-Instruct")
    p.add_argument("--data-tsv", required=True, help="CTI-VSP TSV file (cti_vsp_benchmark_test_500.tsv)")
    p.add_argument(
        "--metric-distribution-json",
        default=str(ROOT_DIR / "benchmarks" / "cti_vsp" / "cti_vsp_gt_metric_distribution.json"),
        help="JSON file listing the ground-truth classes per CVSS metric (for macro-F1 label scoping)",
    )
    p.add_argument("--few-shot-tsv", default=None, help="Optional TSV with few-shot examples")
    p.add_argument("--mode", choices=("plain", "icl", "fine_tuned"), default="plain")
    p.add_argument("--icl-k", type=int, default=3, help="Number of few-shot examples to include")
    p.add_argument("--lora-path", default=None, help="Path to LoRA adapter weights (required for --mode fine_tuned); merged onto the base model from --model-path")
    p.add_argument("--device", default="cuda", help="Device to run model on (cuda or cpu)")
    p.add_argument("--dtype", default="bfloat16", help="Dtype for model init (bfloat16 or float16)")
    p.add_argument("--max-input-tokens", type=int, default=None, help="Optional cap for prompt tokens before scoring; omit to keep the full prompt")
    p.add_argument("--max-prompts", type=int, default=0, help="Limit number of prompts (0 = all)")
    p.add_argument("--batch-size", type=int, default=32, help="Number of prompts to score in a single batched forward pass")
    p.add_argument("--output-jsonl", default="cti_vsp_benchmark_results.jsonl", help="Per-example JSONL output")
    p.add_argument("--resume", action="store_true", help="Resume from existing output JSONL if present")
    return p.parse_args()


def build_chat_messages(args, query_prompt: str, few_shots: list[dict]) -> list[dict]:
    if args.mode == "plain":
        user_content = extract_cve_description_block(query_prompt)
        return [
            {"role": "system", "content": CTI_VSP_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    if args.mode == "icl":
        k = min(args.icl_k, len(few_shots))
        examples = few_shots[:k]
        user_content = extract_cve_description_block(query_prompt)
        messages = [{"role": "system", "content": CTI_VSP_SYSTEM_PROMPT}]
        for ex in examples:
            ex_description = extract_cve_description_block(ex.get("prompt", ""))
            ex_vector = ex.get("label", ex.get("gt", ""))
            messages.append({"role": "user", "content": ex_description})
            messages.append({"role": "assistant", "content": ex_vector})
        messages.append({"role": "user", "content": user_content})
        return messages
    # "fine_tuned" queries the model directly, without a system prompt or
    # few-shot examples, mirroring how the LoRA fine-tuning targets were built.
    return [{"role": "user", "content": query_prompt}]


def render_chat_text(tokenizer, chat_messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _plan_lead_texts(tokenizer) -> list[str]:
    """Two representative renderings of the prompt up to the answer, for build_slot_plan.

    A short and a long user turn, per slot_scoring.build_slot_plan's contract: it
    cross-checks that the resolved fragment/candidate token ids are identical
    across both, which is what proves the plan does not depend on what precedes it.
    """
    user_messages = [
        "CVE Description: short description.",
        "CVE Description: a considerably longer CVE description, with punctuation: "
        "commas, colons, and a trailing period that mirrors real CVE prose.",
    ]
    return [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": CTI_VSP_SYSTEM_PROMPT},
             {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True,
        )
        for u in user_messages
    ]


def generate_batch(
    hf_model,
    tokenizer,
    texts: list[str],
    *,
    max_input_tokens: int | None,
    device: str,
    plan: SlotPlan,
) -> list[dict]:
    """Thin CTI-VSP-specific wrapper around slot_scoring.score_slots().

    Tokenizes/left-pads `texts` (already rendered through the chat template) and
    hands the batch to the benchmark-agnostic constrained-slot-scoring core,
    then turns each example's SlotResult into a dict with the {metric: value}
    prediction and its no-space vector string rendering. No free generation
    happens here: the only per-slot model decision is the value token, forced
    by the plan (see score_slots).
    """
    encoded = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts]
    if max_input_tokens:
        encoded = [ids[-max_input_tokens:] for ids in encoded]

    padded = tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt")
    input_ids = padded["input_ids"].to(device)
    attention_mask = padded["attention_mask"].to(device)

    try:
        slot_results = score_slots(hf_model, input_ids, attention_mask, plan, device=device)
    except torch.cuda.OutOfMemoryError:
        if device == "cuda":
            torch.cuda.empty_cache()
        if len(texts) <= 1:
            raise
        mid = len(texts) // 2
        print(f"CUDA OOM at batch size {len(texts)}. Splitting into sub-batches of {mid} and {len(texts) - mid}...")
        first = generate_batch(hf_model, tokenizer, texts[:mid], max_input_tokens=max_input_tokens, device=device, plan=plan)
        second = generate_batch(hf_model, tokenizer, texts[mid:], max_input_tokens=max_input_tokens, device=device, plan=plan)
        return first + second

    out = []
    for result in slot_results:
        predicted_metrics = {m: result.values[m] for m in CVSS_METRICS}
        out.append({
            "predicted_metrics": predicted_metrics,
            "predicted_vector": build_cvss_vector(predicted_metrics),
            "ties": {m: result.ties[m] for m in CVSS_METRICS},
        })
    return out


def main():
    args = parse_args()
    if args.mode == "fine_tuned" and not args.lora_path:
        raise SystemExit("--lora-path is required when --mode fine_tuned")

    records = load_cti_vsp_tsv(args.data_tsv)
    if args.max_prompts > 0:
        records = records[: args.max_prompts]
    if not records:
        raise SystemExit(f"No records loaded from --data-tsv {args.data_tsv!r}")

    metric_classes = load_cti_vsp_metric_classes(args.metric_distribution_json)
    # Majority class per metric, computed dynamically over exactly the records
    # being evaluated (see compute_majority_classes). Used both for the trivial
    # per-slot baseline and to order each slot's candidate alphabet so the
    # plan's tie policy resolves ties towards the majority class.
    majority_classes, majority_counts = compute_majority_classes(records)

    # Prepare few-shot examples if requested
    few_shots = []
    if args.few_shot_tsv and args.mode == "icl":
        few_shots = load_cti_vsp_tsv(args.few_shot_tsv)

    lora_path = args.lora_path if args.mode == "fine_tuned" else None
    model = LocalModel(args.model_path, device=args.device, dtype=args.dtype, lora_path=lora_path)
    model.load()
    tokenizer = model.tokenizer
    # Batched scoring needs direct access to the underlying HF model (LocalModel/UnifiedGenerator
    # only expose a single-prompt generate() call), without touching the shared wrapper used by the
    # other benchmark scripts.
    hf_model = model._gen._model

    cvss_candidates = [_ordered_candidates(m, majority_classes[m]) for m in CVSS_METRICS]
    plan = build_slot_plan(
        tokenizer,
        CVSS_METRICS,
        CVSS_SCORING_FRAGMENTS,
        cvss_candidates,
        leads=_plan_lead_texts(tokenizer),
        tie_policy="last",
    )
    print(plan.describe(tokenizer))
    print("Majority classes (trivial baseline, dynamic from --data-tsv):", majority_classes)

    # Fail fast, once, at startup: the ground-truth vector format the rest of
    # this script assumes must parse into a full 8-metric vector.
    sample_gt = records[0]["gt"]
    sample_gt_metrics = extract_cvss_metrics_from_text(sample_gt)
    missing = [m for m in CVSS_METRICS if sample_gt_metrics.get(m) is None]
    if missing:
        raise SystemExit(
            f"Could not parse a full CVSS vector from the first record's gt "
            f"(missing {missing}): {sample_gt!r}"
        )

    # In --mode icl the few-shot assistant turns are injected verbatim into every
    # prompt, so run the same parse check over exactly the few-shot examples that
    # will be used (few_shots[:icl_k]).
    if args.mode == "icl":
        n_icl = min(args.icl_k, len(few_shots))
        if n_icl == 0:
            print(
                f"Warning: --mode icl but no few-shot examples to inject "
                f"(few_shot_tsv={args.few_shot_tsv!r}, icl_k={args.icl_k}); running 0-shot."
            )
        for i, ex in enumerate(few_shots[:n_icl]):
            ex_vector = ex.get("label", ex.get("gt", ""))
            ex_metrics = extract_cvss_metrics_from_text(ex_vector)
            ex_missing = [m for m in CVSS_METRICS if ex_metrics.get(m) is None]
            if ex_missing:
                raise SystemExit(
                    f"Few-shot example {i} from --few-shot-tsv {args.few_shot_tsv!r} is "
                    f"missing metrics {ex_missing}: {ex_vector!r}"
                )

    out_path = Path(args.output_jsonl)
    results = []

    # If resuming, load existing results and start after them
    start_idx = 0
    if args.resume and out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as rfh:
                for line in rfh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        results.append(obj)
                    except Exception:
                        continue
            start_idx = len(results)
            print(f"Resuming: found {start_idx} existing results, will continue from index {start_idx}.")
        except Exception as e:
            print("Warning: failed to read existing output to resume:", e)

    batch_size = max(1, args.batch_size)

    # Keep batch boundaries identical to an uninterrupted run: if the resume point falls
    # mid-batch (e.g. the previous run was killed while writing a batch), re-align to the
    # start of that batch and recompute it in full, so results are grouped exactly as they
    # would be in one continuous run with this --batch-size.
    aligned_start = (start_idx // batch_size) * batch_size
    if aligned_start < start_idx:
        print(
            f"Resume point index {start_idx} is not aligned to batch_size={batch_size}; "
            f"re-aligning to batch start {aligned_start} and recomputing that batch for "
            f"consistent batch grouping."
        )
        results = results[:aligned_start]
        start_idx = aligned_start

    # Rewrite the output file to hold exactly the kept results, then continue appending
    # from there. This truncates away any partial-batch tail from an interrupted run.
    with out_path.open("w", encoding="utf-8") as wfh:
        for r in results:
            wfh.write(json.dumps(r, ensure_ascii=False) + "\n")
    fh = out_path.open("a", encoding="utf-8")

    idx = start_idx
    while idx < len(records):
        batch_records = records[idx: idx + batch_size]
        batch_query_prompts = [rec["prompt"] for rec in batch_records]
        batch_chat_messages = [build_chat_messages(args, qp, few_shots) for qp in batch_query_prompts]
        batch_texts = [render_chat_text(tokenizer, cm) for cm in batch_chat_messages]

        for final_model_input in batch_texts:
            print("=== Final model input (incl. special tokens) ===")
            print(final_model_input)
            print("=== End final model input ===")

        try:
            batch_out = generate_batch(
                hf_model,
                tokenizer,
                batch_texts,
                max_input_tokens=args.max_input_tokens,
                device=args.device,
                plan=plan,
            )
        except RuntimeError as e:
            msg = str(e)
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg.lower()
            if "CUDA error" not in msg and not is_oom:
                raise

            # Retry once with a tighter context window to avoid transient GPU kernel failures
            # (or, for OOM, after generate_batch() has already halved the batch down to a
            # single prompt and still couldn't fit it).
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            fallback_max_input = args.max_input_tokens if args.max_input_tokens else 2048
            fallback_max_input = min(fallback_max_input, 2048)
            print(
                f"CUDA generation failed for batch starting at index {idx}. Retrying with max_input_tokens={fallback_max_input}..."
            )

            batch_out = generate_batch(
                hf_model,
                tokenizer,
                batch_texts,
                max_input_tokens=fallback_max_input,
                device=args.device,
                plan=plan,
            )

        for offset, (rec, query_prompt, pred) in enumerate(zip(batch_records, batch_query_prompts, batch_out)):
            gt = rec.get("gt", "")
            gt_metrics = extract_cvss_metrics_from_text(gt)
            predicted_metrics = pred["predicted_metrics"]
            result = {
                "index": idx + offset,
                "prompt": query_prompt,
                "predicted_vector": pred["predicted_vector"],
                "predicted_metrics": predicted_metrics,
                "predicted_score": cvss3_base_score(predicted_metrics),
                "gt": gt,
                "gt_metrics": gt_metrics,
                "gt_score": cvss3_base_score(gt_metrics),
                "correct": metrics_match(predicted_metrics, gt_metrics),
                "ties": pred["ties"],
            }
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(result)

        idx += batch_size

    fh.close()

    summary = evaluate_cti_vsp_predictions(results, metric_classes, majority_classes)
    print("Summary:")
    print(f"  total: {summary['total']}")
    print(
        f"  exact_match_accuracy: {summary['exact_match_accuracy']:.4f}  "
        f"(trivial majority baseline: {summary['exact_match_accuracy_trivial']:.4f})"
    )
    print(
        f"  slot_accuracy: {summary['slot_accuracy']:.4f} "
        f"({summary['slot_correct']}/{summary['slot_total']})  "
        f"(trivial majority baseline: {summary['slot_accuracy_trivial']:.4f})"
    )
    print(
        f"  macro_f1: {summary['macro_f1']:.4f}  "
        f"(trivial majority baseline: {summary['macro_f1_trivial']:.4f})"
    )
    print(
        f"  micro_f1: {summary['micro_f1']:.4f}  "
        f"(trivial majority baseline: {summary['micro_f1_trivial']:.4f})"
    )
    if summary["mad"] is not None:
        print(
            f"  mad: {summary['mad']:.4f}  "
            f"(trivial majority baseline: {summary['mad_trivial']:.4f})  "
            f"(n_scored={summary['mad_n_scored']}, n_unscored={summary['mad_n_unscored']})"
        )
    else:
        print(f"  mad: n/a (no scoreable predictions; n_unscored={summary['mad_n_unscored']})")
    print(f"  majority vector (trivial baseline): {summary['majority_vector']}")
    print("  per_metric:")
    for metric in CVSS_METRICS:
        m = summary["per_metric"][metric]
        print(
            f"    {metric} (majority={m['majority_class']}): macro_f1={m['macro_f1']:.4f}  "
            f"(trivial: {m['macro_f1_trivial']:.4f})"
        )
        for cls, stats in m["classes"].items():
            print(
                f"      {cls}: support={stats['support']}  precision={stats['precision']:.4f}  "
                f"recall={stats['recall']:.4f}  f1={stats['f1']:.4f}  |  "
                f"trivial: precision={stats['precision_trivial']:.4f}  "
                f"recall={stats['recall_trivial']:.4f}  f1={stats['f1_trivial']:.4f}"
            )

    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as sfh:
        json.dump(summary, sfh, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
