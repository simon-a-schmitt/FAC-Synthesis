#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
import json
import math
import re
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.data_loader import load_claudette_tsv
from benchmark_play_ground.model_wrapper import LocalModel
from benchmark_play_ground.slot_scoring import (
    SlotPlan,
    SlotPlanError,
    assert_target_matches_plan,
    build_slot_plan,
    format_example,
    make_fragments,
    score_slots,
)


CLAUDETTE_METRICS = ["LTD", "TER", "CH", "CR", "USE", "LAW", "J", "ARB"]
CLAUDETTE_NEG_CLASS = "N"
CLAUDETTE_ALL_CLASSES = CLAUDETTE_METRICS + [CLAUDETTE_NEG_CLASS]

# Single source of truth for the answer template (see slot_scoring.make_fragments).
# The trailing space in the default `assign` is what makes each Y/N value a clean
# single token for this tokenizer -- build_slot_plan() verifies this and refuses to
# build a plan otherwise. It produces e.g. "LTD: N|TER: N|...|ARB: Y", the same
# spaced format the ground-truth TSV rows are expected to use (they are parsed as
# plain text below and never tokenized).
CLAUDETTE_FRAGMENTS = make_fragments(CLAUDETTE_METRICS)
CLAUDETTE_CANDIDATES = [["Y", "N"]] * len(CLAUDETTE_METRICS)

# Ground-truth vectors (loaded verbatim from the TSV) use the spaced format that
# matches the answer template above, e.g.
# "LTD: N|TER: N|CH: N|CR: N|USE: N|LAW: N|J: N|ARB: N" (no trailing "|"). The
# regex tolerates optional whitespace after each colon and an optional trailing
# "|", so the older compact "LTD:N|...|ARB:N|" form still parses too; it is never
# applied to model output anymore (see generate_batch).
CLAUDETTE_VECTOR_RE = re.compile(
    r"LTD:\s*[YN]\|TER:\s*[YN]\|CH:\s*[YN]\|CR:\s*[YN]\|"
    r"USE:\s*[YN]\|LAW:\s*[YN]\|J:\s*[YN]\|ARB:\s*[YN]\|?"
)


CLAUDETTE_SYSTEM_PROMPT = (
    "You are analyzing a single sentence from the Terms of Service of an\n"
    "online platform under EU consumer law (Directive 93/13/EEC).\n"
    "\n"
    "Decide, for each of the following eight clause types, whether the\n"
    "sentence contains a potentially unfair clause of that type:\n"
    "\n"
    "  LTD  limitation of liability\n"
    "  TER  unilateral termination\n"
    "  CH   unilateral change\n"
    "  CR   content removal\n"
    "  USE  contract by using\n"
    "  LAW  choice of law\n"
    "  J    jurisdiction\n"
    "  ARB  arbitration\n"
    "\n"
    "Most sentences contain no unfair clause of any type.\n"
    "\n"
    "Answer with exactly one line in the following format, using Y or N for\n"
    "each type, and nothing else:\n"
    "\n"
    # Derived from the same fragments the slot plan is built from, so the prompt
    # and the scoring template can never drift apart (see slot_scoring.format_example).
    f"{format_example(CLAUDETTE_FRAGMENTS)}"
)


def extract_claudette_vector_from_text(text: str) -> str | None:
    match = CLAUDETTE_VECTOR_RE.search(text)
    return match.group(0) if match else None


def extract_claudette_metrics_from_text(text: str) -> dict:
    """Parse an "LTD: Y|TER: N|..." vector string into a per-metric Y/N dict.

    Used only for the ground-truth vector loaded verbatim from the TSV (and for the
    startup sanity check below). Predictions no longer go through this parser: each
    slot's chosen candidate comes directly off the SlotResult returned by
    slot_scoring.score_slots (see generate_batch), so there is nothing to parse.
    """
    metrics = {m: None for m in CLAUDETTE_METRICS}
    vector = extract_claudette_vector_from_text(text)
    if not vector:
        return metrics
    for segment in vector.split("|"):
        key, _, value = segment.partition(":")
        key = key.strip()
        if key in metrics and value:
            metrics[key] = value.strip()[0]
    return metrics


def metrics_match(predicted_metrics: dict, gt_metrics: dict) -> bool:
    return all(predicted_metrics.get(m) == gt_metrics.get(m) for m in CLAUDETTE_METRICS)


def _plan_lead_texts(tokenizer) -> list[str]:
    """Two representative renderings of the prompt up to the answer, for build_slot_plan.

    A short and a long user turn, per slot_scoring.build_slot_plan's contract: it
    cross-checks that the resolved fragment/candidate token ids are identical across
    both, which is what proves the plan does not depend on what precedes it.
    """
    user_messages = [
        "short clause.",
        "a considerably longer terms-of-service sentence, with punctuation: "
        "commas, colons, and a trailing period that mirrors real ToS prose.",
    ]
    return [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": CLAUDETTE_SYSTEM_PROMPT},
             {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True,
        )
        for u in user_messages
    ]


def _slot_instance_labels(results: list[dict]) -> tuple[list[str], list[str]]:
    """Flatten each result's 8-slot Y/N vector into per-slot (true, pred) class labels.

    Each (sentence, slot) pair becomes one instance whose class is the slot name
    (e.g. "LTD") if that slot is "Y", or the pooled negative class "N" otherwise.
    This reframes the 8-slot multi-label vector as a single 8+1-way multi-class
    classification problem, with every "no" across every slot pooled into one
    negative class rather than 8 separate per-slot negatives.
    """
    true_labels = []
    pred_labels = []
    for r in results:
        for slot in CLAUDETTE_METRICS:
            true_labels.append(slot if r["gt_metrics"].get(slot) == "Y" else CLAUDETTE_NEG_CLASS)
            pred_labels.append(slot if r["predicted_metrics"].get(slot) == "Y" else CLAUDETTE_NEG_CLASS)
    return true_labels, pred_labels


def _class_stats(true_labels: list[str], pred_labels: list[str], cls: str) -> dict:
    tp = fp = fn = support = 0
    for t, p in zip(true_labels, pred_labels):
        if t == cls:
            support += 1
            if p == cls:
                tp += 1
            else:
                fn += 1
        elif p == cls:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "support": support, "precision": precision, "recall": recall, "f1": f1}


def _micro_f1(stats_by_class: dict, classes: list[str]) -> float:
    tp = sum(stats_by_class[c]["tp"] for c in classes)
    fp = sum(stats_by_class[c]["fp"] for c in classes)
    fn = sum(stats_by_class[c]["fn"] for c in classes)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _macro_f1(stats_by_class: dict, classes: list[str]) -> float:
    return sum(stats_by_class[c]["f1"] for c in classes) / len(classes)


def _slot_label_matrices(results: list[dict]) -> tuple[list[list[int]], list[list[int]]]:
    """Build the (n_examples x 9) multi-label ground-truth and prediction matrices used
    for the LexGLUE-consistent micro/macro F1 (CLAUDETTE-TOS is the UNFAIR-ToS task in
    the LexGLUE benchmark, which scores this kind of multi-label task by running
    sklearn's multilabel-indicator F1 directly over a per-example label matrix).

    One row per example. Columns 0..7 are the 8 clause types (1 if that clause type
    applies to the example, 0 otherwise); column 8 is the negative class "no clause
    type applies", set to 1 iff all of columns 0..7 are 0. Every row therefore has at
    least one 1, matching LexGLUE's convention of an explicit 9th "no label" label
    rather than the absence of the other 8.

    This differs from _slot_instance_labels, which flattens the 8 slots into 8*n
    single-label instances and pools every "no" answer -- across every slot and every
    example -- into one shared negative class, so that class's support scales with
    8*n and swamps the 8 positive classes; here the negative class has exactly one
    instance per example, like the other 8.
    """
    y_true = []
    y_pred = []
    for r in results:
        true_row = [1 if r["gt_metrics"].get(m) == "Y" else 0 for m in CLAUDETTE_METRICS]
        true_row.append(1 if sum(true_row) == 0 else 0)
        y_true.append(true_row)

        pred_row = [1 if r["predicted_metrics"].get(m) == "Y" else 0 for m in CLAUDETTE_METRICS]
        pred_row.append(1 if sum(pred_row) == 0 else 0)
        y_pred.append(pred_row)
    return y_true, y_pred


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _slot_probability_matrix(results: list[dict]) -> tuple[list[list[int]], list[list[float]]]:
    """Build the (n_results x 8) ground-truth and score matrices used for AUPRC.

    Row i, column k is slot CLAUDETTE_METRICS[k] of result i: y_true is 1 if that slot
    is "Y" in the ground truth. y_score is p(Y|slot k) = sigmoid(s_k(x)), i.e. the
    log-odds score s_k(x) = log p(Y|slot k) - log p(N|slot k) mapped back into an
    actual Y-vs-N probability. Every result contributes a row: slot_log_odds always
    has all 8 entries, since each slot's Y/N decision is forced by construction (see
    slot_scoring.score_slots) and can no longer be missing.
    """
    y_true = []
    y_score = []
    for r in results:
        y_true.append([1 if r["gt_metrics"].get(m) == "Y" else 0 for m in CLAUDETTE_METRICS])
        slot_log_odds = r["slot_log_odds"]
        y_score.append([_sigmoid(slot_log_odds[m]) for m in CLAUDETTE_METRICS])
    return y_true, y_score


def _slot_positive_rate(results: list[dict]) -> dict[str, float]:
    """Fraction of examples for which the model predicted "Y" at each slot.

    A calibration/bias signal independent of ground truth: CLAUDETTE-ToS is heavily
    skewed toward "N" (see CLAUDETTE_SYSTEM_PROMPT), so a slot's positive rate should
    normally sit well below 0.5, and a slot stuck near 0 or drifting far above its
    ground-truth prevalence flags a miscalibrated or degenerate model.
    """
    n = len(results)
    if not n:
        return {m: 0.0 for m in CLAUDETTE_METRICS}
    return {
        m: sum(1 for r in results if r["predicted_metrics"].get(m) == "Y") / n
        for m in CLAUDETTE_METRICS
    }


def _slot_tie_rate(results: list[dict]) -> dict[str, float]:
    """Fraction of examples where a slot's Y/N decision was an exact log-prob tie.

    Ties are resolved deterministically by the plan's tie_policy (see
    slot_scoring.build_slot_plan), but a high tie rate at a slot means many of its
    decisions are effectively coin flips rather than confident predictions.
    """
    n = len(results)
    if not n:
        return {m: 0.0 for m in CLAUDETTE_METRICS}
    return {
        m: sum(1 for r in results if r.get("ties", {}).get(m)) / n
        for m in CLAUDETTE_METRICS
    }


def _tie_rate_overall(results: list[dict]) -> float:
    n = len(results) * len(CLAUDETTE_METRICS)
    if not n:
        return 0.0
    total_ties = sum(1 for r in results for m in CLAUDETTE_METRICS if r.get("ties", {}).get(m))
    return total_ties / n


def evaluate_claudette_predictions(results: list[dict]) -> dict:
    """Score predictions under two multi-class framings of the 8 clause-type slots.

    8-class scenario: the 8 clause types (LTD, TER, ... ARB) are the classes;
    micro/macro F1 are computed over those 8 classes only.
    8+1-class scenario: adds a negative class "N" as a 9th class, computed two ways:
      - "micro_f1_8plus1" / "macro_f1_8plus1": the LexGLUE-consistent implementation
        (see _slot_label_matrices) -- one row per example, "N" is a per-example label
        that is 1 iff none of the 8 clause types apply to that example, and micro/macro
        F1 are the standard sklearn multilabel-indicator F1 over the resulting
        (n_examples x 9) matrices.
      - "micro_f1_8plus1_slot_pair" / "macro_f1_8plus1_slot_pair": the original
        implementation (see _slot_instance_labels) -- the 8 slots are flattened into
        8*n single-label (sentence, slot) instances, and "N" pools every "no" answer
        across every slot and every example into one shared negative class, so its
        support scales with 8*n rather than n.

    For every metric, also reports how a trivial classifier that always
    predicts "N" for every slot would score, as a baseline for comparison.

    Additionally computes micro/macro AUPRC over the 8 clause-type slots, treated as
    an 8-label multi-label problem: for each slot k, the score is p(Y|slot k) =
    sigmoid(s_k(x)) with s_k(x) = log p(Y|slot k) - log p(N|slot k), read directly off
    the model's output distribution at the (a priori known) position where it decided
    that slot (see slot_scoring.score_slots). Micro-AUPRC pools all 8*n slot instances
    into a single precision-recall curve; macro-AUPRC averages the per-slot AUPRC
    across the 8 slots. predicted_metrics -- the Y/N vector used for the F1 metrics --
    is taken straight from SlotResult.values per slot (the plan's argmax over {Y, N}
    under its tie policy; see main() / generate_batch). That choice agrees with
    sign(s_k(x)) in every case except an exact log-prob tie, which the plan's tie
    policy ("last" -> "N") resolves the same way sign(s_k(x)) <= 0 does, so F1 and
    AUPRC stay consistent: a slot counted "Y" for F1 has p(Y|slot k) > 0.5 in the
    AUPRC curve, and vice versa.

    Also reports a trivial-classifier AUPRC baseline: a classifier that assigns every
    slot instance the same (non-discriminating) score has a precision-recall curve
    flat at that slot's positive prevalence, so its AUPRC equals that prevalence. Since
    every one of the 8 slots has the same number of instances (n, one per example),
    the macro average of the 8 per-slot prevalences equals the single prevalence
    pooled over all 8*n slot instances -- so both the micro and macro trivial-AUPRC
    baselines reduce to that one pooled positive prevalence. The weighted-AUPRC
    baseline instead averages the 8 per-slot prevalences weighted by each slot's own
    positive count, so it generally differs from the micro/macro baseline whenever the
    slots' positive counts differ.

    "parse_failures" is always 0: every slot's Y/N answer token is forced by
    construction (see slot_scoring.score_slots), so there is no longer a code path
    that can fail to produce a usable answer for a slot. The field is kept, constant,
    purely for schema compatibility with existing downstream evaluation scripts that
    read it.

    "positive_rate_by_slot" / "tie_rate_by_slot" / "tie_rate_overall" are calibration
    diagnostics, not accuracy metrics -- see _slot_positive_rate, _slot_tie_rate and
    _tie_rate_overall.
    """
    from sklearn.metrics import average_precision_score, f1_score

    total = len(results)

    true_labels, pred_labels = _slot_instance_labels(results)
    trivial_pred_labels = [CLAUDETTE_NEG_CLASS] * len(true_labels)

    stats = {cls: _class_stats(true_labels, pred_labels, cls) for cls in CLAUDETTE_ALL_CLASSES}
    trivial_stats = {cls: _class_stats(true_labels, trivial_pred_labels, cls) for cls in CLAUDETTE_ALL_CLASSES}

    # Structurally impossible now (see docstring), kept at a constant 0 for schema
    # compatibility with existing downstream evaluation scripts.
    n_parse_failures = 0

    if results:
        y_true, y_score = _slot_probability_matrix(results)
        micro_auprc = average_precision_score(y_true, y_score, average="micro")
        macro_auprc = average_precision_score(y_true, y_score, average="macro")
        weighted_auprc = average_precision_score(y_true, y_score, average="weighted")
        per_slot_auprc_values = average_precision_score(y_true, y_score, average=None)
    else:
        micro_auprc = 0.0
        macro_auprc = 0.0
        weighted_auprc = 0.0
        per_slot_auprc_values = [0.0] * len(CLAUDETTE_METRICS)
    per_slot_auprc = {m: float(v) for m, v in zip(CLAUDETTE_METRICS, per_slot_auprc_values)}

    n_positive_slot_instances = sum(1 for t in true_labels if t != CLAUDETTE_NEG_CLASS)
    auprc_trivial_all_N = n_positive_slot_instances / len(true_labels) if true_labels else 0.0

    # weighted trivial baseline: weighted mean of each slot's own prevalence (that
    # slot's trivial AUPRC), weighted by the slot's positive count -- the same
    # weighting sklearn's average="weighted" applies to the real AUPRC values above.
    # Unlike the micro/macro trivial baselines, this generally differs from the pooled
    # prevalence whenever the 8 slots don't all have the same positive count.
    if results:
        per_slot_positive_counts = [sum(row[k] for row in y_true) for k in range(len(CLAUDETTE_METRICS))]
        n_examples = len(y_true)
        total_positive_count = sum(per_slot_positive_counts)
        if total_positive_count:
            weighted_auprc_trivial_all_N = sum(
                c * (c / n_examples) for c in per_slot_positive_counts
            ) / total_positive_count
        else:
            weighted_auprc_trivial_all_N = 0.0
    else:
        weighted_auprc_trivial_all_N = 0.0

    y_true_matrix, y_pred_matrix = _slot_label_matrices(results)
    if results:
        trivial_pred_matrix = [[0] * len(CLAUDETTE_METRICS) + [1] for _ in results]
        micro_f1_8plus1 = f1_score(y_true_matrix, y_pred_matrix, average="micro", zero_division=0)
        macro_f1_8plus1 = f1_score(y_true_matrix, y_pred_matrix, average="macro", zero_division=0)
        micro_f1_8plus1_trivial_all_N = f1_score(y_true_matrix, trivial_pred_matrix, average="micro", zero_division=0)
        macro_f1_8plus1_trivial_all_N = f1_score(y_true_matrix, trivial_pred_matrix, average="macro", zero_division=0)
    else:
        micro_f1_8plus1 = 0.0
        macro_f1_8plus1 = 0.0
        micro_f1_8plus1_trivial_all_N = 0.0
        macro_f1_8plus1_trivial_all_N = 0.0

    positive_rate_by_slot = _slot_positive_rate(results)
    tie_rate_by_slot = _slot_tie_rate(results)

    classes_out = {}
    for cls in CLAUDETTE_ALL_CLASSES:
        s = stats[cls]
        t = trivial_stats[cls]
        classes_out[cls] = {
            "support": s["support"],
            "precision": s["precision"],
            "recall": s["recall"],
            "f1": s["f1"],
            "precision_trivial_all_N": t["precision"],
            "recall_trivial_all_N": t["recall"],
            "f1_trivial_all_N": t["f1"],
        }
        if cls in per_slot_auprc:
            classes_out[cls]["auprc"] = per_slot_auprc[cls]
        if cls in positive_rate_by_slot:
            classes_out[cls]["positive_rate"] = positive_rate_by_slot[cls]
            classes_out[cls]["tie_rate"] = tie_rate_by_slot[cls]

    return {
        "total": total,
        "total_slot_instances": len(true_labels),
        "parse_failures": n_parse_failures,
        "micro_f1_8": _micro_f1(stats, CLAUDETTE_METRICS),
        "macro_f1_8": _macro_f1(stats, CLAUDETTE_METRICS),
        "micro_f1_8_trivial_all_N": _micro_f1(trivial_stats, CLAUDETTE_METRICS),
        "macro_f1_8_trivial_all_N": _macro_f1(trivial_stats, CLAUDETTE_METRICS),
        "micro_f1_8plus1_slot_pair": _micro_f1(stats, CLAUDETTE_ALL_CLASSES),
        "macro_f1_8plus1_slot_pair": _macro_f1(stats, CLAUDETTE_ALL_CLASSES),
        "micro_f1_8plus1_slot_pair_trivial_all_N": _micro_f1(trivial_stats, CLAUDETTE_ALL_CLASSES),
        "macro_f1_8plus1_slot_pair_trivial_all_N": _macro_f1(trivial_stats, CLAUDETTE_ALL_CLASSES),
        "micro_f1_8plus1": float(micro_f1_8plus1),
        "macro_f1_8plus1": float(macro_f1_8plus1),
        "micro_f1_8plus1_trivial_all_N": float(micro_f1_8plus1_trivial_all_N),
        "macro_f1_8plus1_trivial_all_N": float(macro_f1_8plus1_trivial_all_N),
        "micro_auprc_8": float(micro_auprc),
        "macro_auprc_8": float(macro_auprc),
        "weighted_auprc_8": float(weighted_auprc),
        "micro_auprc_8_trivial_all_N": auprc_trivial_all_N,
        "macro_auprc_8_trivial_all_N": auprc_trivial_all_N,
        "weighted_auprc_8_trivial_all_N": float(weighted_auprc_trivial_all_N),
        "positive_rate_by_slot": positive_rate_by_slot,
        "tie_rate_by_slot": tie_rate_by_slot,
        "tie_rate_overall": _tie_rate_overall(results),
        "classes": classes_out,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Local model directory for Llama-3.1-8b-Instruct")
    p.add_argument("--data-tsv", required=True, help="CLAUDETTE-TOS TSV file (claudette_tos_test.tsv)")
    p.add_argument("--few-shot-tsv", default=None, help="Optional TSV with few-shot examples")
    p.add_argument("--mode", choices=("plain", "icl", "fine_tuned"), default="plain")
    p.add_argument("--icl-k", type=int, default=3, help="Number of few-shot examples to include")
    p.add_argument("--lora-path", default=None, help="Path to LoRA adapter weights (required for --mode fine_tuned); merged onto the base model from --model-path")
    p.add_argument("--device", default="cuda", help="Device to run model on (cuda or cpu)")
    p.add_argument("--dtype", default="bfloat16", help="Dtype for model init (bfloat16 or float16)")
    p.add_argument("--max-input-tokens", type=int, default=None, help="Optional cap for prompt tokens before generation; omit to keep the full prompt")
    p.add_argument("--max-prompts", type=int, default=0, help="Limit number of prompts (0 = all)")
    p.add_argument("--batch-size", type=int, default=32, help="Number of prompts to generate in a single batched forward pass")
    p.add_argument("--output-jsonl", default="claudette_benchmark_results.jsonl", help="Per-example JSONL output")
    p.add_argument("--resume", action="store_true", help="Resume from existing output JSONL if present")
    return p.parse_args()


def build_chat_messages(args, query_prompt: str, few_shots: list[dict]) -> list[dict]:
    if args.mode == "plain":
        return [
            {"role": "system", "content": CLAUDETTE_SYSTEM_PROMPT},
            {"role": "user", "content": query_prompt},
        ]
    if args.mode == "icl":
        k = min(args.icl_k, len(few_shots))
        examples = few_shots[:k]
        messages = [{"role": "system", "content": CLAUDETTE_SYSTEM_PROMPT}]
        for ex in examples:
            ex_sentence = ex.get("prompt", "")
            ex_vector = ex.get("label", ex.get("gt", ""))
            messages.append({"role": "user", "content": ex_sentence})
            messages.append({"role": "assistant", "content": ex_vector})
        messages.append({"role": "user", "content": query_prompt})
        return messages
    # "fine_tuned" queries the model directly, without few-shot examples, but still
    # with the same system prompt as the other arms.
    return [
        {"role": "system", "content": CLAUDETTE_SYSTEM_PROMPT},
        {"role": "user", "content": query_prompt},
    ]


def render_chat_text(tokenizer, chat_messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch(
    hf_model,
    tokenizer,
    texts: list[str],
    *,
    max_input_tokens: int | None,
    device: str,
    plan: SlotPlan,
) -> tuple[list[dict], list[str], list[dict], list[dict]]:
    """Thin CLAUDETTE-specific wrapper around slot_scoring.score_slots().

    Tokenizes/left-pads `texts` (already rendered through the chat template) and
    hands the resulting batch to the benchmark-agnostic constrained-slot-scoring
    core, then turns each example's SlotResult into, per example:
      - the {metric: "Y"/"N"} dict taken straight from SlotResult.values (the
        plan's chosen candidate per slot -- the single source of truth for the
        prediction),
      - the CLAUDETTE vector string, rendered from exactly those same values via
        plan.answer_string (the template the plan was built from),
      - the {metric: log-odds} dict, and
      - the {metric: tie} dict.
    No free generation happens here: the only per-slot model decision is the Y/N
    value token, forced by the plan (see score_slots).
    """
    # `texts` were already rendered through the chat template (tokenize=False), so the
    # special/control tokens (BOS, header tokens, ...) are already present as literal text.
    # add_special_tokens=False avoids the tokenizer prepending a second BOS on top of that.
    encoded = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts]
    if max_input_tokens:
        encoded = [ids[-max_input_tokens:] for ids in encoded]

    # tokenizer.padding_side is "left" (set in generator_uni.build_model), so this left-pads
    # the batch, which is what a causal LM needs for correct batched generation.
    padded = tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt")
    input_ids = padded["input_ids"].to(device)
    attention_mask = padded["attention_mask"].to(device)

    try:
        slot_results = score_slots(hf_model, input_ids, attention_mask, plan, device=device)
    except torch.cuda.OutOfMemoryError:
        # Each forward call in score_slots() only processes a handful of new tokens
        # (one chosen value token + one short fragment) rather than the full padded
        # sequence, so the per-step vocab-logits peak is already much smaller than a
        # free-generation path's (batch_size * full_seq_len * vocab_size). Very long
        # prompts (the initial prompt+fragment[0] forward call) can still blow past
        # available memory before the requested batch size is reachable, though,
        # independent of --max-input-tokens. Splitting the batch in half and retrying
        # is the standard fallback for that.
        if device == "cuda":
            torch.cuda.empty_cache()
        if len(texts) <= 1:
            raise
        mid = len(texts) // 2
        print(f"CUDA OOM at batch size {len(texts)}. Splitting into sub-batches of {mid} and {len(texts) - mid}...")
        first_pred, first_vectors, first_scores, first_ties = generate_batch(
            hf_model, tokenizer, texts[:mid], max_input_tokens=max_input_tokens, device=device, plan=plan,
        )
        second_pred, second_vectors, second_scores, second_ties = generate_batch(
            hf_model, tokenizer, texts[mid:], max_input_tokens=max_input_tokens, device=device, plan=plan,
        )
        return (
            first_pred + second_pred,
            first_vectors + second_vectors,
            first_scores + second_scores,
            first_ties + second_ties,
        )

    predicted_metrics_batch = []
    vectors = []
    slot_log_odds_batch = []
    slot_ties_batch = []
    for result in slot_results:
        # SlotResult.values is the plan's chosen candidate per slot (argmax over
        # {Y, N} under the plan's tie policy). It is the single source of truth for
        # the prediction: `vector` is only its string rendering, and the caller's
        # predicted_metrics is this dict verbatim.
        pred = {m: result.values[m] for m in CLAUDETTE_METRICS}
        predicted_metrics_batch.append(pred)
        vectors.append(plan.answer_string([pred[m] for m in CLAUDETTE_METRICS]))
        slot_log_odds_batch.append({m: result.log_odds(m) for m in CLAUDETTE_METRICS})
        slot_ties_batch.append({m: result.ties[m] for m in CLAUDETTE_METRICS})
    return predicted_metrics_batch, vectors, slot_log_odds_batch, slot_ties_batch


def main():
    args = parse_args()
    if args.mode == "fine_tuned" and not args.lora_path:
        raise SystemExit("--lora-path is required when --mode fine_tuned")

    records = load_claudette_tsv(args.data_tsv)
    if args.max_prompts > 0:
        records = records[: args.max_prompts]
    if not records:
        raise SystemExit(f"No records loaded from --data-tsv {args.data_tsv!r}")

    # Prepare few-shot examples if requested
    few_shots = []
    if args.few_shot_tsv and args.mode == "icl":
        few_shots = load_claudette_tsv(args.few_shot_tsv)

    lora_path = args.lora_path if args.mode == "fine_tuned" else None
    model = LocalModel(args.model_path, device=args.device, dtype=args.dtype, lora_path=lora_path)
    model.load()
    tokenizer = model.tokenizer
    # Batched generation needs direct access to the underlying HF model (LocalModel/UnifiedGenerator
    # only expose a single-prompt generate() call), without touching the shared wrapper used by the
    # other benchmark scripts.
    hf_model = model._gen._model

    plan = build_slot_plan(
        tokenizer,
        CLAUDETTE_METRICS,
        CLAUDETTE_FRAGMENTS,
        CLAUDETTE_CANDIDATES,
        leads=_plan_lead_texts(tokenizer),
        # ["Y", "N"] with "last": most sentences are "N" (see CLAUDETTE_SYSTEM_PROMPT),
        # so an exact log-prob tie at the decision boundary resolves to the majority
        # class instead of systematically inflating "Y".
        tie_policy="last",
    )
    print(plan.describe(tokenizer))

    # Sanity check (fail fast, once, at startup): the ground-truth vector format the
    # rest of this script assumes must match what the plan can actually score. If a
    # fine-tuned model was trained on targets in a different format than the plan's
    # answer template, this raises SlotPlanError immediately instead of silently
    # producing numbers that don't mean what they claim to.
    sample_gt = records[0]["gt"]
    sample_gt_metrics = extract_claudette_metrics_from_text(sample_gt)
    sample_values = [sample_gt_metrics[m] for m in CLAUDETTE_METRICS]
    if any(v is None for v in sample_values):
        raise SystemExit(f"Could not parse a full Y/N vector from the first record's gt for the startup sanity check: {sample_gt!r}")
    assert_target_matches_plan(plan, sample_gt, sample_values)

    # In --mode icl the few-shot assistant turns are injected verbatim into every
    # prompt as the answer format the model is meant to imitate. Run the same
    # parse + plan-format sanity check over exactly the few-shot examples that
    # will be used (few_shots[:icl_k]), so a few-shot TSV in the wrong vector
    # format -- or a single malformed row -- fails fast at startup instead of
    # silently teaching the model a format the scoring plan cannot represent.
    if args.mode == "icl":
        n_icl = min(args.icl_k, len(few_shots))
        if n_icl == 0:
            print(
                f"Warning: --mode icl but no few-shot examples to inject "
                f"(few_shot_tsv={args.few_shot_tsv!r}, icl_k={args.icl_k}); running 0-shot."
            )
        for i, ex in enumerate(few_shots[:n_icl]):
            ex_vector = ex.get("label", ex.get("gt", ""))
            ex_metrics = extract_claudette_metrics_from_text(ex_vector)
            ex_values = [ex_metrics[m] for m in CLAUDETTE_METRICS]
            if any(v is None for v in ex_values):
                raise SystemExit(
                    f"Few-shot example {i} from --few-shot-tsv {args.few_shot_tsv!r} does not "
                    f"contain a full Y/N vector: {ex_vector!r}"
                )
            try:
                assert_target_matches_plan(plan, ex_vector, ex_values)
            except SlotPlanError as e:
                raise SystemExit(
                    f"Few-shot example {i} from --few-shot-tsv {args.few_shot_tsv!r} is not in "
                    f"the scoring plan's answer format:\n{e}"
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
            batch_predicted_metrics, batch_vectors, batch_slot_log_odds, batch_slot_ties = generate_batch(
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

            batch_predicted_metrics, batch_vectors, batch_slot_log_odds, batch_slot_ties = generate_batch(
                hf_model,
                tokenizer,
                batch_texts,
                max_input_tokens=fallback_max_input,
                device=args.device,
                plan=plan,
            )

        for offset, (rec, query_prompt, pred_metrics, vector, slot_log_odds, slot_ties) in enumerate(
            zip(batch_records, batch_query_prompts, batch_predicted_metrics, batch_vectors, batch_slot_log_odds, batch_slot_ties)
        ):
            gt = rec.get("gt", "")
            # predicted_metrics comes straight from SlotResult.values (the plan's chosen
            # Y/N candidate per slot), returned by generate_batch. `vector` is only its
            # string rendering (plan.answer_string of the same values). The AUPRC score
            # s_k(x) = slot_log_odds[m] has the same sign as this choice in every case
            # except an exact log-prob tie, which the plan's tie policy ("last" -> "N")
            # resolves deterministically -- so F1 and AUPRC still cannot disagree about
            # which slots were predicted "Y".
            predicted_metrics = dict(pred_metrics)
            gt_metrics = extract_claudette_metrics_from_text(gt)
            result = {
                "index": idx + offset,
                "prompt": query_prompt,
                "raw_output": vector,
                "predicted_vector": vector,
                "predicted_metrics": predicted_metrics,
                "gt": gt,
                "gt_metrics": gt_metrics,
                "correct": metrics_match(predicted_metrics, gt_metrics),
                "slot_log_odds": slot_log_odds,
                "ties": slot_ties,
            }
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(result)

        idx += batch_size

    fh.close()

    summary = evaluate_claudette_predictions(results)
    print("Summary:")
    print(f"  total: {summary['total']}")
    print(f"  total_slot_instances: {summary['total_slot_instances']}")
    print(
        f"  parse_failures: {summary['parse_failures']}  "
        f"(always 0: every slot's answer is forced by construction, kept for schema compatibility)"
    )
    print(f"  tie_rate_overall: {summary['tie_rate_overall']:.4f}")
    print()
    print("  8-class scenario (LTD, TER, CH, CR, USE, LAW, J, ARB):")
    print(
        f"    micro_f1: {summary['micro_f1_8']:.4f}  "
        f"(trivial all-N baseline: {summary['micro_f1_8_trivial_all_N']:.4f})"
    )
    print(
        f"    macro_f1: {summary['macro_f1_8']:.4f}  "
        f"(trivial all-N baseline: {summary['macro_f1_8_trivial_all_N']:.4f})"
    )
    print(
        f"    micro_auprc: {summary['micro_auprc_8']:.4f}  "
        f"(trivial all-N baseline: {summary['micro_auprc_8_trivial_all_N']:.4f})"
    )
    print(
        f"    macro_auprc: {summary['macro_auprc_8']:.4f}  "
        f"(trivial all-N baseline: {summary['macro_auprc_8_trivial_all_N']:.4f})"
    )
    print(
        f"    weighted_auprc: {summary['weighted_auprc_8']:.4f}  "
        f"(trivial all-N baseline: {summary['weighted_auprc_8_trivial_all_N']:.4f})"
    )
    print()
    print("  8+1-class scenario (adds negative class 'N'), LexGLUE-style (per-example, 9 labels):")
    print(
        f"    micro_f1: {summary['micro_f1_8plus1']:.4f}  "
        f"(trivial all-N baseline: {summary['micro_f1_8plus1_trivial_all_N']:.4f})"
    )
    print(
        f"    macro_f1: {summary['macro_f1_8plus1']:.4f}  "
        f"(trivial all-N baseline: {summary['macro_f1_8plus1_trivial_all_N']:.4f})"
    )
    print()
    print("  8+1-class scenario (adds pooled negative class 'N'), slot-pair style (legacy):")
    print(
        f"    micro_f1: {summary['micro_f1_8plus1_slot_pair']:.4f}  "
        f"(trivial all-N baseline: {summary['micro_f1_8plus1_slot_pair_trivial_all_N']:.4f})"
    )
    print(
        f"    macro_f1: {summary['macro_f1_8plus1_slot_pair']:.4f}  "
        f"(trivial all-N baseline: {summary['macro_f1_8plus1_slot_pair_trivial_all_N']:.4f})"
    )
    print()
    print("  per_class:")
    for cls in CLAUDETTE_ALL_CLASSES:
        c = summary["classes"][cls]
        auprc_str = f"  auprc={c['auprc']:.4f}" if "auprc" in c else ""
        rate_str = (
            f"  positive_rate={c['positive_rate']:.4f}  tie_rate={c['tie_rate']:.4f}"
            if "positive_rate" in c else ""
        )
        print(
            f"    {cls}: support={c['support']}  precision={c['precision']:.4f}  "
            f"recall={c['recall']:.4f}  f1={c['f1']:.4f}{auprc_str}{rate_str}  |  "
            f"trivial all-N: precision={c['precision_trivial_all_N']:.4f}  "
            f"recall={c['recall_trivial_all_N']:.4f}  f1={c['f1_trivial_all_N']:.4f}"
        )

    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as sfh:
        json.dump(summary, sfh, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
