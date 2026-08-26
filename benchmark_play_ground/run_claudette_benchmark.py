#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
import json
import re
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.data_loader import load_claudette_tsv
from benchmark_play_ground.model_wrapper import LocalModel


CLAUDETTE_METRICS = ["LTD", "TER", "CH", "CR", "USE", "LAW", "J", "ARB"]
CLAUDETTE_NEG_CLASS = "N"
CLAUDETTE_ALL_CLASSES = CLAUDETTE_METRICS + [CLAUDETTE_NEG_CLASS]

# Fallback log-odds assigned to a slot whose Y/N answer token position could not be
# located in the generated output (e.g. malformed/truncated generation). Such a slot
# is scored as a maximally confident "N" so it doesn't produce a NaN/undefined point
# on the precision-recall curve.
CLAUDETTE_MISSING_LOG_ODDS = -1e9

CLAUDETTE_VECTOR_RE = re.compile(
    r"LTD:[YN]\|TER:[YN]\|CH:[YN]\|CR:[YN]\|USE:[YN]\|LAW:[YN]\|J:[YN]\|ARB:[YN]\|?"
)
CLAUDETTE_LOOSE_RE = re.compile(r"LTD:[A-Za-z:|]+")


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
    "LTD:?|TER:?|CH:?|CR:?|USE:?|LAW:?|J:?|ARB:?"
)


def extract_claudette_vector_from_text(text: str) -> str | None:
    match = CLAUDETTE_VECTOR_RE.search(text)
    return match.group(0) if match else None


def extract_claudette_metrics_from_text(text: str) -> dict:
    metrics = {m: None for m in CLAUDETTE_METRICS}
    vector = extract_claudette_vector_from_text(text)
    if vector is None:
        loose = CLAUDETTE_LOOSE_RE.search(text)
        vector = loose.group(0) if loose else None
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


def locate_slot_char_positions(text: str) -> dict:
    """Find, for each of the 8 metric slots, the character offset of its Y/N value
    within `text` (the raw decoded model output).

    Mirrors extract_claudette_metrics_from_text's parsing (same vector extraction,
    same "|"-split) but additionally tracks *where* each value character sits, so it
    can be mapped back to the generation step that produced it.
    """
    positions = {m: None for m in CLAUDETTE_METRICS}
    vector = extract_claudette_vector_from_text(text)
    if vector is None:
        loose = CLAUDETTE_LOOSE_RE.search(text)
        vector = loose.group(0) if loose else None
    if not vector:
        return positions
    vector_start = text.find(vector)
    if vector_start == -1:
        return positions
    offset = vector_start
    for segment in vector.split("|"):
        key, sep, value = segment.partition(":")
        key = key.strip()
        if key in positions and value:
            positions[key] = offset + len(key) + len(sep)
        offset += len(segment) + 1  # +1 accounts for the "|" separator
    return positions


def _single_char_token_id(tokenizer, ch: str) -> int:
    ids = tokenizer.encode(ch, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected {ch!r} to tokenize to a single token, got {ids!r}")
    return ids[0]


def _token_char_lengths(tokenizer, token_ids: list[int]) -> list[int]:
    """Cumulative length (in characters) of the decoded text after each token.

    Decoding is redone incrementally (rather than per-token) so that tokenizer
    spacing/merge quirks match exactly how the full generated text was decoded.
    """
    lengths = []
    for i in range(1, len(token_ids) + 1):
        lengths.append(len(tokenizer.decode(token_ids[:i], skip_special_tokens=True)))
    return lengths


def _char_offset_to_token_index(char_lengths: list[int], char_offset: int) -> int | None:
    for i, length in enumerate(char_lengths):
        if char_offset < length:
            return i
    return None


def compute_slot_log_odds(tokenizer, token_ids: list[int], step_scores: tuple, batch_idx: int, raw_text: str, y_id: int, n_id: int) -> dict:
    """Compute s_k(x) = log p(Y | position k) - log p(N | position k) for each of the
    8 metric slots, where "position k" is the generation step at which the model
    produced slot k's Y/N answer character.

    `step_scores` is the `scores` tuple from `model.generate(..., output_scores=True,
    return_dict_in_generate=True)`: one [vocab] logit row per generation step, shared
    across the batch. Missing slots (answer token position not found) get
    CLAUDETTE_MISSING_LOG_ODDS.
    """
    result = {m: CLAUDETTE_MISSING_LOG_ODDS for m in CLAUDETTE_METRICS}
    positions = locate_slot_char_positions(raw_text)
    char_lengths = _token_char_lengths(tokenizer, token_ids)
    for m, char_offset in positions.items():
        if char_offset is None:
            continue
        tok_idx = _char_offset_to_token_index(char_lengths, char_offset)
        if tok_idx is None or tok_idx >= len(step_scores):
            continue
        logits = step_scores[tok_idx][batch_idx].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        result[m] = (log_probs[y_id] - log_probs[n_id]).item()
    return result


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


def _slot_log_odds_matrix(results: list[dict]) -> tuple[list[list[int]], list[list[float]]]:
    """Build the (n_results x 8) ground-truth and log-odds-score matrices used for AUPRC.

    Row i, column k is slot CLAUDETTE_METRICS[k] of result i: y_true is 1 if that slot
    is "Y" in the ground truth, y_score is s_k(x) = log p(Y|position k) - log p(N|position k)
    (falls back to CLAUDETTE_MISSING_LOG_ODDS if the result has no "slot_log_odds").
    """
    y_true = []
    y_score = []
    for r in results:
        slot_log_odds = r.get("slot_log_odds") or {}
        y_true.append([1 if r["gt_metrics"].get(m) == "Y" else 0 for m in CLAUDETTE_METRICS])
        y_score.append([slot_log_odds.get(m, CLAUDETTE_MISSING_LOG_ODDS) for m in CLAUDETTE_METRICS])
    return y_true, y_score


def evaluate_claudette_predictions(results: list[dict]) -> dict:
    """Score predictions under two multi-class framings of the 8 clause-type slots.

    8-class scenario: the 8 clause types (LTD, TER, ... ARB) are the classes;
    micro/macro F1 are computed over those 8 classes only.
    8+1-class scenario: adds a single pooled negative class "N" (every slot
    that is "no", across all 8 slots and all sentences) as a 9th class.

    For every metric, also reports how a trivial classifier that always
    predicts "N" for every slot would score, as a baseline for comparison.

    Additionally computes micro/macro AUPRC over the 8 clause-type slots, treated as
    an 8-label multi-label problem: for each slot k, the score is the log-odds
    s_k(x) = log p(Y|position k) - log p(N|position k) pulled from the model's output
    distribution at the generation step where it answered that slot (see
    compute_slot_log_odds). Micro-AUPRC pools all 8*n slot instances into a single
    precision-recall curve; macro-AUPRC averages the per-slot AUPRC across the 8 slots.
    """
    from sklearn.metrics import average_precision_score

    total = len(results)

    true_labels, pred_labels = _slot_instance_labels(results)
    trivial_pred_labels = [CLAUDETTE_NEG_CLASS] * len(true_labels)

    stats = {cls: _class_stats(true_labels, pred_labels, cls) for cls in CLAUDETTE_ALL_CLASSES}
    trivial_stats = {cls: _class_stats(true_labels, trivial_pred_labels, cls) for cls in CLAUDETTE_ALL_CLASSES}

    y_true, y_score = _slot_log_odds_matrix(results)
    micro_auprc = average_precision_score(y_true, y_score, average="micro") if total else 0.0
    macro_auprc = average_precision_score(y_true, y_score, average="macro") if total else 0.0
    per_slot_auprc_values = average_precision_score(y_true, y_score, average=None) if total else [0.0] * len(CLAUDETTE_METRICS)
    per_slot_auprc = {m: float(v) for m, v in zip(CLAUDETTE_METRICS, per_slot_auprc_values)}

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

    return {
        "total": total,
        "total_slot_instances": len(true_labels),
        "micro_f1_8": _micro_f1(stats, CLAUDETTE_METRICS),
        "macro_f1_8": _macro_f1(stats, CLAUDETTE_METRICS),
        "micro_f1_8_trivial_all_N": _micro_f1(trivial_stats, CLAUDETTE_METRICS),
        "macro_f1_8_trivial_all_N": _macro_f1(trivial_stats, CLAUDETTE_METRICS),
        "micro_f1_8plus1": _micro_f1(stats, CLAUDETTE_ALL_CLASSES),
        "macro_f1_8plus1": _macro_f1(stats, CLAUDETTE_ALL_CLASSES),
        "micro_f1_8plus1_trivial_all_N": _micro_f1(trivial_stats, CLAUDETTE_ALL_CLASSES),
        "macro_f1_8plus1_trivial_all_N": _macro_f1(trivial_stats, CLAUDETTE_ALL_CLASSES),
        "micro_auprc_8": float(micro_auprc),
        "macro_auprc_8": float(macro_auprc),
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
    p.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens to generate per prompt (answers are a single short label line)")
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
    # "fine_tuned" queries the model directly, without few-shot examples
    return [{"role": "user", "content": query_prompt}]


def render_chat_text(tokenizer, chat_messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch(hf_model, tokenizer, texts: list[str], *, max_new_tokens: int, max_input_tokens: int | None, device: str, y_id: int, n_id: int) -> tuple[list[str], list[dict]]:
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
        outputs = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stop_strings=["\n\n"],
            tokenizer=tokenizer,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    except torch.cuda.OutOfMemoryError:
        # This transformers version computes logits over the *full* padded sequence
        # (no logits_to_keep slicing), so peak memory scales with batch_size * seq_len *
        # vocab_size. Long prompts can blow this up well before the requested batch size
        # is actually reachable, independent of --max-input-tokens. Splitting the batch
        # in half and retrying is the standard fallback for that.
        if device == "cuda":
            torch.cuda.empty_cache()
        if len(texts) <= 1:
            raise
        mid = len(texts) // 2
        print(f"CUDA OOM at batch size {len(texts)}. Splitting into sub-batches of {mid} and {len(texts) - mid}...")
        first_texts, first_scores = generate_batch(hf_model, tokenizer, texts[:mid], max_new_tokens=max_new_tokens, max_input_tokens=max_input_tokens, device=device, y_id=y_id, n_id=n_id)
        second_texts, second_scores = generate_batch(hf_model, tokenizer, texts[mid:], max_new_tokens=max_new_tokens, max_input_tokens=max_input_tokens, device=device, y_id=y_id, n_id=n_id)
        return first_texts + second_texts, first_scores + second_scores

    gen_tokens = outputs.sequences[:, input_ids.shape[1]:]
    decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    slot_log_odds_batch = [
        compute_slot_log_odds(tokenizer, gen_tokens[b].tolist(), outputs.scores, b, decoded[b], y_id, n_id)
        for b in range(len(texts))
    ]
    return decoded, slot_log_odds_batch


def main():
    args = parse_args()
    if args.mode == "fine_tuned" and not args.lora_path:
        raise SystemExit("--lora-path is required when --mode fine_tuned")

    records = load_claudette_tsv(args.data_tsv)
    if args.max_prompts > 0:
        records = records[: args.max_prompts]

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
    y_id = _single_char_token_id(tokenizer, "Y")
    n_id = _single_char_token_id(tokenizer, "N")

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
            batch_raw, batch_slot_log_odds = generate_batch(
                hf_model,
                tokenizer,
                batch_texts,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                device=args.device,
                y_id=y_id,
                n_id=n_id,
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

            batch_raw, batch_slot_log_odds = generate_batch(
                hf_model,
                tokenizer,
                batch_texts,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=fallback_max_input,
                device=args.device,
                y_id=y_id,
                n_id=n_id,
            )

        for offset, (rec, query_prompt, raw, slot_log_odds) in enumerate(zip(batch_records, batch_query_prompts, batch_raw, batch_slot_log_odds)):
            gt = rec.get("gt", "")
            pred_vector = extract_claudette_vector_from_text(raw)
            predicted_metrics = extract_claudette_metrics_from_text(raw)
            gt_metrics = extract_claudette_metrics_from_text(gt)
            result = {
                "index": idx + offset,
                "prompt": query_prompt,
                "raw_output": raw,
                "predicted_vector": pred_vector,
                "predicted_metrics": predicted_metrics,
                "gt": gt,
                "gt_metrics": gt_metrics,
                "correct": metrics_match(predicted_metrics, gt_metrics),
                "slot_log_odds": slot_log_odds,
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
    print(f"    micro_auprc: {summary['micro_auprc_8']:.4f}")
    print(f"    macro_auprc: {summary['macro_auprc_8']:.4f}")
    print()
    print("  8+1-class scenario (adds pooled negative class 'N'):")
    print(
        f"    micro_f1: {summary['micro_f1_8plus1']:.4f}  "
        f"(trivial all-N baseline: {summary['micro_f1_8plus1_trivial_all_N']:.4f})"
    )
    print(
        f"    macro_f1: {summary['macro_f1_8plus1']:.4f}  "
        f"(trivial all-N baseline: {summary['macro_f1_8plus1_trivial_all_N']:.4f})"
    )
    print()
    print("  per_class:")
    for cls in CLAUDETTE_ALL_CLASSES:
        c = summary["classes"][cls]
        auprc_str = f"  auprc={c['auprc']:.4f}" if "auprc" in c else ""
        print(
            f"    {cls}: support={c['support']}  precision={c['precision']:.4f}  "
            f"recall={c['recall']:.4f}  f1={c['f1']:.4f}{auprc_str}  |  "
            f"trivial all-N: precision={c['precision_trivial_all_N']:.4f}  "
            f"recall={c['recall_trivial_all_N']:.4f}  f1={c['f1_trivial_all_N']:.4f}"
        )

    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as sfh:
        json.dump(summary, sfh, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
