#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
import json
import re
from pathlib import Path
import torch

from cvss import CVSS3
from cvss.exceptions import CVSSError

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.data_loader import load_cti_vsp_tsv, load_cti_vsp_metric_classes
from benchmark_play_ground.prompt_builder import extract_cve_description_block
from benchmark_play_ground.model_wrapper import LocalModel


CVSS_METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

CVSS_VECTOR_RE = re.compile(
    r"CVSS:3\.[01]/AV:[NALP]/AC:[LH]/PR:[NLH]/UI:[NR]/S:[UC]/C:[NLH]/I:[NLH]/A:[NLH]"
)
CVSS_LOOSE_RE = re.compile(r"CVSS:3\.[01]/[A-Za-z:/]+")


CTI_VSP_SYSTEM_PROMPT = (
    "Analyze the CVE description and output the CVSS v3.1 Base vector string. "
    "Do not explain your reasoning. Output only the vector string and nothing else.\n"
    "Valid options for each metric:\n"
    "- Attack Vector (AV): N, A, L, P\n"
    "- Attack Complexity (AC): L, H\n"
    "- Privileges Required (PR): N, L, H\n"
    "- User Interaction (UI): N, R\n"
    "- Scope (S): U, C\n"
    "- Confidentiality (C): N, L, H\n"
    "- Integrity (I): N, L, H\n"
    "- Availability (A): N, L, H\n"
    "Output format (exactly this, no other text): "
    "CVSS:3.1/AV:_/AC:_/PR:_/UI:_/S:_/C:_/I:_/A:_"
)


def extract_cvss_vector_from_text(text: str) -> str | None:
    match = CVSS_VECTOR_RE.search(text)
    return match.group(0) if match else None


def extract_cvss_metrics_from_text(text: str) -> dict:
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
    """Reassemble a canonical 'CVSS:3.1/AV:.../.../A:...' string from a metrics dict.

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


def evaluate_cti_vsp_predictions(results: list[dict], metric_classes: dict) -> dict:
    total = len(results)
    exact_match = sum(1 for r in results if r["correct"])

    metric_slot_total = total * len(CVSS_METRICS)
    metric_slot_correct = sum(
        1
        for r in results
        for m in CVSS_METRICS
        if r["predicted_metrics"].get(m) == r["gt_metrics"].get(m)
    )

    per_metric = {}
    for metric in CVSS_METRICS:
        classes = metric_classes.get(metric, [])
        class_stats = {}
        weighted_f1_num = 0.0
        total_support = 0
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
            class_stats[cls] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
            weighted_f1_num += f1 * support
            total_support += support

        supported_f1s = [s["f1"] for s in class_stats.values() if s["support"] > 0]
        macro_f1 = sum(supported_f1s) / len(supported_f1s) if supported_f1s else 0.0
        weighted_f1 = weighted_f1_num / total_support if total_support else 0.0
        per_metric[metric] = {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "classes": class_stats}

    overall_macro_f1 = sum(per_metric[m]["macro_f1"] for m in CVSS_METRICS) / len(CVSS_METRICS)
    overall_weighted_f1 = sum(per_metric[m]["weighted_f1"] for m in CVSS_METRICS) / len(CVSS_METRICS)

    score_diffs = [
        abs(r["predicted_score"] - r["gt_score"])
        for r in results
        if r["predicted_score"] is not None and r["gt_score"] is not None
    ]
    n_scored = len(score_diffs)

    return {
        "total": total,
        "exact_match": exact_match,
        "exact_match_accuracy": exact_match / total if total else 0.0,
        "metric_slot_correct": metric_slot_correct,
        "metric_slot_total": metric_slot_total,
        "metric_slot_accuracy": metric_slot_correct / metric_slot_total if metric_slot_total else 0.0,
        "overall_macro_f1": overall_macro_f1,
        "overall_weighted_f1": overall_weighted_f1,
        "per_metric": per_metric,
        "cvss_score_mad": sum(score_diffs) / n_scored if n_scored else None,
        "cvss_score_n_scored": n_scored,
        "cvss_score_n_unscored": total - n_scored,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Local model directory for Llama-3.1-8b-Instruct")
    p.add_argument("--data-tsv", required=True, help="CTI-VSP TSV file (cti_vsp_benchmark_test_500.tsv)")
    p.add_argument(
        "--metric-distribution-json",
        default=str(ROOT_DIR / "benchmarks" / "cti_vsp" / "cti_vsp_gt_metric_distribution.json"),
        help="JSON file listing the ground-truth classes per CVSS metric (for macro-F1 labels)",
    )
    p.add_argument("--few-shot-tsv", default=None, help="Optional TSV with few-shot examples")
    p.add_argument("--mode", choices=("plain", "icl", "fine_tuned"), default="plain")
    p.add_argument("--icl-k", type=int, default=3, help="Number of few-shot examples to include")
    p.add_argument("--lora-path", default=None, help="Path to LoRA adapter weights (required for --mode fine_tuned); merged onto the base model from --model-path")
    p.add_argument("--device", default="cuda", help="Device to run model on (cuda or cpu)")
    p.add_argument("--dtype", default="bfloat16", help="Dtype for model init (bfloat16 or float16)")
    p.add_argument("--max-input-tokens", type=int, default=None, help="Optional cap for prompt tokens before generation; omit to keep the full prompt")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens to generate per prompt (CVSS answers may include per-metric reasoning before the final vector)")
    p.add_argument("--max-prompts", type=int, default=0, help="Limit number of prompts (0 = all)")
    p.add_argument("--output-jsonl", default="cti_vsp_benchmark_results.jsonl", help="Per-example JSONL output")
    p.add_argument("--resume", action="store_true", help="Resume from existing output JSONL if present")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "fine_tuned" and not args.lora_path:
        raise SystemExit("--lora-path is required when --mode fine_tuned")

    records = load_cti_vsp_tsv(args.data_tsv)
    if args.max_prompts > 0:
        records = records[: args.max_prompts]

    metric_classes = load_cti_vsp_metric_classes(args.metric_distribution_json)

    # Prepare few-shot examples if requested
    few_shots = []
    if args.few_shot_tsv and args.mode == "icl":
        few_shots = load_cti_vsp_tsv(args.few_shot_tsv)

    lora_path = args.lora_path if args.mode == "fine_tuned" else None
    model = LocalModel(args.model_path, device=args.device, dtype=args.dtype, lora_path=lora_path)
    model.load()
    tokenizer = model.tokenizer

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

    # Open output file: append if resuming, otherwise overwrite
    mode = "a" if (args.resume and out_path.exists()) else "w"
    fh = out_path.open(mode, encoding="utf-8")

    for idx in range(start_idx, len(records)):
        rec = records[idx]
        query_prompt = rec["prompt"]
        if args.mode == "plain":
            user_content = extract_cve_description_block(query_prompt)
            prompt = [
                {"role": "system", "content": CTI_VSP_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        elif args.mode == "icl":
            # choose up to k few-shot examples
            k = min(args.icl_k, len(few_shots))
            examples = few_shots[:k]
            user_content = extract_cve_description_block(query_prompt)
            prompt = [{"role": "system", "content": CTI_VSP_SYSTEM_PROMPT}]
            for ex in examples:
                ex_description = extract_cve_description_block(ex.get("prompt", ""))
                ex_vector = ex.get("label", ex.get("gt", ""))
                prompt.append({"role": "user", "content": ex_description})
                prompt.append({"role": "assistant", "content": ex_vector})
            prompt.append({"role": "user", "content": user_content})
        else:
            # "fine_tuned" queries the model directly, without few-shot examples
            prompt = query_prompt

        chat_messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        final_model_input = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print("=== Final model input (incl. special tokens) ===")
        print(final_model_input)
        print("=== End final model input ===")

        try:
            raw = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                stop_strings=["\nCVE Description:", "\n\n"],
                tokenizer=tokenizer,
                use_cache=True,
                max_input_tokens=args.max_input_tokens,
                use_chat_template=True,
            )
        except RuntimeError as e:
            msg = str(e)
            if "CUDA error" not in msg:
                raise

            # Retry once with a tighter context window to avoid transient GPU kernel failures.
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            fallback_max_input = args.max_input_tokens if args.max_input_tokens else 2048
            fallback_max_input = min(fallback_max_input, 2048)
            print(
                f"CUDA generation failed at index {idx}. Retrying with max_input_tokens={fallback_max_input}, use_cache=True..."
            )

            raw = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                stop_strings=["\nCVE Description:", "\n\n"],
                tokenizer=tokenizer,
                use_cache=True,
                max_input_tokens=fallback_max_input,
                use_chat_template=True,
            )

        gt = rec.get("gt", "")
        pred_vector = extract_cvss_vector_from_text(raw)
        predicted_metrics = extract_cvss_metrics_from_text(raw)
        gt_metrics = extract_cvss_metrics_from_text(gt)
        result = {
            "index": idx,
            "prompt": query_prompt,
            "raw_output": raw,
            "predicted_vector": pred_vector,
            "predicted_metrics": predicted_metrics,
            "predicted_score": cvss3_base_score(predicted_metrics),
            "gt": gt,
            "gt_metrics": gt_metrics,
            "gt_score": cvss3_base_score(gt_metrics),
            "correct": metrics_match(predicted_metrics, gt_metrics),
        }
        fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        fh.flush()
        results.append(result)

    fh.close()

    summary = evaluate_cti_vsp_predictions(results, metric_classes)
    print("Summary:")
    print(f"  total: {summary['total']}")
    print(f"  exact_match: {summary['exact_match']}")
    print(f"  exact_match_accuracy: {summary['exact_match_accuracy']:.4f}")
    print(
        f"  metric_slot_accuracy: {summary['metric_slot_accuracy']:.4f} "
        f"({summary['metric_slot_correct']}/{summary['metric_slot_total']})"
    )
    if summary["cvss_score_mad"] is not None:
        print(
            f"  cvss_score_mad: {summary['cvss_score_mad']:.4f} "
            f"(n_scored={summary['cvss_score_n_scored']}, n_unscored={summary['cvss_score_n_unscored']})"
        )
    else:
        print(f"  cvss_score_mad: n/a (no scoreable predictions; n_unscored={summary['cvss_score_n_unscored']})")
    print(f"  overall_macro_f1: {summary['overall_macro_f1']:.4f}")
    print(f"  overall_weighted_f1: {summary['overall_weighted_f1']:.4f}")
    print("  per_metric:")
    for metric in CVSS_METRICS:
        m = summary["per_metric"][metric]
        print(f"    {metric}: macro_f1={m['macro_f1']:.4f}  weighted_f1={m['weighted_f1']:.4f}")
        for cls, stats in m["classes"].items():
            print(
                f"      {cls}: support={stats['support']}  precision={stats['precision']:.4f}  "
                f"recall={stats['recall']:.4f}  f1={stats['f1']:.4f}"
            )

    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as sfh:
        json.dump(summary, sfh, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
