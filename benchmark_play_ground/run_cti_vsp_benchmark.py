#!/usr/bin/env python3
import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.data_loader import load_cti_vsp_tsv, load_cti_vsp_metric_classes
from benchmark_play_ground.prompt_builder import CTI_VSP_INSTRUCTION_PROMPT, build_cti_vsp_icl_prompt
from benchmark_play_ground.model_wrapper import LocalModel
from benchmark_play_ground.evaluator import (
    CVSS_METRICS,
    extract_cvss_vector_from_text,
    extract_cvss_metrics_from_text,
    evaluate_cti_vsp_predictions,
)


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
        if args.mode == "icl":
            # choose up to k few-shot examples
            k = min(args.icl_k, len(few_shots))
            examples = few_shots[:k]
            prompt = build_cti_vsp_icl_prompt(examples, query_prompt, CTI_VSP_INSTRUCTION_PROMPT)
        else:
            # "plain" and "fine_tuned" both query the model directly, without few-shot examples
            prompt = query_prompt

        print(prompt)

        try:
            raw = model.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_input_tokens=args.max_input_tokens,
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
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_input_tokens=fallback_max_input,
            )

        gt = rec.get("gt", "")
        pred_vector = extract_cvss_vector_from_text(raw)
        result = {
            "index": idx,
            "prompt": query_prompt,
            "raw_output": raw,
            "predicted_vector": pred_vector,
            "predicted_metrics": extract_cvss_metrics_from_text(raw),
            "gt": gt,
            "gt_metrics": extract_cvss_metrics_from_text(gt),
            "correct": bool(pred_vector) and pred_vector == gt,
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
    print(f"  overall_macro_f1: {summary['overall_macro_f1']:.4f}")
    print("  per_metric_macro_f1:")
    for metric in CVSS_METRICS:
        print(f"    {metric}: {summary['per_metric_macro_f1'][metric]:.4f}")

    summary_path = out_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as sfh:
        json.dump(summary, sfh, indent=2)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
