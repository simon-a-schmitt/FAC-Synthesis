#!/usr/bin/env python3
import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.data_loader import load_casehold_tsv
from benchmark_play_ground.prompt_builder import INSTRUCTION_PROMPT, build_plain_prompt, build_icl_prompt
from benchmark_play_ground.model_wrapper import LocalModel
from benchmark_play_ground.evaluator import extract_label_from_text, evaluate_predictions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Local model directory for Llama-3.1-8b-Instruct")
    p.add_argument("--data-tsv", required=True, help="CaseHold TSV file (casehold_extract.tsv)")
    p.add_argument("--few-shot-tsv", default=None, help="Optional TSV with few-shot examples")
    p.add_argument("--mode", choices=("plain", "icl"), default="plain")
    p.add_argument("--icl-k", type=int, default=3, help="Number of few-shot examples to include")
    p.add_argument("--device", default="cuda", help="Device to run model on (cuda or cpu)")
    p.add_argument("--dtype", default="bfloat16", help="Dtype for model init (bfloat16 or float16)")
    p.add_argument("--max-input-tokens", type=int, default=None, help="Optional cap for prompt tokens before generation; omit to keep the full prompt")
    p.add_argument("--max-prompts", type=int, default=0, help="Limit number of prompts (0 = all)")
    p.add_argument("--output-jsonl", default="casehold_benchmark_results.jsonl", help="Per-example JSONL output")
    p.add_argument("--resume", action="store_true", help="Resume from existing output JSONL if present")
    return p.parse_args()


def main():
    args = parse_args()
    records = load_casehold_tsv(args.data_tsv, instruction_prompt=INSTRUCTION_PROMPT)
    if args.max_prompts > 0:
        records = records[: args.max_prompts]

    # Prepare few-shot examples if requested
    few_shots = []
    if args.few_shot_tsv and args.mode == "icl":
        few_shots = load_casehold_tsv(args.few_shot_tsv, instruction_prompt=INSTRUCTION_PROMPT)

    model = LocalModel(args.model_path, device=args.device, dtype=args.dtype)
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
        if args.mode == "plain":
            prompt = query_prompt
        else:
            # choose up to k few-shot examples
            k = min(args.icl_k, len(few_shots))
            examples = few_shots[:k]
            prompt = build_icl_prompt(examples, query_prompt, INSTRUCTION_PROMPT)


        print(prompt)
        
        raw = model.generate(
            prompt,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0,
            use_cache=False,
            max_input_tokens=args.max_input_tokens,
        )
        pred = extract_label_from_text(raw)
        result = {
            "index": idx,
            "prompt": query_prompt,
            "raw_output": raw,
            "predicted": pred,
            "gt": rec.get("gt", ""),
            "correct": pred == rec.get("gt", ""),
        }
        fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        fh.flush()
        results.append(result)

    fh.close()
    summary = evaluate_predictions(results)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
