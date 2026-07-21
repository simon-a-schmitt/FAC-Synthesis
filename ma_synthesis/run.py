"""Main entry point for ma_synthesis.

Usage examples:
  # blackbox path, generate 10 samples (uses default seed file)
  python run.py --path blackbox --n 10 --out output/blackbox.jsonl

  # blackbox path with explicit seed file
  python run.py --path blackbox --n 10 --seeds data/seeds/prompts_seed_casehold.json --out output/blackbox.jsonl

  # feature-guided path, generate 5 samples from a feature file
  python run.py --path feature_guided --n 5 --features data/features.jsonl --out output/feature_guided.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.model_wrapper import LocalModel
from ma_synthesis.blackbox.step1 import run_step1 as blackbox_step1
from ma_synthesis.common.pipeline import parse_step1_context, run_pipeline
from ma_synthesis.feature_guided.step1 import run_step1 as feature_guided_step1
from ma_synthesis.log_token_usage import update_token_log


MODEL_PATH_DEFAULT = Path(
    "/pfs/work9/workspace/scratch/ka_ai3967-master_thesis_exp/models/llama-3.1-70b"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"


SEEDS_DEFAULT = Path(__file__).parent / "data" / "seeds" / "prompts_seed_casehold.json"


def load_seeds(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_features(path: Path) -> list[dict]:
    features = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                features.append(json.loads(line))
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ma_synthesis data generation pipeline")
    parser.add_argument(
        "--path",
        choices=["blackbox", "feature_guided"],
        required=True,
        help="Which synthesis path to use",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/samples.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="(blackbox only) Path to JSON seed file. Defaults to data/seeds/prompts_seed_casehold.json",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="(feature_guided only) Path to JSONL file with SAE feature data",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODEL_PATH_DEFAULT),
        help="Path to the local model directory (e.g. $WS_PATH/models/llama-3.1-70b)",
    )
    parser.add_argument(
        "--token-log",
        type=str,
        default=None,
        help=(
            "Path to the token-usage log file, accumulated across job submissions "
            "and keyed by --path/step. Defaults to token_usage_log.json in "
            "$SLURM_SUBMIT_DIR (or the current directory outside SLURM)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seeds: list[dict] = []
    if args.path == "blackbox":
        seeds_path = Path(args.seeds) if args.seeds else SEEDS_DEFAULT
        seeds = load_seeds(seeds_path)
        if not seeds:
            print("[ERROR] Seed file is empty or could not be parsed.", flush=True)
            sys.exit(1)
        if len(seeds) < 2:
            print("[ERROR] Blackbox path needs at least 2 seeds to sample from.", flush=True)
            sys.exit(1)
        print(f"[INFO] Loaded {len(seeds)} seeds from {seeds_path}", flush=True)

    features: list[dict] = []
    if args.path == "feature_guided":
        if args.features is None:
            print("[ERROR] --features is required for the feature_guided path.", flush=True)
            sys.exit(1)
        features = load_features(Path(args.features))
        if not features:
            print("[ERROR] Feature file is empty or could not be parsed.", flush=True)
            sys.exit(1)
        print(f"[INFO] Loaded {len(features)} features from {args.features}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.token_log:
        token_log_path = Path(args.token_log)
    else:
        token_log_path = Path(os.environ.get("SLURM_SUBMIT_DIR", Path.cwd())) / "token_usage_log.json"

    print(f"[INFO] Loading model from {args.model_path} ...", flush=True)
    model = LocalModel(model_path=args.model_path, device=DEVICE, dtype=DTYPE)
    model.load()
    print("[INFO] Model loaded.", flush=True)

    with open(out_path, "a", encoding="utf-8") as out_f:
        accepted = 0
        attempt = 0
        while accepted < args.n:
            attempt += 1
            print(f"\n[INFO] === Attempt {attempt} (accepted {accepted}/{args.n}) ===", flush=True)

            # --- Step 1 (path-specific) ---
            if args.path == "blackbox":
                step1_output, step1_meta, step1_usage = blackbox_step1(model, seeds=seeds)
                meta = {"path": "blackbox", **step1_meta}
            else:
                feature = features[(attempt - 1) % len(features)]
                feature_description = feature.get("description", "")
                feature_text_spans = feature.get("text_spans", "")
                step1_output, step1_usage = feature_guided_step1(
                    model,
                    feature_description=feature_description,
                    feature_text_spans=feature_text_spans,
                )
                meta = {
                    "path": "feature_guided",
                    "feature_id": feature.get("feature_id"),
                }

            update_token_log(
                token_log_path, args.path, "step_1",
                step1_usage["input_tokens"], step1_usage["output_tokens"],
            )

            print(f"[Step 1 output]\n{step1_output}\n", flush=True)

            # --- Parse/validate step 1 context ---
            step1_output, accept = parse_step1_context(step1_output)
            if not accept:
                reject_record = {**meta, "accept": False}
                out_f.write(json.dumps(reject_record, ensure_ascii=False) + "\n")
                out_f.flush()
                print("[INFO] Rejected: no holding tag found.", flush=True)
                continue

            # --- Steps 2–4 (common pipeline) ---
            outputs = run_pipeline(model, step1_output)

            for step_name, usage in outputs["token_usage"].items():
                update_token_log(
                    token_log_path, args.path, step_name,
                    usage["input_tokens"], usage["output_tokens"],
                )

            record = {
                **meta,
                "sample_index": accepted,
                "final_question": outputs["final_question"],
                "final_label": outputs["final_label"],
                "accept": True,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            accepted += 1
            print(f"[INFO] Sample {accepted}/{args.n} accepted and written to {out_path}", flush=True)

    print(f"\n[INFO] Done. {args.n} accepted samples written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
