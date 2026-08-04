"""Main entry point for ma_synthesis_v2.

Usage examples:
  # blackbox arm, generate 10 samples (uses default seed / gt-pool files)
  python run.py --path blackbox --n 10 --out output/blackbox.jsonl

  # blackbox arm with explicit seed / gt-pool files
  python run.py --path blackbox --n 10 \\
      --seeds data/cti_vsp_benchmark_seed_group_01.tsv \\
      --gt-pool data/cti_vsp_gt_pool.txt \\
      --out output/blackbox.jsonl
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
from ma_synthesis.log_token_usage import update_token_log
from ma_synthesis_v2.blackbox.step1 import load_gt_pool, load_seeds, run_step1 as blackbox_step1


MODEL_PATH_DEFAULT = Path(
    "/pfs/work9/workspace/scratch/ka_ai3967-master_thesis_exp/models/llama-3.1-70b"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"

SEEDS_DEFAULT = Path(__file__).parent / "data" / "cti_vsp_benchmark_seed_group_01.tsv"
GT_POOL_DEFAULT = Path(__file__).parent / "data" / "cti_vsp_gt_pool.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ma_synthesis_v2 data generation pipeline")
    parser.add_argument(
        "--path",
        choices=["blackbox"],
        required=True,
        help="Which synthesis arm to use",
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
        help="(blackbox only) Path to the seed TSV file. Defaults to data/cti_vsp_benchmark_seed_group_01.tsv",
    )
    parser.add_argument(
        "--gt-pool",
        type=str,
        default=None,
        help="(blackbox only) Path to the target-vector pool file. Defaults to data/cti_vsp_gt_pool.txt",
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
    gt_pool: list[str] = []
    if args.path == "blackbox":
        seeds_path = Path(args.seeds) if args.seeds else SEEDS_DEFAULT
        seeds = load_seeds(seeds_path)
        if not seeds:
            print("[ERROR] Seed file is empty or could not be parsed.", flush=True)
            sys.exit(1)
        print(f"[INFO] Loaded {len(seeds)} seeds from {seeds_path}", flush=True)

        gt_pool_path = Path(args.gt_pool) if args.gt_pool else GT_POOL_DEFAULT
        gt_pool = load_gt_pool(gt_pool_path)
        if not gt_pool:
            print("[ERROR] Target vector pool file is empty or could not be parsed.", flush=True)
            sys.exit(1)
        print(f"[INFO] Loaded {len(gt_pool)} target vectors from {gt_pool_path}", flush=True)

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
        for i in range(args.n):
            print(f"\n[INFO] === Sample {i + 1}/{args.n} ===", flush=True)

            # --- Step 1 (arm-specific) ---
            if args.path == "blackbox":
                description, meta, usage = blackbox_step1(model, seeds=seeds, gt_pool=gt_pool)
                meta = {"path": "blackbox", **meta}

            update_token_log(
                token_log_path, args.path, "step_1",
                usage["input_tokens"], usage["output_tokens"],
            )

            print(f"[Step 1 output]\n{description}\n", flush=True)

            record = {
                **meta,
                "sample_index": i,
                "description": description,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"[INFO] Sample {i + 1}/{args.n} written to {out_path}", flush=True)

    print(f"\n[INFO] Done. {args.n} samples written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
