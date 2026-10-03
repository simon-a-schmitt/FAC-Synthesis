"""Blackbox arm: generate --n synthetic examples for one domain via the OpenRouter API, with
ROUGE-L dedup against seeds + accepted pool.

Prompt: prompts/<domain>/generation.py (SYSTEM_PROMPT + BLACKBOX_TEMPLATE). Seeds:
benchmarks/<domain>/<seed set>/. Context drawing, dedup, waves and checkpoints: see
shared/generation.py. Output in blackbox/<domain>/output and blackbox/<domain>/log.

Usage:
    python run_generation.py --model deepseek --domain cti_vsp --seed-group 01 --n 400 \
        --rouge-threshold 0.7 --prefix cti_vsp_bb_deepseek_01 --max-concurrent-requests 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ARM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ARM_DIR.parent))

from shared.generation import add_generation_args, run_standalone_arm  # noqa: E402

ARM = "blackbox"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_generation_args(parser)
    parser.add_argument("--n", type=int, required=True, help="Number of ACCEPTED samples to generate.")
    parser.add_argument("--max-calls", type=int, default=None, help="Cap on API calls (default: 50 x --n).")
    return parser


if __name__ == "__main__":
    run_standalone_arm(build_arg_parser().parse_args(), ARM, ARM_DIR)
