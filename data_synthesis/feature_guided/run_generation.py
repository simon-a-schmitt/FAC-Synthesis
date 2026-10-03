"""Feature-guided arm: generate --n synthetic examples for one domain via the OpenRouter API, each
call targeted at one task-relevant SAE feature and every candidate verified with the local
Llama-3.1-8B + SAE.

Prompt: prompts/<domain>/generation.py (SYSTEM_PROMPT + FEATURE_GUIDED_TEMPLATE). Context drawing,
dedup, waves and checkpoints: shared/generation.py (identical to the blackbox arm). Features, SAE
check, seed coverage and feature schedule: shared/feature_guidance.py. Output in
feature_guided/<domain>/output and feature_guided/<domain>/log.

The run stops once --n candidates are accepted (or, keeping the checkpoint for --resume, at
--max-calls / when every relevant feature is exhausted).

Usage: see feature_guided_generation_job.sh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ARM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ARM_DIR.parent))

from shared.feature_guidance import add_feature_args  # noqa: E402
from shared.generation import add_generation_args, run_standalone_arm  # noqa: E402

ARM = "feature_guided"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_generation_args(parser)
    parser.add_argument("--n", type=int, required=True, help="Number of ACCEPTED samples to generate.")
    parser.add_argument("--max-calls", type=int, default=None, help="Cap on API calls (default: 50 x --n).")
    add_feature_args(parser)
    return parser


if __name__ == "__main__":
    run_standalone_arm(build_arg_parser().parse_args(), ARM, ARM_DIR, feature_guided=True)
