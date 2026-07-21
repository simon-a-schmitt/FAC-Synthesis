"""Update the cumulative Llama token-usage log for a ma_synthesis run.

Mirrors log_gpu_hours.py: an flock-guarded read-modify-write, keyed by
run_path (blackbox / feature_guided) and pipeline step (step_1..step_3).
Entries accumulate across separate job submissions belonging to the same
generation run; resetting the log for a new, separate run is done by hand.

`update_token_log` is meant to be imported directly by run.py (called once
per LLM call, right after each generate()), so usage is persisted even if
the job is later killed or times out. A CLI entry point is provided too,
for parity with log_gpu_hours.py and manual testing.
"""

import argparse
import fcntl
import json
from pathlib import Path


def update_token_log(
    log_file: Path,
    run_path: str,
    step: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path = log_file.with_suffix(log_file.suffix + ".lock")

    with open(lock_path, "w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            if log_file.exists() and log_file.stat().st_size > 0:
                data = json.loads(log_file.read_text(encoding="utf-8"))
            else:
                data = {}

            branch = data.setdefault(run_path, {})
            entry = branch.setdefault(step, {"input_tokens": 0, "output_tokens": 0})
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens

            log_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            return entry
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append token usage to the token-usage log")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--run-path", required=True, choices=["blackbox", "feature_guided"])
    parser.add_argument("--step", required=True, choices=["step_1", "step_2", "step_3"])
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    entry = update_token_log(
        Path(args.log_file), args.run_path, args.step, args.input_tokens, args.output_tokens
    )
    print(
        f"[INFO] Logged {args.run_path}/{args.step}: "
        f"+{args.input_tokens} input, +{args.output_tokens} output "
        f"-> totals: {entry['input_tokens']} input, {entry['output_tokens']} output",
        flush=True,
    )


if __name__ == "__main__":
    main()
