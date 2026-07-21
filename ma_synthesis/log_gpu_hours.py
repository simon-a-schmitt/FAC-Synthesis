"""Update the cumulative GPU-hours / sample-count log for a ma_synthesis run.

Called once per SLURM job from ma_synthesis_job_nrh.sh. Uses an flock-guarded
read-modify-write so that concurrently finishing jobs (multiple submissions
belonging to the same generation run) don't clobber each other's updates.
Entries are keyed by --run-path (blackbox / feature_guided) and accumulate
until the user resets the log file by hand for a new, separate run.
"""

import argparse
import fcntl
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append job usage to the GPU-hours log")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--run-path", required=True, choices=["blackbox", "feature_guided"])
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_hours = args.elapsed_seconds * args.num_gpus / 3600.0

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = log_path.with_suffix(log_path.suffix + ".lock")

    with open(lock_path, "w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            if log_path.exists() and log_path.stat().st_size > 0:
                data = json.loads(log_path.read_text(encoding="utf-8"))
            else:
                data = {}

            entry = data.setdefault(
                args.run_path, {"gpu_hours": 0.0, "samples_generated": 0, "job_ids": []}
            )
            entry["gpu_hours"] = round(entry["gpu_hours"] + gpu_hours, 4)
            entry["samples_generated"] += args.samples
            entry["job_ids"].append(args.job_id)

            log_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)

    print(
        f"[INFO] Logged job {args.job_id} ({args.run_path}): "
        f"+{gpu_hours:.4f} GPU hours, +{args.samples} samples "
        f"-> totals: {entry['gpu_hours']:.4f} h, {entry['samples_generated']} samples",
        flush=True,
    )


if __name__ == "__main__":
    main()
