"""Submits the labeling-benchmark runs that are not done as SLURM jobs, one job per run - idempotent:
it reads the state off the run directories + its own submissions (lb_status.py) and submits only runs
whose status is missing / stale / failed / timeout. Calling it again is the "reconcile" step.

  plain / icl / gold / bb / fg / hybrid  slurm/lbench_job.sh (GPU_PARTITION, gres gpu:1)
  api                                    slurm/lbench_api_job.sh (CPU_PARTITION)
Partitions are read from experiments/config/cluster.env, time limits from config.yaml (slurm).
Runs whose fine-tuning in experiments is not done (waiting_ft) are skipped - submit again once it is;
there is no SLURM dependency on experiments jobs. Every submission is appended to
labeling_benchmark/state/submissions.jsonl (experiments/state is never touched).

Usage (from the repo root FAC-Synthesis/, benchmark env):
    python benchmark_play_ground/labeling_benchmark/lb_submit.py --arm fg [--bench B] [--setup S] [--seed-set K] [--group G] [--dry-run]
    python benchmark_play_ground/labeling_benchmark/lb_submit.py <run_id> [<run_id> ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from collections import Counter
from datetime import datetime

from lb_runs import LB_DIR, PROJECT_DIR, add_filter_args, filters_of, load_configs, select
from lb_status import ACTIVE, append_submission, collect

CLUSTER_ENV = os.environ.get("CLUSTER_ENV", str(PROJECT_DIR / "experiments" / "config" / "cluster.env"))
SLURM_DIR = LB_DIR.relative_to(PROJECT_DIR) / "slurm"
LOG_DIR = LB_DIR / "logs"
SUBMIT_STATES = ("missing", "stale", "failed", "timeout")


def partitions() -> dict[str, str]:
    script = (f"source {shlex.quote(CLUSTER_ENV)} >/dev/null && "
              'printf "GPU_PARTITION=%s\\nCPU_PARTITION=%s\\n" "$GPU_PARTITION" "$CPU_PARTITION"')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    values = dict(line.split("=", 1) for line in out.splitlines())
    if not all(values.values()):
        raise SystemExit(f"error: {CLUSTER_ENV} does not set GPU_PARTITION / CPU_PARTITION.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_filter_args(parser)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned sbatch commands only.")
    args = parser.parse_args()

    exp_config, lb_config = load_configs()
    rows = collect(exp_config, lb_config, select(exp_config, lb_config, args.run_ids, **filters_of(args)))
    parts = partitions()
    times = lb_config["slurm"]
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    counts = Counter(row["status"] for row in rows)
    print(f"{len(rows)} run(s) selected: " + ", ".join(f"{k}={n}" for k, n in sorted(counts.items()))
          + (" (dry run, nothing is submitted)" if args.dry_run else ""))
    n_submitted = 0
    for row in rows:
        r = row["r"]
        if row["status"] not in SUBMIT_STATES:
            if row["status"] == "waiting_ft" or row["status"] in ACTIVE:
                print(f"  skip {r['run_id']} ({row['status']})")
            continue
        if r["resource"] == "gpu":
            opts = [f"--partition={parts['GPU_PARTITION']}", f"--time={times['gpu_time']}", "--gres=gpu:1"]
            script = SLURM_DIR / "lbench_job.sh"
        else:
            opts = [f"--partition={parts['CPU_PARTITION']}", f"--time={times['cpu_time']}"]
            script = SLURM_DIR / "lbench_api_job.sh"
        cmd = ["sbatch", "--parsable", *opts, str(script), r["run_id"]]
        if args.dry_run:
            print(f"  {'<dry-run>':12s} {shlex.join(cmd)}")
        else:
            job_id = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True,
                                    check=True).stdout.strip().split(";")[0]
            print(f"  {job_id:12s} {shlex.join(cmd)}")
            append_submission({"run_id": r["run_id"], "job_id": job_id, "previous_status": row["status"],
                               "submitted_at": datetime.now().isoformat(timespec="seconds"), "sbatch": shlex.join(cmd)})
        n_submitted += 1
    print(f"\n{'Planned' if args.dry_run else 'Submitted'}: {n_submitted} job(s)")


if __name__ == "__main__":
    main()
