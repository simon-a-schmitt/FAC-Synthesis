"""Submits the missing stages of runs as SLURM job chains - idempotent: it reads the state off the file
system + submissions (experiments/status.py) and submits only stages that are neither done nor
pending/running. Calling it again is the "reconcile" step (no daemon).

Chains (sbatch --parsable, --dependency=afterok:<id> --kill-on-invalid-dep=yes):
    bb / fg / hybrid:   gen (packed) -> label_build (per run) -> ft_bench (per run)
    plain / icl / gold: ft_bench (per run), no dependency
    api:                api_bench_job.sh (per run, CPU), no dependency; recorded as job stage ft_bench
A downstream stage depends on its upstream job if that is submitted now or still pending/running;
on nothing if the upstream stage is done. A failed / timed-out stage counts as missing.

Packing: gen runs of the same GEN_RESOURCE in packs of GEN_PACK_SIZE (gpu) / GEN_PACK_SIZE_CPU (cpu),
one experiments/slurm/gen_job.sh job per pack. API throttle: at most GEN_MAX_LANES gen jobs at a time -
packs are assigned round-robin to job names gen_lane_<i>, each with --dependency=singleton.
Partition / time / gres of every job class come from experiments/config/cluster.env.

Submission order: group-major (seed set, group, benchmark, arm). Every submission is appended to
experiments/state/submissions.jsonl.

Usage:
    python experiments/submit.py [--bench B] [--arm A] [--setup S] [--seed-set K] [--group G] [--dry-run]
    python experiments/submit.py <run_id> [<run_id> ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from datetime import datetime

from runs import PROJECT_DIR, load_config
from status import (ACTIVE, add_filter_args, append_submission, collect, filters_of, run_order_key,
                    select_specs)

CLUSTER_ENV = os.environ.get("CLUSTER_ENV", str(PROJECT_DIR / "experiments" / "config" / "cluster.env"))
CLUSTER_KEYS = ("GEN_PARTITION", "GEN_TIME", "GEN_GRES", "GEN_PACK_SIZE", "GEN_CPU_PARTITION", "GEN_CPU_TIME",
                "GEN_PACK_SIZE_CPU", "GEN_MAX_LANES", "LABEL_BUILD_PARTITION", "LABEL_BUILD_TIME",
                "FT_BENCH_PARTITION", "FT_BENCH_TIME", "API_BENCH_PARTITION", "API_BENCH_TIME")
SLURM_DIR = "experiments/slurm"


def cluster_settings() -> dict[str, str]:
    script = f'source {shlex.quote(CLUSTER_ENV)} >/dev/null && ' + " && ".join(
        f'printf "%s=%s\\n" {k} "${k}"' for k in CLUSTER_KEYS)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    values = dict(line.split("=", 1) for line in out.splitlines())
    missing = [k for k in CLUSTER_KEYS if not values.get(k)]
    if missing:
        raise SystemExit(f"error: {CLUSTER_ENV} does not set {missing}.")
    return values


class Submitter:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.n_placeholder = 0
        self.n_gen_packs = 0  # round-robin over the gen lanes
        self.submitted: list[tuple[str, str, list[str]]] = []  # (job_id, stage, run_ids)

    def sbatch(self, stage: str, run_ids: list[str], options: list[str], script: str,
               depends_on: str | None = None, extra: dict | None = None) -> str:
        cmd = ["sbatch", "--parsable", *options]
        if depends_on:
            cmd += [f"--dependency=afterok:{depends_on}", "--kill-on-invalid-dep=yes"]
        cmd += [f"{SLURM_DIR}/{script}", *run_ids]
        if self.dry_run:
            self.n_placeholder += 1
            job_id = f"<{stage}#{self.n_placeholder}>"
            print(f"{job_id:16s} " + shlex.join(cmd))
        else:
            job_id = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True,
                                    check=True).stdout.strip().split(";")[0]
            print(f"{job_id:16s} " + shlex.join(cmd))
            now = datetime.now().isoformat(timespec="seconds")
            for run_id in run_ids:
                append_submission({"run_id": run_id, "stage": stage, "job_id": job_id, "submitted_at": now,
                                   "depends_on": depends_on, "pack": run_ids if stage == "gen" else None,
                                   **(extra or {}), "sbatch": shlex.join(cmd)})
        self.submitted.append((job_id, stage, run_ids))
        return job_id


def plan_and_submit(config: dict, rows: list[dict], cs: dict[str, str], sub: Submitter) -> None:
    def needs(row: dict, *stages: str) -> bool:
        """Some stage of the job is not done, and none of them is pending/running."""
        st = [row["status"][s] for s in stages if row["status"][s] != "-"]
        return bool(st) and not any(s in ACTIVE for s in st) and any(s != "done" for s in st)

    def active_job(row: dict, job_stage: str, *stages: str) -> str | None:
        if any(row["status"][s] in ACTIVE for s in stages):
            return row["jobs"][job_stage]["job_id"]
        return None

    # 1) gen: packs per resource, in run order; packs round-robin on lanes.
    gen_jobs: dict[str, str] = {}
    for resource in ("cpu", "gpu"):
        todo = [row for row in rows if row["status"]["gen"] != "-" and needs(row, "gen")
                and row["r"]["gen_resource"] == resource]
        size = int(cs["GEN_PACK_SIZE_CPU"] if resource == "cpu" else cs["GEN_PACK_SIZE"])
        for i in range(0, len(todo), size):
            pack = [row["r"]["run_id"] for row in todo[i:i + size]]
            lane = sub.n_gen_packs % int(cs["GEN_MAX_LANES"])
            sub.n_gen_packs += 1
            if resource == "gpu":
                opts = [f"--partition={cs['GEN_PARTITION']}", f"--time={cs['GEN_TIME']}", f"--gres={cs['GEN_GRES']}"]
            else:
                opts = [f"--partition={cs['GEN_CPU_PARTITION']}", f"--time={cs['GEN_CPU_TIME']}", "--gres=none"]
            opts += [f"--job-name=gen_lane_{lane}", "--dependency=singleton"]
            job_id = sub.sbatch("gen", pack, opts, "gen_job.sh", extra={"lane": lane})
            gen_jobs.update({run_id: job_id for run_id in pack})

    # 2) label_build and ft_bench per run.
    lb_opts = [f"--partition={cs['LABEL_BUILD_PARTITION']}", f"--time={cs['LABEL_BUILD_TIME']}"]
    fb_opts = [f"--partition={cs['FT_BENCH_PARTITION']}", f"--time={cs['FT_BENCH_TIME']}"]
    api_opts = [f"--partition={cs['API_BENCH_PARTITION']}", f"--time={cs['API_BENCH_TIME']}"]
    for row in rows:
        run_id, st = row["r"]["run_id"], row["status"]
        upstream = None
        if st["label_build"] != "-":
            gen_dep = gen_jobs.get(run_id) or active_job(row, "gen", "gen")
            if needs(row, "label_build"):
                upstream = sub.sbatch("label_build", [run_id], lb_opts, "label_build_job.sh", gen_dep)
            else:
                upstream = active_job(row, "label_build", "label_build")
        if needs(row, "ft", "bench"):
            if row["spec"].arm == "api":
                sub.sbatch("ft_bench", [run_id], api_opts, "api_bench_job.sh")
            else:
                sub.sbatch("ft_bench", [run_id], fb_opts, "ft_bench_job.sh", upstream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_filter_args(parser)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned sbatch commands only.")
    args = parser.parse_args()

    config = load_config()
    specs = sorted(select_specs(config, args.run_ids, **filters_of(args)), key=lambda s: run_order_key(config, s))
    rows = collect(config, specs)
    cs = cluster_settings()

    print(f"{len(rows)} run(s) selected" + (" (dry run, nothing is submitted)" if args.dry_run else "") + ":")
    for row in rows:
        open_ = {s: v for s, v in row["status"].items() if v not in ("done", "-")}
        print(f"  {row['r']['run_id']:55s} " + (", ".join(f"{s}={v}" for s, v in open_.items()) or "complete"))
    print()
    sub = Submitter(args.dry_run)
    plan_and_submit(config, rows, cs, sub)
    by_stage = {}
    for _, stage, run_ids in sub.submitted:
        by_stage.setdefault(stage, [0, 0])
        by_stage[stage][0] += 1
        by_stage[stage][1] += len(run_ids)
    summary = ", ".join(f"{stage}: {n_jobs} job(s) / {n_runs} run(s)" for stage, (n_jobs, n_runs) in by_stage.items())
    print(f"\n{'Planned' if args.dry_run else 'Submitted'}: {summary or 'nothing (all done or in flight)'}")


if __name__ == "__main__":
    main()
