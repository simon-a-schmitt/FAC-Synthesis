"""Status of runs: per stage done / pending / running / failed / timeout / stale / missing, or "-" if
the stage does not apply to the arm.

  done                  experiments/stages.py (the same checks as the job scripts)
  pending / running /   not done, and the latest submission of the stage's job
  failed / timeout      (experiments/state/submissions.jsonl, written by experiments/submit.py) is in
                        that SLURM state (sacct). A job that COMPLETED without the stage being done
                        counts as failed; a job cancelled because its dependency failed, too.
  stale / missing       not done, no active or failed submission; stale = a done marker written
                        under another config (fingerprint, see runs.py)

Job stages: gen, label_build, ft_bench (one job for the ft and bench columns; api runs: api_bench_job.sh).

Usage:
    python experiments/status.py [--bench B] [--arm A] [--setup S] [--seed-set K] [--group G]
    python experiments/status.py <run_id> [<run_id> ...]
    python experiments/status.py ... --summary-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime

from runs import ARMS, EXPERIMENTS_DIR, RunSpec, expand, load_config, parse_run_id, resolve
from stages import STAGE_NAMES, run_states

SUBMISSIONS = EXPERIMENTS_DIR / "state" / "submissions.jsonl"
JOB_STAGE = {"gen": "gen", "label_build": "label_build", "ft": "ft_bench", "bench": "ft_bench"}
ACTIVE = ("pending", "running")
SLURM_STATE = {
    "PENDING": "pending", "REQUEUED": "pending", "REQUEUE_HOLD": "pending", "REQUEUE_FED": "pending",
    "RUNNING": "running", "COMPLETING": "running", "CONFIGURING": "running", "SUSPENDED": "running",
    "RESIZING": "running", "STAGE_OUT": "running", "SIGNALING": "running",
    "TIMEOUT": "timeout",
    "COMPLETED": "completed",  # -> failed if the stage is still not done
}


# ---------------------------------------------------------------------------
# Submissions + SLURM
# ---------------------------------------------------------------------------

def load_submissions() -> dict[tuple[str, str], dict]:
    """Latest submission per (run_id, job stage)."""
    latest: dict[tuple[str, str], dict] = {}
    if SUBMISSIONS.exists():
        for line in SUBMISSIONS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                latest[(rec["run_id"], rec["stage"])] = rec
    return latest


def append_submission(rec: dict) -> None:
    SUBMISSIONS.parent.mkdir(parents=True, exist_ok=True)
    with SUBMISSIONS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def slurm_states(job_ids: set[str]) -> dict[str, str]:
    """job_id -> pending / running / timeout / completed / failed (sacct; unknown jobs are left out)."""
    if not job_ids:
        return {}
    out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(sorted(job_ids)), "-o", "JobID,State"],
                         capture_output=True, text=True, check=True).stdout
    states = {}
    for line in out.splitlines():
        job_id, _, state = line.partition("|")
        state = state.split()[0] if state.strip() else ""
        if job_id in job_ids and state:
            states[job_id] = SLURM_STATE.get(state, "failed")  # FAILED, CANCELLED, OUT_OF_MEMORY, NODE_FAIL, ...
    return states


# ---------------------------------------------------------------------------
# Status of runs
# ---------------------------------------------------------------------------

def select_specs(config: dict, run_ids: list[str] | None = None, **filters) -> list[RunSpec]:
    if run_ids:
        return [parse_run_id(r) for r in run_ids]
    return [s for s in expand(config) if all(v is None or getattr(s, k) == v for k, v in filters.items())]


def collect(config: dict, specs: list[RunSpec]) -> list[dict]:
    """Per run: {"spec", "r", "files": stage -> stages.py state, "status": stage -> status,
    "jobs": job stage -> latest submission}."""
    n_total = config["global"]["n_total"]
    submissions = load_submissions()
    rows = []
    for spec in specs:
        r = resolve(spec, config)
        files = run_states(r, n_total)
        jobs = {js: submissions[(r["run_id"], js)] for js in set(JOB_STAGE.values()) if (r["run_id"], js) in submissions}
        rows.append({"spec": spec, "r": r, "files": files, "jobs": jobs})
    slurm = slurm_states({job["job_id"] for row in rows for job in row["jobs"].values()})
    for row in rows:
        status = {}
        for stage, file_state in row["files"].items():
            if file_state in ("done", "-"):
                status[stage] = file_state
                continue
            job = row["jobs"].get(JOB_STAGE[stage])
            slurm_state = slurm.get(job["job_id"]) if job else None
            if slurm_state in ACTIVE + ("timeout", "failed"):
                status[stage] = slurm_state
            elif slurm_state == "completed":
                # Finished without the stage being done: failed - unless the stage is merely stale,
                # i.e. it was done but its marker no longer matches the current config.
                status[stage] = "stale" if file_state == "stale" else "failed"
            else:
                status[stage] = file_state  # missing / stale
        row["status"] = status
    return rows


def run_order_key(config: dict, spec: RunSpec) -> tuple:
    """Group-major: seed set, group, benchmark, arm (the submission order of submit.py)."""
    seed_sets, benches = list(config["global"]["seed_sets"]), list(config["benchmarks"])
    return (seed_sets.index(spec.seed_set) if spec.seed_set else -1, spec.group or "", benches.index(spec.bench),
            ARMS.index(spec.arm), spec.setup or "", spec.n_fg or 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_ids", nargs="*", help="Explicit run_ids (instead of filters).")
    parser.add_argument("--bench")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--setup")
    parser.add_argument("--seed-set")
    parser.add_argument("--group")


def filters_of(args: argparse.Namespace) -> dict:
    return {"bench": args.bench, "arm": args.arm, "setup": args.setup, "seed_set": args.seed_set, "group": args.group}


def print_table(rows: list[dict]) -> None:
    width = max([len(row["r"]["run_id"]) for row in rows] + [6])
    print(f"{'run_id':{width}s}  " + "  ".join(f"{s:11s}" for s in STAGE_NAMES))
    for row in rows:
        print(f"{row['r']['run_id']:{width}s}  " + "  ".join(f"{row['status'][s]:11s}" for s in STAGE_NAMES))


def print_summary(rows: list[dict], config: dict) -> None:
    """Per benchmark x arm: runs complete (every applicable stage done) and stage states."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["spec"].bench, row["spec"].arm)].append(row)
    print(f"\n{'bench':20s} {'arm':7s} {'runs':>5s} {'complete':>9s}  stage states (not done)")
    for bench in config["benchmarks"]:
        for arm in ARMS:
            g = groups.get((bench, arm))
            if not g:
                continue
            complete = sum(all(v in ("done", "-") for v in row["status"].values()) for row in g)
            open_states = Counter(f"{stage}:{v}" for row in g for stage, v in row["status"].items()
                                  if v not in ("done", "-"))
            detail = ", ".join(f"{k}={n}" for k, n in sorted(open_states.items()))
            print(f"{bench:20s} {arm:7s} {len(g):5d} {complete:9d}  {detail}")
    total_complete = sum(all(v in ("done", "-") for v in row["status"].values()) for row in rows)
    print(f"{'total':20s} {'':7s} {len(rows):5d} {total_complete:9d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_filter_args(parser)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    config = load_config()
    specs = sorted(select_specs(config, args.run_ids, **filters_of(args)), key=lambda s: run_order_key(config, s))
    rows = collect(config, specs)
    if not args.summary_only:
        print_table(rows)
    print_summary(rows, config)
    print(f"\n(as of {datetime.now().isoformat(timespec='seconds')}; submissions: {SUBMISSIONS})")


if __name__ == "__main__":
    main()
