"""Status of the labeling-benchmark runs (lb_runs.py):

  done        lbench.done carries the current lb_fingerprint
  waiting_ft  gold/bb/fg/hybrid run whose fine-tuning in experiments is not done yet
  pending / running / failed / timeout
              not done, and the latest submission (labeling_benchmark/state/submissions.jsonl,
              written by lb_submit.py) is in that SLURM state (sacct); a job that COMPLETED without
              the run being done counts as failed
  stale / missing
              not done, no active or failed submission; stale = done under another fingerprint

The column exp_bench shows whether the experiments benchmark of the run is done (its outputs are
evaluated by lb_collect.py); "-" for api runs. Separate from experiments/state - nothing there is
read or written except the experiments run outputs.

Usage:
    python benchmark_play_ground/labeling_benchmark/lb_status.py [--bench B] [--arm A] [--setup S] [--seed-set K] [--group G]
    python benchmark_play_ground/labeling_benchmark/lb_status.py <run_id> [<run_id> ...] [--summary-only]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime

from lb_runs import (ARMS, LB_DIR, add_filter_args, experiments_bench_done, filters_of, lb_state, load_configs,
                     resolve_full, select)

import status as exp_status  # noqa: E402  experiments/status.py (sacct query only)

SUBMISSIONS = LB_DIR / "state" / "submissions.jsonl"
ACTIVE = ("pending", "running")


def load_submissions() -> dict[str, dict]:
    """Latest submission per run_id."""
    latest: dict[str, dict] = {}
    if SUBMISSIONS.exists():
        for line in SUBMISSIONS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                latest[rec["run_id"]] = rec
    return latest


def append_submission(rec: dict) -> None:
    SUBMISSIONS.parent.mkdir(parents=True, exist_ok=True)
    with SUBMISSIONS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def collect(exp_config: dict, lb_config: dict, specs: list) -> list[dict]:
    """Per run: {"spec", "r", "file": lb_state, "status", "job", "exp_bench"}."""
    submissions = load_submissions()
    rows = []
    for spec in specs:
        r, exp_r = resolve_full(spec, exp_config, lb_config)
        rows.append({"spec": spec, "r": r, "file": lb_state(r, exp_r), "job": submissions.get(r["run_id"]),
                     "exp_bench": "-" if exp_r is None else ("done" if experiments_bench_done(exp_r) else "not done")})
    slurm = exp_status.slurm_states({row["job"]["job_id"] for row in rows if row["job"]})
    for row in rows:
        state = slurm.get(row["job"]["job_id"]) if row["job"] else None
        if row["file"] in ("done", "waiting_ft") or state is None:
            row["status"] = row["file"]
        elif state in ACTIVE + ("timeout", "failed"):
            row["status"] = state
        else:  # completed without the run being done
            row["status"] = "stale" if row["file"] == "stale" else "failed"
    return rows


def print_table(rows: list[dict]) -> None:
    width = max([len(row["r"]["run_id"]) for row in rows] + [6])
    print(f"{'run_id':{width}s}  {'lbench':11s}  exp_bench")
    for row in rows:
        print(f"{row['r']['run_id']:{width}s}  {row['status']:11s}  {row['exp_bench']}")


def print_summary(rows: list[dict], exp_config: dict) -> None:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["spec"].bench, row["spec"].arm)].append(row)
    print(f"\n{'bench':20s} {'arm':7s} {'runs':>5s} {'done':>5s} {'exp_done':>8s}  lbench states (not done)")
    for bench in exp_config["benchmarks"]:
        for arm in ARMS:
            g = groups.get((bench, arm))
            if not g:
                continue
            open_states = Counter(row["status"] for row in g if row["status"] != "done")
            print(f"{bench:20s} {arm:7s} {len(g):5d} {sum(row['status'] == 'done' for row in g):5d} "
                  f"{sum(row['exp_bench'] == 'done' for row in g):8d}  "
                  + ", ".join(f"{k}={n}" for k, n in sorted(open_states.items())))
    print(f"{'total':20s} {'':7s} {len(rows):5d} {sum(row['status'] == 'done' for row in rows):5d} "
          f"{sum(row['exp_bench'] == 'done' for row in rows):8d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_filter_args(parser)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    exp_config, lb_config = load_configs()
    rows = collect(exp_config, lb_config, select(exp_config, lb_config, args.run_ids, **filters_of(args)))
    if not args.summary_only:
        print_table(rows)
    print_summary(rows, exp_config)
    print(f"\n(as of {datetime.now().isoformat(timespec='seconds')}; submissions: {SUBMISSIONS})")


if __name__ == "__main__":
    main()
