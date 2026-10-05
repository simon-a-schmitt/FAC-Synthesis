"""Merges one executed stage into experiments/runs/<run_id>/run_meta.json (never overwrites
other stages): {slurm_job_id, node, gpu, started_at, finished_at, wall_seconds, git_commit,
git_dirty} under "stages"/<stage>, plus - once - the sha256 of TEST_TSV and SEED_FILE under
"inputs". A later, different hash of an input keeps the recorded one (it belongs to the results
already there) and prints a warning. "gpu" is left out when no GPU is visible (CPU jobs).
--extra-json adds stage-specific fields (e.g. n_majority_fallback for label_build, pack for gen).

Usage (from the job scripts, after a stage has finished successfully):
    python experiments/run_meta.py <run_id> --stage ft --started-at 2026-10-05T09:44:04+02:00 \
        --finished-at 2026-10-05T09:49:24+02:00 --wall-seconds 320
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from runs import PROJECT_DIR, load_config, parse_run_id, resolve


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> tuple[str, bool]:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(PROJECT_DIR), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def gpu_name() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.splitlines()[0] if out else None


def merge_stage(r: dict, stage: str, started_at: str, finished_at: str, wall_seconds: int,
                extra: dict | None = None) -> Path:
    path = Path(r["run_meta_json"])
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    meta["run_id"] = r["run_id"]

    for key, p in (("test_tsv", r.get("test_tsv")), ("seed_file", r.get("seed_file"))):
        if not p:
            continue
        entry = {"path": p, "sha256": sha256(p)}
        old = meta.setdefault("inputs", {}).setdefault(key, entry)
        if old["sha256"] != entry["sha256"]:
            print(f"WARNUNG: sha256 von {p} weicht von run_meta.json ab ({old['sha256']} -> {entry['sha256']}).")

    commit, dirty = git_state()
    record = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.uname().nodename,
        "gpu": gpu_name(),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": wall_seconds,
        "git_commit": commit,
        "git_dirty": dirty,
    }
    if record["gpu"] is None:
        del record["gpu"]
    record.update(extra or {})
    meta.setdefault("stages", {})[stage] = record

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_id")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--started-at", required=True)
    parser.add_argument("--finished-at", required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--extra-json", type=json.loads, default=None,
                        help='Additional fields of the stage record, e.g. \'{"n_majority_fallback": 0}\'.')
    args = parser.parse_args()

    try:
        r = resolve(parse_run_id(args.run_id), load_config())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    path = merge_stage(r, args.stage, args.started_at, args.finished_at, args.wall_seconds, args.extra_json)
    print(f"[INFO] run_meta.json aktualisiert (stage {args.stage}): {path}")


if __name__ == "__main__":
    main()
