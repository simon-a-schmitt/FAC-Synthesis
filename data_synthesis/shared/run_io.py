"""JSON I/O, run logs and checkpoint/resume handling shared by all generation arms and labeling."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data) -> None:
    """Atomic: written to <path>.tmp and renamed, so an interrupted write never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# SIGTERM (SLURM time limit / requeue of experiments/slurm/gen_job.sh)
# ---------------------------------------------------------------------------
# The on-disk state of a generation run is consistent only between two persist() calls: persist()
# writes accepted / rejected / discarded / failed and the checkpoint one after another, so a run
# killed in the middle of it could leave e.g. the accepted pool one wave ahead of its checkpoint.
# A SIGTERM outside persist() exits at once (in-flight calls of the current wave are lost, the files
# hold the last completed wave, --resume redoes that wave with the same slot seeds); inside persist()
# it is deferred until the files are complete.
_SIGTERM = {"defer": 0, "pending": False}


def _exit_on_sigterm() -> None:
    print("[signal] SIGTERM: exiting; outputs + checkpoint hold the last completed wave (continue with --resume).",
          flush=True)
    sys.stderr.flush()
    os._exit(128 + signal.SIGTERM)  # no interpreter shutdown: it would wait for the in-flight API calls


def _on_sigterm(signum, frame) -> None:
    if _SIGTERM["defer"]:
        if not _SIGTERM["pending"]:
            print("[signal] SIGTERM while writing outputs - exiting once they are complete.", flush=True)
        _SIGTERM["pending"] = True
        return
    _exit_on_sigterm()


def install_sigterm_handler() -> None:
    signal.signal(signal.SIGTERM, _on_sigterm)


@contextmanager
def sigterm_deferred():
    """Defers a SIGTERM until the block (writing a consistent set of output files) is done."""
    _SIGTERM["defer"] += 1
    try:
        yield
    finally:
        _SIGTERM["defer"] -= 1
        if not _SIGTERM["defer"] and _SIGTERM["pending"]:
            _exit_on_sigterm()


def ignore_sigterm_until_exit() -> None:
    """For the final section of a run (last persist, run log, checkpoint removal): it takes seconds
    and completes the run, so a SIGTERM arriving now is only noted and the run finishes normally."""
    _SIGTERM["defer"] += 1


def load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8").strip()
    return json.loads(content) if content else []


def load_json_dict(path: Path) -> dict | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    return json.loads(content) if content else None


def format_wall_clock_slurm(started_at: str, finished_at: str) -> str:
    """finished_at - started_at (ISO timestamps) as HH:MM:SS, SLURM's "Job Wall-clock time" format.
    Measured from inside the script, so it undercounts SLURM's job time (prologue, startup, ...)."""
    elapsed = round((datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds())
    hours, remainder = divmod(max(elapsed, 0), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _env_int(*keys: str) -> int | None:
    for key in keys:
        try:
            return int(os.environ[key])
        except (KeyError, ValueError):
            pass
    return None


def collect_hardware() -> dict:
    """Hardware of the current (SLURM) job, same fields as the fine-tuning log's "hardware" block
    (plus cpu_model). GPU fields are empty on nodes without nvidia-smi (e.g. the cpu partition)."""
    gpus, driver = [], None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        for line in out.strip().splitlines():
            name, mem, driver = (x.strip() for x in line.split(","))
            gpus.append({"name": name, "memory_total_mib": int(mem)})
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"[warn] nvidia-smi query failed: {exc}", file=sys.stderr)

    node_ram_gb, cpu_model = None, None
    try:
        with open("/proc/meminfo") as f:
            kb = int(next(line for line in f if line.startswith("MemTotal:")).split()[1])
        node_ram_gb = round(kb / 1024**2, 1)
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            cpu_model = next(line for line in f if line.startswith("model name")).split(":", 1)[1].strip()
    except Exception:
        pass

    mem_per_node_mb = _env_int("SLURM_MEM_PER_NODE")
    mem_per_cpu_mb = _env_int("SLURM_MEM_PER_CPU")
    cpus_on_node = _env_int("SLURM_CPUS_ON_NODE")
    if mem_per_node_mb is None and mem_per_cpu_mb and cpus_on_node:
        mem_per_node_mb = mem_per_cpu_mb * cpus_on_node

    t_start, t_end = _env_int("SLURM_JOB_START_TIME"), _env_int("SLURM_JOB_END_TIME")
    return {
        "node": os.uname().nodename,
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "num_nodes": _env_int("SLURM_JOB_NUM_NODES", "SLURM_NNODES"),
        "ntasks": _env_int("SLURM_NTASKS"),
        "num_gpus": len(gpus) or None,
        "gpus_requested_per_node": os.environ.get("SLURM_GPUS_ON_NODE") or os.environ.get("SLURM_GPUS_PER_NODE"),
        "gpu_model": gpus[0]["name"] if gpus else None,
        "gpus": gpus,
        "gpu_driver": driver,
        "cpu_model": cpu_model,
        "cpus_allocated": cpus_on_node or len(os.sched_getaffinity(0)),
        "mem_requested_gb": round(mem_per_node_mb / 1024, 1) if mem_per_node_mb else None,
        "node_ram_total_gb": node_ram_gb,
        "time_limit_min": (t_end - t_start) // 60 if t_start and t_end else None,
    }


def append_run_log(log_path: Path, prefix: str, path: str, run_entry: dict) -> None:
    """Adds run_entry to the log's "runs" (replacing an earlier entry of the same run_id, so a crash
    right after writing the log never double-counts a run on --resume) and recomputes the
    cumulative totals over all runs."""
    existing = load_json_dict(log_path) or {}
    runs = [run for run in existing.get("runs", []) if run.get("run_id") != run_entry["run_id"]]
    runs.append(run_entry)
    prompt_tokens = sum(run.get("prompt_tokens", 0) for run in runs)
    completion_tokens = sum(run.get("completion_tokens", 0) for run in runs)
    log = {
        "prefix": prefix,
        "path": path,
        "cumulative_prompt_tokens": prompt_tokens,
        "cumulative_completion_tokens": completion_tokens,
        "cumulative_total_tokens": prompt_tokens + completion_tokens,
    }
    sae_totals = [run["sae_total"]["gpu_seconds"] for run in runs if run.get("sae_total")]
    if sae_totals:
        log["cumulative_sae_gpu_seconds"] = round(sum(sae_totals), 6)
    log["runs"] = runs
    save_json(log_path, log)
    print(f"Wrote run log to {log_path}")


def save_checkpoint(path: Path, data: dict) -> None:
    save_json(path, {**data, "saved_at": utc_now()})


def load_checkpoint(
    path: Path, resume: bool, resolved_args: dict, strict_keys: tuple[str, ...], noted_keys: tuple[str, ...]
) -> dict | None:
    """The checkpoint to resume from (None for a fresh run). Refuses to start fresh over an existing
    checkpoint, to resume without one, or to resume with a changed strict arg (model, seed group,
    feature schedule, ...); a changed noted arg (targets, caps, concurrency) is only printed."""
    checkpoint = load_json_dict(path)
    if not resume:
        if checkpoint is not None:
            raise SystemExit(
                f"Found an incomplete checkpoint at {path} (run_id={checkpoint['run_id']}). Pass --resume to "
                "continue it, or delete it (and, for hybrid, the run's output files) to start a fresh run."
            )
        return None
    if checkpoint is None:
        raise SystemExit(f"--resume was given but no checkpoint found at {path}; nothing to resume.")
    old_args = checkpoint["resolved_args"]
    for key in strict_keys:
        if old_args.get(key) != resolved_args[key]:
            raise SystemExit(
                f"Checkpoint at {path} was created with {key}={old_args.get(key)!r}, but this invocation has "
                f"{key}={resolved_args[key]!r}. Use the original value, or delete the checkpoint to start fresh."
            )
    for key in noted_keys:
        if old_args.get(key) != resolved_args[key]:
            print(f"[resume] Note: {key} changed since the checkpoint was saved "
                  f"({old_args.get(key)!r} -> {resolved_args[key]!r}).")
    print(f"[resume] Continuing run {checkpoint['run_id']} (checkpoint saved at {checkpoint['saved_at']}).")
    return checkpoint
