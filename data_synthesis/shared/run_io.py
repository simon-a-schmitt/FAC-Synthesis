"""JSON I/O, run logs and checkpoint/resume handling shared by all generation arms and labeling."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
