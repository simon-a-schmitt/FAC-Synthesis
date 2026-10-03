"""Helpers shared by the summarize_*.py scripts of all arms and of labeling: the call/sample join
key, token usage and sampling-verification roll-ups over logged call records."""

from __future__ import annotations

import json
from collections import Counter


def format_seconds(seconds: float | None) -> str | None:
    """HH:MM:SS, as shared.run_io.format_wall_clock_slurm."""
    if seconds is None:
        return None
    hours, remainder = divmod(max(round(seconds), 0), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def slot_key(run_id: str | None, entry: dict) -> tuple:
    """Join key between a call record and the samples it produced; present in every file version."""
    return (run_id, entry.get("wave_idx"), entry.get("slot_index"))


def is_generation_failure(entry: dict) -> bool:
    """Generation failures always carry call_number + error; some _failed.json files also hold
    entries of another format (labeling step failures), which are no generation calls."""
    return "call_number" in entry and "error" in entry


def sum_usage(call_records: list[dict]) -> dict:
    """Token usage (and OpenRouter cost, where reported) summed over the given call records."""
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cached_prompt_tokens": 0,
        "cost": 0.0,
    }
    for record in call_records:
        usage = (record.get("openrouter_response") or {}).get("usage") or {}
        prompt = usage.get("prompt_tokens") or 0
        completion = usage.get("completion_tokens") or 0
        totals["prompt_tokens"] += prompt
        totals["completion_tokens"] += completion
        totals["total_tokens"] += usage.get("total_tokens") or (prompt + completion)
        totals["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        totals["cached_prompt_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        totals["cost"] += usage.get("cost") or 0.0
    totals["cost"] = round(totals["cost"], 10)
    return totals


def count_values(values) -> dict:
    return dict(Counter(v if isinstance(v, str) else json.dumps(v) for v in values))


def verification_summary(call_records: list[dict]) -> dict:
    """Like shared.openrouter.summarize_sampling_verification, but calls from before sampling_verification was
    logged count as "not logged" instead of as failures; the provider/model OpenRouter routed
    them to is taken from openrouter_response, which every version logged."""
    logged = [r["sampling_verification"] for r in call_records if r.get("sampling_verification")]
    n_unverified = sum(1 for v in logged if not v.get("verified"))
    return {
        "n_calls": len(call_records),
        "n_verification_logged": len(logged),
        "n_verification_not_logged": len(call_records) - len(logged),
        "n_unverified": n_unverified,
        "all_logged_verified": (n_unverified == 0) if logged else None,
        "routed_provider_counts": count_values((r.get("openrouter_response") or {}).get("provider") for r in call_records),
        "routed_model_counts": count_values((r.get("openrouter_response") or {}).get("model") for r in call_records),
        "quantization_counts": count_values(v.get("quantization") for v in logged),
        "requested_temperature_counts": count_values(v.get("requested_temperature") for v in logged),
        "requested_top_p_counts": count_values(v.get("requested_top_p") for v in logged),
    }
