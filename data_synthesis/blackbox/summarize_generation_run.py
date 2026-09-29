"""Summarize a completed blackbox generation run (all runs written under one --prefix) into a JSON file.

Requires <path>/output/<prefix>_accepted.json, <path>/output/<prefix>_rejected.json and
<path>/log/<prefix>_log.json (<path>/output/<prefix>_failed.json is read if present). Computes, in
total and per run_id:
  - the number of generated (= accepted + rouge_duplicate), accepted and rejected samples and the
    acceptance rate. Only "rouge_duplicate" counts as a rejection: a "target_reached" sample was
    never checked (the run's --n was already reached by an earlier slot of the same wave), so it
    and the call that produced it are reported separately and left out of the acceptance rate,
    the call count and the token usage,
  - token usage summed over those calls (failed calls returned no response, so carry no usage),
  - the provider/sampling-parameter verification, where the log has it,
and writes it to <path>/log/<prefix>_summary.json.

Also works on runs from older versions of run_generation.py: calls and samples are joined on
(run_id, wave_idx, slot_index), which every version has written, not on call_id/outcomes. Calls
without a logged sampling_verification are valid calls; they are fully counted and only reported
as "verification not logged".

Usage:
    python summarize_generation_run.py --path toxicity_detection --prefix toxicity_bb_reporting_test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent
RUN_GENERATION_SCRIPT = BASE_DIR / "run_generation.py"


def _load_run_generation_module():
    spec = importlib.util.spec_from_file_location("blackbox_run_generation", RUN_GENERATION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Must be registered before exec: its @dataclass definitions look themselves up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Path resolution and JSON I/O are shared with the generator.
bb = _load_run_generation_module()

REJECTION_REASON = "rouge_duplicate"
TARGET_REACHED_REASON = "target_reached"


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


def _count(values) -> dict:
    return dict(Counter(v if isinstance(v, str) else json.dumps(v) for v in values))


def verification_summary(call_records: list[dict]) -> dict:
    """Like bb.summarize_sampling_verification, but calls from before sampling_verification was
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
        "routed_provider_counts": _count((r.get("openrouter_response") or {}).get("provider") for r in call_records),
        "routed_model_counts": _count((r.get("openrouter_response") or {}).get("model") for r in call_records),
        "quantization_counts": _count(v.get("quantization") for v in logged),
        "requested_temperature_counts": _count(v.get("requested_temperature") for v in logged),
        "requested_top_p_counts": _count(v.get("requested_top_p") for v in logged),
    }


def run_stats(call_records: list[dict], accepted: list[dict], rejected: list[dict], failed: list[dict]) -> dict:
    """Stats for a set of call records and the samples they produced (target_reached excluded)."""
    rouge_rejected = [e for e in rejected if e.get("rejected_reason") == REJECTION_REASON]
    target_reached = [e for e in rejected if e.get("rejected_reason") == TARGET_REACHED_REASON]

    # A target_reached call is one whose every sample was rejected as target_reached (never
    # checked). Calls without any candidate stay regular calls.
    tr_call_keys = {e["_key"] for e in target_reached} - {e["_key"] for e in accepted + rouge_rejected}
    counted_calls = [r for r in call_records if r["_key"] not in tr_call_keys]
    tr_calls = [r for r in call_records if r["_key"] in tr_call_keys]

    n_accepted = len(accepted)
    n_rejected = len(rouge_rejected)
    n_generated = n_accepted + n_rejected
    return {
        "n_calls": len(counted_calls) + len(failed),
        "n_successful_calls": len(counted_calls),
        "n_failed_calls": len(failed),
        "n_generated": n_generated,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "acceptance_rate": round(n_accepted / n_generated, 4) if n_generated else None,
        "tokens": sum_usage(counted_calls),
        # Reported only, excluded from everything above.
        "target_reached": {
            "n_samples": len(target_reached),
            "n_calls": len(tr_calls),
            "tokens": sum_usage(tr_calls),
        },
        "rejected_by_reason": dict(Counter(e.get("rejected_reason") for e in rejected)),
        "sampling_verification": verification_summary(counted_calls),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize token usage and accepted/rejected counts of a completed generation run (by --prefix)."
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Domain subfolder (e.g. 'toxicity_detection'), either a name under blackbox/ or a path to it.",
    )
    parser.add_argument("--prefix", type=str, required=True, help="The --prefix the run was generated with.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    domain_dir = bb.resolve_domain_dir(args.path)
    output_dir = domain_dir / "output"
    log_dir = domain_dir / "log"
    files = {
        "accepted": output_dir / f"{args.prefix}_accepted.json",
        "rejected": output_dir / f"{args.prefix}_rejected.json",
        "failed": output_dir / f"{args.prefix}_failed.json",
        "log": log_dir / f"{args.prefix}_log.json",
    }
    missing = [str(files[key]) for key in ("accepted", "rejected", "log") if not files[key].exists()]
    if missing:
        raise SystemExit(f"Missing required file(s) for prefix {args.prefix!r}: {missing}")

    accepted = [e for e in bb.load_json_list(files["accepted"]) if e.get("type") == "synthetic"]
    rejected = bb.load_json_list(files["rejected"])
    failed = [e for e in bb.load_json_list(files["failed"]) if is_generation_failure(e)]
    log_runs = bb.load_json_dict(files["log"])["runs"]

    for entry in accepted + rejected + failed:
        entry["_key"] = slot_key(entry.get("run_id"), entry)

    runs = []
    all_calls: list[dict] = []
    for run in log_runs:
        run_id = run["run_id"]
        call_records = run.get("calls", [])
        for record in call_records:
            record["_key"] = slot_key(run_id, record)
        all_calls.extend(call_records)
        runs.append(
            {
                "run_id": run_id,
                "model_id": run.get("model_id"),
                "seed_group": run.get("seed_group"),
                "n_requested": run.get("n_requested"),
                "rouge_threshold": run.get("rouge_threshold"),
                "model_params": run.get("model_params"),
                "started_at": run.get("started_at"),
                "finished_at": run.get("finished_at"),
                **run_stats(
                    call_records,
                    [e for e in accepted if e.get("run_id") == run_id],
                    [e for e in rejected if e.get("run_id") == run_id],
                    [e for e in failed if e.get("run_id") == run_id],
                ),
            }
        )

    summary = {
        "prefix": args.prefix,
        "path": str(domain_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counting_rules": (
            f"Only '{REJECTION_REASON}' counts as a rejection. '{TARGET_REACHED_REASON}' samples and "
            "their calls are reported under 'target_reached' and excluded from n_calls, tokens, "
            "n_generated, acceptance_rate and sampling_verification. n_generated = n_accepted + n_rejected. "
            "Calls without a logged sampling_verification (older runs) are valid calls and fully counted."
        ),
        "files": {key: str(path) for key, path in files.items() if path.exists()},
        "totals": {"n_runs": len(runs), **run_stats(all_calls, accepted, rejected, failed)},
        "runs": runs,
    }

    output_path = log_dir / f"{args.prefix}_summary.json"
    bb.save_json(output_path, summary)

    totals = summary["totals"]
    tokens = totals["tokens"]
    tr = totals["target_reached"]
    verification = totals["sampling_verification"]
    print(f"Prefix {args.prefix!r}: {len(runs)} run(s), {totals['n_calls']} call(s) ({totals['n_failed_calls']} failed)")
    print(f"  generated {totals['n_generated']} = accepted {totals['n_accepted']} + rejected (ROUGE) {totals['n_rejected']} "
          f"-> acceptance rate {totals['acceptance_rate']}")
    print(f"  tokens: prompt {tokens['prompt_tokens']} + completion {tokens['completion_tokens']} "
          f"= total {tokens['total_tokens']} (cost {tokens['cost']})")
    print(f"  excluded target_reached: {tr['n_samples']} sample(s) from {tr['n_calls']} call(s), "
          f"{tr['tokens']['total_tokens']} token(s)")
    print(f"  sampling verification logged for {verification['n_verification_logged']}/{verification['n_calls']} call(s), "
          f"{verification['n_unverified']} unverified")
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
