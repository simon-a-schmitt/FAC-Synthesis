"""Summarize a completed blackbox generation run (all runs written under one --prefix) into a JSON file.

Requires blackbox/<domain>/output/<prefix>_accepted.json, blackbox/<domain>/output/<prefix>_rejected.json and
blackbox/<domain>/log/<prefix>_log.json (blackbox/<domain>/output/<prefix>_discarded.json and _failed.json are read
if present). Computes, in total and per run_id:
  - the number of generated (= accepted + rouge_duplicate), accepted and rejected samples and the
    acceptance rate accepted / (accepted + rejected). Only "rouge_duplicate" counts as a rejection:
    a discarded sample (reason "target_reached") was never checked (the run's --n was already
    reached by an earlier slot of the same wave), so it and the call that produced it are reported
    separately under "discarded" and left out of the acceptance rate, the call count and the token
    usage. Older runs wrote discarded samples to _rejected.json (rejected_reason "target_reached");
    they are reclassified as discarded on reading (n_reclassified_from_rejected), the files are
    left unchanged,
  - token usage summed over those calls (failed calls returned no response, so carry no usage),
  - the provider/sampling-parameter verification, where the log has it,
and writes it to blackbox/<domain>/log/<prefix>_summary.json.

Also works on runs from older versions of the generator: calls and samples are joined on
(run_id, wave_idx, slot_index), which every version has written, not on call_id/outcomes. Calls
without a logged sampling_verification are valid calls; they are fully counted and only reported
as "verification not logged".

Usage:
    python summarize_generation_run.py --domain toxicity_detection --prefix toxicity_bb_reporting_test
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ARM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ARM_DIR.parent))

from shared.benchmarks import DOMAINS  # noqa: E402
from shared.run_io import load_json_dict, load_json_list, save_json, utc_now  # noqa: E402
from shared.summary import (  # noqa: E402
    DISCARD_REASON,
    is_generation_failure,
    slot_key,
    split_discarded,
    sum_usage,
    verification_summary,
)

REJECTION_REASON = "rouge_duplicate"


def discarded_block(discarded: list[dict], discarded_calls: list[dict]) -> dict:
    """Reported only, excluded from the call count, tokens and samples."""
    return {
        "reason": DISCARD_REASON,
        "n_samples": len(discarded),
        "n_reclassified_from_rejected": sum(1 for e in discarded if e.get("_reclassified")),
        "n_calls": len(discarded_calls),
        "tokens": sum_usage(discarded_calls),
    }


def run_stats(call_records: list[dict], accepted: list[dict], rejected: list[dict], discarded: list[dict],
              failed: list[dict]) -> dict:
    """Stats for a set of call records and the samples they produced (discarded reported separately)."""
    rouge_rejected = [e for e in rejected if e.get("rejected_reason") == REJECTION_REASON]

    # A discarded call is one whose every sample was discarded (never checked). Calls without any
    # candidate stay regular calls.
    dc_call_keys = {e["_key"] for e in discarded} - {e["_key"] for e in accepted + rouge_rejected}
    counted_calls = [r for r in call_records if r["_key"] not in dc_call_keys]
    dc_calls = [r for r in call_records if r["_key"] in dc_call_keys]

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
        "discarded": discarded_block(discarded, dc_calls),
        "rejected_by_reason": dict(Counter(e.get("rejected_reason") for e in rejected)),
        "sampling_verification": verification_summary(counted_calls),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize token usage and accepted/rejected counts of a completed generation run (by --prefix)."
    )
    parser.add_argument("--domain", type=str, required=True, choices=DOMAINS)
    parser.add_argument("--prefix", type=str, required=True, help="The --prefix the run was generated with.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    domain_dir = ARM_DIR / args.domain
    output_dir = domain_dir / "output"
    log_dir = domain_dir / "log"
    files = {
        "accepted": output_dir / f"{args.prefix}_accepted.json",
        "rejected": output_dir / f"{args.prefix}_rejected.json",
        "discarded": output_dir / f"{args.prefix}_discarded.json",
        "failed": output_dir / f"{args.prefix}_failed.json",
        "log": log_dir / f"{args.prefix}_log.json",
    }
    missing = [str(files[key]) for key in ("accepted", "rejected", "log") if not files[key].exists()]
    if missing:
        raise SystemExit(f"Missing required file(s) for prefix {args.prefix!r}: {missing}")

    accepted = [e for e in load_json_list(files["accepted"]) if e.get("type") == "synthetic"]
    rejected, discarded = split_discarded(load_json_list(files["rejected"]), load_json_list(files["discarded"]))
    failed = [e for e in load_json_list(files["failed"]) if is_generation_failure(e)]
    log_runs = load_json_dict(files["log"])["runs"]

    for entry in accepted + rejected + discarded + failed:
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
                "slurm_job_id": run.get("slurm_job_id"),
                "hardware": run.get("hardware"),
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
                    [e for e in discarded if e.get("run_id") == run_id],
                    [e for e in failed if e.get("run_id") == run_id],
                ),
            }
        )

    summary = {
        "prefix": args.prefix,
        "path": str(domain_dir),
        "generated_at": utc_now(),
        "counting_rules": (
            f"Only '{REJECTION_REASON}' counts as a rejection. Discarded samples (reason '{DISCARD_REASON}', "
            "never checked; in older runs stored in _rejected.json and reclassified on reading) and their calls "
            "are reported under 'discarded' and excluded from n_calls, tokens, n_generated, acceptance_rate and "
            "sampling_verification. n_generated = n_accepted + n_rejected; acceptance_rate = n_accepted / n_generated. "
            "Calls without a logged sampling_verification (older runs) are valid calls and fully counted."
        ),
        "files": {key: str(path) for key, path in files.items() if path.exists()},
        "totals": {"n_runs": len(runs), **run_stats(all_calls, accepted, rejected, discarded, failed)},
        "runs": runs,
    }

    output_path = log_dir / f"{args.prefix}_summary.json"
    save_json(output_path, summary)

    totals = summary["totals"]
    tokens = totals["tokens"]
    dc = totals["discarded"]
    verification = totals["sampling_verification"]
    print(f"Prefix {args.prefix!r}: {len(runs)} run(s), {totals['n_calls']} call(s) ({totals['n_failed_calls']} failed)")
    print(f"  generated {totals['n_generated']} = accepted {totals['n_accepted']} + rejected (ROUGE) {totals['n_rejected']} "
          f"-> acceptance rate {totals['acceptance_rate']}")
    print(f"  tokens: prompt {tokens['prompt_tokens']} + completion {tokens['completion_tokens']} "
          f"= total {tokens['total_tokens']} (cost {tokens['cost']})")
    print(f"  discarded ({dc['reason']}, excluded above): {dc['n_samples']} sample(s) "
          f"[{dc['n_reclassified_from_rejected']} reclassified from _rejected.json] from {dc['n_calls']} call(s), "
          f"{dc['tokens']['total_tokens']} token(s)")
    print(f"  sampling verification logged for {verification['n_verification_logged']}/{verification['n_calls']} call(s), "
          f"{verification['n_unverified']} unverified")
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
