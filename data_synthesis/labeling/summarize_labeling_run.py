"""Summarize the labeling run(s) written under one prefix into a JSON file.

Requires labeling/<domain>/log/<prefix>_log.json, as written by run_labeling.py (prefix = the
labeled input's name without "_accepted", i.e. the name of labeling/<domain>/<prefix>.tsv).
Computes, in total and per run_id:
  - the number of API calls (initial + retry rounds), split into labeled / unparsed / failed
    (api_error), and per round,
  - the number of synthetic samples labeled by the API, reused from an earlier run's TSV, and
    assigned the majority label after every retry was used up,
  - token usage (and OpenRouter cost, where reported) summed over every call that returned a
    response - unparsed responses included, since they consumed tokens too (failed calls returned
    no response, so carry no usage) - plus the share of it spent on unparsed responses,
  - the finish_reason counts (a "length" means a label line was cut off by max_tokens),
  - the provider/sampling-parameter verification,
and writes it to labeling/<domain>/log/<prefix>_summary.json.

Unlike the generation arms there is no accepted/rejected split: every synthetic sample ends up
labeled (by the API or by the majority fallback).

Usage:
    python summarize_labeling_run.py --domain toxicity_detection --prefix toxicity_fg_llama_d0_6_t0_0
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))

# Call records use the same "openrouter_response"/"sampling_verification" keys as generation.
from shared.benchmarks import DOMAINS  # noqa: E402
from shared.run_io import load_json_dict, save_json, utc_now  # noqa: E402
from shared.summary import count_values, sum_usage, verification_summary  # noqa: E402


def call_stats(call_records: list[dict]) -> dict:
    """Stats over a set of call records (one per API call, any round)."""
    responded = [c for c in call_records if c.get("status") != "api_error"]
    unparsed = [c for c in call_records if c.get("status") == "unparsed"]
    labeled = [c for c in call_records if c.get("status") == "labeled"]
    tokens = sum_usage(responded)
    return {
        "n_calls": len(call_records),
        "n_labeled_calls": len(labeled),
        "n_unparsed_calls": len(unparsed),
        "n_failed_calls": len(call_records) - len(responded),
        "calls_by_round": {
            round_label: dict(Counter(c.get("status") for c in call_records if c.get("round") == round_label))
            for round_label in dict.fromkeys(c.get("round") for c in call_records)
        },
        "tokens": tokens,
        "tokens_unparsed": sum_usage(unparsed),
        "avg_total_tokens_per_labeled_sample": round(tokens["total_tokens"] / len(labeled), 2) if labeled else None,
        "finish_reason_counts": count_values((c.get("openrouter_response") or {}).get("finish_reason") for c in responded),
        "sampling_verification": verification_summary(responded),
    }


def sample_stats(runs: list[dict]) -> dict:
    return {
        "n_labeled_by_api": sum(r.get("n_to_label", 0) - r.get("n_majority_fallback", 0) for r in runs),
        "n_reused": sum(r.get("n_reused", 0) for r in runs),
        "n_majority_fallback": sum(r.get("n_majority_fallback", 0) for r in runs),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize token usage and sampling verification of the labeling run(s) of one prefix."
    )
    parser.add_argument("--domain", type=str, required=True, choices=DOMAINS)
    parser.add_argument(
        "--prefix", type=str, required=True,
        help="Name of the labeled output (labeling/<domain>/<prefix>.tsv), i.e. the input JSON's name without '_accepted'.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    domain_dir = BASE_DIR / args.domain
    log_dir = domain_dir / "log"
    log_path = log_dir / f"{args.prefix}_log.json"
    log = load_json_dict(log_path)
    if log is None:
        raise SystemExit(f"Missing or empty log file for prefix {args.prefix!r}: {log_path}")
    log_runs = log.get("runs", [])

    runs = []
    all_calls: list[dict] = []
    for run in log_runs:
        call_records = run.get("calls", [])
        all_calls.extend(call_records)
        runs.append(
            {
                "run_id": run["run_id"],
                "slurm_job_id": run.get("slurm_job_id"),
                "hardware": run.get("hardware"),
                "source": run.get("source"),
                "model_id": run.get("model_id"),
                "model_params": run.get("model_params"),
                "input_json": run.get("input_json"),
                "started_at": run.get("started_at"),
                "finished_at": run.get("finished_at"),
                "wall_clock_time": run.get("wall_clock_time"),
                "n_synthetic": run.get("n_synthetic"),
                **sample_stats([run]),
                "majority_label": run.get("majority_label"),
                "synthetic_label_counts": run.get("synthetic_label_counts"),
                **call_stats(call_records),
            }
        )

    summary = {
        "prefix": args.prefix,
        "path": str(domain_dir),
        "generated_at": utc_now(),
        "counting_rules": (
            "One call = one API request (initial or retry round). tokens are summed over every call that "
            "returned a response, unparsed ones included; tokens_unparsed is the part spent on unparsed "
            "responses. Failed calls (api_error) returned no response and carry no tokens or verification. "
            "n_labeled_by_api excludes samples that got the majority-label fallback; n_reused samples were "
            "taken from an earlier run's TSV without an API call."
        ),
        "files": {"log": str(log_path)},
        "totals": {
            "n_runs": len(runs),
            **sample_stats(log_runs),
            # The last run rewrote the full TSV, so its label counts describe the final output.
            "final_synthetic_label_counts": log_runs[-1].get("synthetic_label_counts") if log_runs else None,
            **call_stats(all_calls),
        },
        "runs": runs,
    }

    output_path = log_dir / f"{args.prefix}_summary.json"
    save_json(output_path, summary)

    totals = summary["totals"]
    tokens = totals["tokens"]
    verification = totals["sampling_verification"]
    print(
        f"Prefix {args.prefix!r}: {len(runs)} run(s), {totals['n_calls']} call(s) "
        f"({totals['n_labeled_calls']} labeled, {totals['n_unparsed_calls']} unparsed, {totals['n_failed_calls']} failed)"
    )
    print(
        f"  samples: {totals['n_labeled_by_api']} labeled by API, {totals['n_reused']} reused, "
        f"{totals['n_majority_fallback']} majority fallback"
    )
    print(
        f"  tokens: prompt {tokens['prompt_tokens']} + completion {tokens['completion_tokens']} "
        f"= total {tokens['total_tokens']} (cost {tokens['cost']}); "
        f"{totals['tokens_unparsed']['total_tokens']} on unparsed responses; "
        f"{totals['avg_total_tokens_per_labeled_sample']} per labeled sample"
    )
    print(f"  finish_reason: {totals['finish_reason_counts']}")
    print(
        f"  sampling verification: {verification['n_verification_logged']}/{verification['n_calls']} call(s) logged, "
        f"{verification['n_unverified']} unverified; providers {verification['routed_provider_counts']}, "
        f"quantizations {verification['quantization_counts']}"
    )
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
