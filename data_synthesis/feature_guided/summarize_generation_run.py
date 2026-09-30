"""Summarize a feature-guided generation run (all runs written under one --prefix) into a JSON file.

Feature-guided counterpart of ../blackbox/summarize_generation_run.py. Requires
<path>/output/<prefix>_accepted.json and _rejected.json plus at least one of
<path>/log/<prefix>_log.json (completed runs) and <path>/output/<prefix>_checkpoint.json (a run
that has not reached --n yet, e.g. between two --resume invocations; reported with
"status": "incomplete"). <path>/output/<prefix>_failed.json is read if present. Computes, in total
and per run_id:
  - API: calls (successful / failed) and token usage summed over the calls' OpenRouter usage, plus
    the provider/sampling-parameter verification. As in the blackbox summary, a call whose every
    sample was rejected as "target_reached" (never checked, --n already reached) is reported
    separately and left out of the call count, the tokens and the acceptance rate.
  - Samples: n_generated = n_accepted + n_rejected, with n_rejected split into
    "feature_inactive" (SAE activation check) and "rouge_duplicate" (ROUGE-L dedup).
  - Seed check: task-relevant features (per label, e.g. Yes/Probably/Maybe) and how many of them
    the seeds already cover vs. how many were left missing (= the features of pass 0).
  - Feature triggering: which features were attempted, reached (>= 1 accepted sample), exhausted
    (no accepted sample after --attempts-per-feature attempts in a pass) or still open (the run
    ended first), and the mean number of attempts reached features needed. An attempt is a call
    that returned a response and was checked, i.e. neither failed nor target_reached - exactly
    what the run's feature schedule counts. Reported for pass 0 (the seed-missing features), per
    pass, and over all passes (a feature counts as reached once, with the attempts of its first
    successful pass).
  - SAE compute: forward passes, forward tokens and net GPU seconds of the seed check and of the
    candidate activation checks (for runs of run_generation.py versions that tracked it; older
    runs report null). The SAE passes of target_reached samples did run, but are left out of
    these numbers like their tokens and reported on their own under "target_reached".
and writes it to <path>/log/<prefix>_summary.json.

Calls and samples are joined on (run_id, wave_idx, slot_index), which every version of
run_generation.py has written, so older logs (without call_id/outcomes) work as well.

Usage:
    python summarize_generation_run.py --path toxicity_detection --prefix toxicity_fg_llama_d0_6_t0_0
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent
RUN_GENERATION_SCRIPT = BASE_DIR / "run_generation.py"
BLACKBOX_SUMMARY_SCRIPT = BASE_DIR.parent / "blackbox" / "summarize_generation_run.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Must be registered before exec: @dataclass definitions look themselves up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Path resolution, feature loading and JSON I/O are shared with the generator; token usage,
# sampling verification and the call/sample join key with the blackbox summary.
fg = _load_module(RUN_GENERATION_SCRIPT, "feature_guided_run_generation")
bbs = _load_module(BLACKBOX_SUMMARY_SCRIPT, "blackbox_summarize_generation_run")
bb = fg.bb

REJECTION_REASONS = ("feature_inactive", "rouge_duplicate")
TARGET_REACHED_REASON = "target_reached"
SAE_STAT_KEYS = ("n_forward_passes", "n_forward_tokens", "gpu_seconds")


def format_seconds(seconds: float | None) -> str | None:
    """HH:MM:SS, as bb.format_wall_clock_slurm."""
    if seconds is None:
        return None
    hours, remainder = divmod(max(round(seconds), 0), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _mean(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


# ---------------------------------------------------------------------------
# Run sources: completed runs (log) and the incomplete run (checkpoint)
# ---------------------------------------------------------------------------

def run_from_log(run: dict) -> dict:
    """Normalises a completed run of the log to the fields this script uses."""
    has_sae = "sae_candidate_check" in run
    return {
        "status": "completed",
        "run_id": run["run_id"],
        "model_id": run.get("model_id"),
        "seed_group": run.get("seed_group"),
        "n_requested": run.get("n_requested"),
        "rouge_threshold": run.get("rouge_threshold"),
        "threshold": run.get("threshold"),
        "attempts_per_feature": run.get("attempts_per_feature"),
        "feature_scores": run.get("feature_scores"),
        "feature_labels": run.get("feature_labels"),
        "model_params": run.get("model_params"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "calls": run.get("calls", []),
        "seed_covered": run.get("seed_covered"),
        "seed_uncovered": run.get("seed_uncovered"),
        "seed_coverage_by_label": run.get("seed_coverage_by_label"),
        "exhausted_in_pass": run.get("exhausted_in_pass"),
        "sae_tracked": has_sae,
        "sae_seed_check": run.get("sae_seed_check"),
        "sae_candidate_check": run.get("sae_candidate_check"),
        "sae_tracking_complete": run.get("sae_tracking_complete", False) if has_sae else False,
    }


def run_from_checkpoint(checkpoint: dict) -> dict:
    """Normalises the checkpoint of a not yet completed run to the fields this script uses."""
    resolved = checkpoint.get("resolved_args", {})
    counters = checkpoint.get("counters", {})
    schedule = checkpoint.get("schedule") or {}
    has_sae = "sae_candidate_check" in counters
    return {
        "status": "incomplete",
        "run_id": checkpoint["run_id"],
        "model_id": None,
        "model": resolved.get("model"),
        "seed_group": resolved.get("seed_group"),
        "n_requested": resolved.get("n"),
        "rouge_threshold": resolved.get("rouge_threshold"),
        "threshold": resolved.get("threshold"),
        "attempts_per_feature": resolved.get("attempts_per_feature"),
        "feature_scores": resolved.get("feature_scores"),
        "feature_labels": resolved.get("feature_labels"),
        "model_params": None,
        "started_at": checkpoint.get("started_at"),
        "finished_at": None,
        "checkpoint_saved_at": checkpoint.get("saved_at"),
        "next_wave_idx": checkpoint.get("wave_idx"),
        "calls": checkpoint.get("call_records", []),
        "seed_covered": schedule.get("seed_covered"),
        "seed_uncovered": schedule.get("seed_uncovered"),
        "seed_coverage_by_label": None,
        "exhausted_in_pass": schedule.get("exhausted_in_pass"),
        "sae_tracked": has_sae,
        "sae_seed_check": counters.get("sae_seed_check"),
        "sae_candidate_check": counters.get("sae_candidate_check"),
        "sae_tracking_complete": counters.get("sae_tracking_complete", False) if has_sae else False,
    }


# ---------------------------------------------------------------------------
# Seed coverage
# ---------------------------------------------------------------------------

_FEATURE_LABEL_CACHE: dict[tuple, dict[int, str] | None] = {}


def feature_labels_by_id(feature_scores: str | None, labels: list[str] | None) -> dict[int, str] | None:
    """{feature_id: label} of the run's relevant features, re-read from its feature scores file."""
    if not feature_scores or not labels:
        return None
    key = (feature_scores, tuple(labels))
    if key not in _FEATURE_LABEL_CACHE:
        try:
            features = fg.load_features(Path(feature_scores), list(labels))
            _FEATURE_LABEL_CACHE[key] = {f["feature_id"]: f["label"] for f in features}
        except (OSError, SystemExit) as exc:
            print(f"[warn] Could not load feature labels from {feature_scores}: {exc}", file=sys.stderr)
            _FEATURE_LABEL_CACHE[key] = None
    return _FEATURE_LABEL_CACHE[key]


def seed_coverage(run: dict) -> dict | None:
    """Relevant features per label and how many the seeds cover / leave missing."""
    if run["seed_covered"] is None or run["seed_uncovered"] is None:
        return None  # run from before the seed-coverage schedule
    by_label = run["seed_coverage_by_label"]
    if by_label is None:
        label_of = feature_labels_by_id(run["feature_scores"], run["feature_labels"])
        if label_of is not None:
            covered = set(run["seed_covered"])
            by_label = {}
            for label in run["feature_labels"]:
                ids = [fid for fid, fl in label_of.items() if fl == label]
                n_covered = sum(fid in covered for fid in ids)
                by_label[label] = {"total": len(ids), "covered": n_covered, "uncovered": len(ids) - n_covered}
    n_covered = len(run["seed_covered"])
    n_missing = len(run["seed_uncovered"])
    return {
        "n_relevant": n_covered + n_missing,
        "n_covered_by_seeds": n_covered,
        "n_missing_after_seed_check": n_missing,
        "by_label": (
            {label: {"n_relevant": v["total"], "n_covered_by_seeds": v["covered"], "n_missing_after_seed_check": v["uncovered"]}
             for label, v in by_label.items()}
            if by_label is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Feature triggering
# ---------------------------------------------------------------------------

def feature_attempts(call_records: list[dict], accepted_keys: set, checked_keys: set) -> dict[tuple[int, int], dict]:
    """{(pass_idx, feature_id): {label, attempts, reached_at_attempt}} over the counted attempts,
    i.e. calls with at least one checked (accepted / feature_inactive / rouge_duplicate) sample or
    without any candidate - the calls the run's feature schedule counted as attempts."""
    per_pass: dict[tuple[int, int], dict] = {}
    for record in sorted(call_records, key=lambda r: (r.get("wave_idx", 0), r.get("slot_index", 0))):
        key = record["_key"]
        if record.get("n_parsed_candidates", 0) > 0 and key not in checked_keys:
            continue  # target_reached only: never checked, not an attempt
        entry = per_pass.setdefault(
            (record.get("pass_idx", 0), record["feature_id"]),
            {"label": record.get("feature_label"), "attempts": 0, "reached_at_attempt": None},
        )
        entry["attempts"] += 1
        if key in accepted_keys and entry["reached_at_attempt"] is None:
            entry["reached_at_attempt"] = record.get("feature_attempt", entry["attempts"] - 1) + 1
    return per_pass


def trigger_block(entries: dict[int, dict], exhausted: set[int] | None) -> dict:
    """Counts over {feature_id: {label, attempts, reached_at_attempt}} of one scope."""
    attempted = set(entries)
    reached = {fid for fid, e in entries.items() if e["reached_at_attempt"] is not None}
    exhausted = (exhausted or set()) & attempted
    attempts_to_reach = [entries[fid]["reached_at_attempt"] for fid in reached]
    by_label: dict[str, dict] = defaultdict(lambda: {"n_attempted": 0, "n_reached": 0, "n_exhausted": 0})
    for fid, e in entries.items():
        stats = by_label[e["label"]]
        stats["n_attempted"] += 1
        stats["n_reached"] += fid in reached
        stats["n_exhausted"] += fid in exhausted
    return {
        "n_features_attempted": len(attempted),
        "n_features_reached": len(reached),
        "n_features_exhausted": len(exhausted),
        # Over several passes a feature can be reached in one pass and exhausted in a later one,
        # so reached + exhausted may exceed attempted there; never_reached has no such overlap.
        "n_features_never_reached": len(attempted - reached),
        "n_features_open": len(attempted - reached - exhausted),
        "n_attempts": sum(e["attempts"] for e in entries.values()),
        "mean_attempts_to_reach": _mean(attempts_to_reach),
        "attempts_to_reach_distribution": {str(k): v for k, v in sorted(Counter(attempts_to_reach).items())},
        "by_label": dict(by_label),
    }


def trigger_stats(run: dict, per_pass: dict[tuple[int, int], dict]) -> dict:
    exhausted_in_pass = run["exhausted_in_pass"]
    if exhausted_in_pass is None:
        # Older log without the schedule: exhausted = attempted up to the limit without an accept.
        limit = run["attempts_per_feature"]
        exhausted_in_pass = {
            str(fid): pass_idx for (pass_idx, fid), e in per_pass.items()
            if limit and e["reached_at_attempt"] is None and e["attempts"] >= limit
        }
    exhausted_by_pass: dict[int, set[int]] = defaultdict(set)
    for fid, pass_idx in exhausted_in_pass.items():
        exhausted_by_pass[int(pass_idx)].add(int(fid))

    by_pass: dict[int, dict[int, dict]] = defaultdict(dict)
    for (pass_idx, fid), entry in per_pass.items():
        by_pass[pass_idx][fid] = entry

    # Over all passes: each feature once, with the attempts of its first successful pass (or of
    # its last pass, if it was never reached).
    overall: dict[int, dict] = {}
    for pass_idx in sorted(by_pass):
        for fid, entry in by_pass[pass_idx].items():
            if fid not in overall or overall[fid]["reached_at_attempt"] is None:
                overall[fid] = entry
    all_exhausted = set().union(*exhausted_by_pass.values()) if exhausted_by_pass else set()

    return {
        "pass_0_seed_missing_features": trigger_block(by_pass.get(0, {}), exhausted_by_pass.get(0)),
        "all_passes": {
            "n_passes_started": (max(by_pass) + 1) if by_pass else 0,
            **trigger_block(overall, all_exhausted),
        },
        "by_pass": {str(p): trigger_block(by_pass[p], exhausted_by_pass.get(p)) for p in sorted(by_pass)},
    }


# ---------------------------------------------------------------------------
# SAE compute
# ---------------------------------------------------------------------------

def sae_block(stats: dict | None) -> dict | None:
    if stats is None:
        return None
    return {**{k: stats[k] for k in SAE_STAT_KEYS}, "gpu_time": format_seconds(stats["gpu_seconds"])}


def sum_sae(blocks: list[dict | None]) -> dict | None:
    blocks = [b for b in blocks if b is not None]
    if not blocks:
        return None
    total = {k: sum(b[k] for b in blocks) for k in SAE_STAT_KEYS}
    total["gpu_seconds"] = round(total["gpu_seconds"], 6)
    return total


def sae_from_calls(call_records: list[dict]) -> dict:
    total = {
        "n_forward_passes": sum(r.get("sae_n_forward_passes", 0) for r in call_records),
        "n_forward_tokens": sum(r.get("sae_n_forward_tokens", 0) for r in call_records),
        "gpu_seconds": round(sum(r.get("sae_gpu_seconds", 0.0) for r in call_records), 6),
    }
    return total


def sae_from_samples(samples: list[dict]) -> dict:
    """SAE compute of the given samples: each ran through exactly one SAE forward pass."""
    return {
        "n_forward_passes": len(samples),
        "n_forward_tokens": sum(e["sae_n_forward_tokens"] for e in samples),
        "gpu_seconds": round(sum(e["sae_seconds"] for e in samples), 6),
    }


def sae_stats(runs: list[dict], tr_samples: list[dict]) -> tuple[dict, dict | None]:
    """(SAE block without target_reached samples, SAE block of the target_reached samples).
    The runs' candidate-check counters include the target_reached samples (their SAE pass ran
    before the target check), so these are subtracted sample by sample - also those of calls
    that additionally produced a checked sample."""
    tracked = [r for r in runs if r["sae_tracked"]]
    if not tracked:
        return {"tracked": False, "note": "SAE compute was not tracked by the run_generation.py version of these run(s)."}, None
    tracked_ids = {r["run_id"] for r in tracked}
    tr_tracked = [e for e in tr_samples if e.get("run_id") in tracked_ids]
    tr_complete = all("sae_seconds" in e and "sae_n_forward_tokens" in e for e in tr_tracked)
    tr = sae_from_samples([e for e in tr_tracked if "sae_seconds" in e and "sae_n_forward_tokens" in e])
    seed = sum_sae([r["sae_seed_check"] for r in tracked])
    candidate_all = sum_sae([r["sae_candidate_check"] for r in tracked])
    candidate = {k: candidate_all[k] - tr[k] for k in SAE_STAT_KEYS} if candidate_all is not None else None
    if candidate is not None:
        candidate["gpu_seconds"] = round(max(candidate["gpu_seconds"], 0.0), 6)
    total = sum_sae([seed, candidate])
    return {
        "tracked": True,
        "tracking_complete": (len(tracked) == len(runs) and tr_complete
                              and all(r["sae_tracking_complete"] for r in tracked)),
        "seed_check": sae_block(seed),
        "candidate_check": sae_block(candidate),
        "total": sae_block(total),
    }, sae_block(tr)


# ---------------------------------------------------------------------------
# Per-run / total stats
# ---------------------------------------------------------------------------

def run_stats(runs: list[dict], accepted: list[dict], rejected: list[dict], failed: list[dict]) -> dict:
    call_records = [c for r in runs for c in r["calls"]]
    by_reason = {reason: [e for e in rejected if e.get("rejected_reason") == reason] for reason in REJECTION_REASONS}
    target_reached = [e for e in rejected if e.get("rejected_reason") == TARGET_REACHED_REASON]
    checked = accepted + by_reason["feature_inactive"] + by_reason["rouge_duplicate"]
    accepted_keys = {e["_key"] for e in accepted}
    checked_keys = {e["_key"] for e in checked}

    tr_call_keys = {e["_key"] for e in target_reached} - checked_keys
    counted_calls = [c for c in call_records if c["_key"] not in tr_call_keys]
    tr_calls = [c for c in call_records if c["_key"] in tr_call_keys]

    n_accepted = len(accepted)
    n_rejected = sum(len(v) for v in by_reason.values())
    n_generated = n_accepted + n_rejected
    sae, tr_sae = sae_stats(runs, target_reached)
    stats = {
        "api": {
            "n_calls": len(counted_calls) + len(failed),
            "n_successful_calls": len(counted_calls),
            "n_failed_calls": len(failed),
            "n_calls_without_candidate": sum(1 for c in counted_calls if c.get("n_parsed_candidates", 0) == 0),
            "tokens": bbs.sum_usage(counted_calls),
            "sampling_verification": bbs.verification_summary(counted_calls),
        },
        "samples": {
            "n_generated": n_generated,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "n_rejected_feature_inactive": len(by_reason["feature_inactive"]),
            "n_rejected_rouge_duplicate": len(by_reason["rouge_duplicate"]),
            "acceptance_rate": round(n_accepted / n_generated, 4) if n_generated else None,
        },
        # Reported only, excluded from api, samples and sae.
        "target_reached": {
            "n_samples": len(target_reached),
            "n_calls": len(tr_calls),
            "tokens": bbs.sum_usage(tr_calls),
            "sae": tr_sae,
        },
        "sae": sae,
    }
    if len(runs) == 1:
        run = runs[0]
        stats["seed_check"] = seed_coverage(run)
        stats["feature_triggering"] = trigger_stats(run, feature_attempts(run["calls"], accepted_keys, checked_keys))
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize token usage, accepted/rejected counts, feature triggering and SAE compute "
        "of a feature-guided generation run (by --prefix)."
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Domain subfolder (e.g. 'toxicity_detection'), either a name under feature_guided/ or a path to it.",
    )
    parser.add_argument("--prefix", type=str, required=True, help="The --prefix the run was generated with.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    domain_dir = fg.resolve_domain_dir(args.path)
    output_dir = domain_dir / "output"
    log_dir = domain_dir / "log"
    files = {
        "accepted": output_dir / f"{args.prefix}_accepted.json",
        "rejected": output_dir / f"{args.prefix}_rejected.json",
        "failed": output_dir / f"{args.prefix}_failed.json",
        "log": log_dir / f"{args.prefix}_log.json",
        "checkpoint": output_dir / f"{args.prefix}_checkpoint.json",
    }
    missing = [str(files[key]) for key in ("accepted", "rejected") if not files[key].exists()]
    if missing:
        raise SystemExit(f"Missing required file(s) for prefix {args.prefix!r}: {missing}")
    if not files["log"].exists() and not files["checkpoint"].exists():
        raise SystemExit(f"Neither {files['log']} nor {files['checkpoint']} exists for prefix {args.prefix!r}.")

    accepted = [e for e in bb.load_json_list(files["accepted"]) if e.get("type") == "synthetic"]
    rejected = bb.load_json_list(files["rejected"])
    failed = [e for e in bb.load_json_list(files["failed"]) if bbs.is_generation_failure(e)]
    runs = [run_from_log(r) for r in (bb.load_json_dict(files["log"]) or {}).get("runs", [])]
    checkpoint = bb.load_json_dict(files["checkpoint"])
    if checkpoint is not None:
        runs.append(run_from_checkpoint(checkpoint))

    for entry in accepted + rejected + failed:
        entry["_key"] = bbs.slot_key(entry.get("run_id"), entry)
    for run in runs:
        for record in run["calls"]:
            record["_key"] = bbs.slot_key(run["run_id"], record)

    run_summaries = []
    for run in runs:
        run_id = run["run_id"]
        meta = {k: v for k, v in run.items()
                if k not in ("calls", "seed_covered", "seed_uncovered", "seed_coverage_by_label", "exhausted_in_pass")
                and not k.startswith("sae_")}
        run_summaries.append(
            {
                **meta,
                **run_stats(
                    [run],
                    [e for e in accepted if e.get("run_id") == run_id],
                    [e for e in rejected if e.get("run_id") == run_id],
                    [e for e in failed if e.get("run_id") == run_id],
                ),
            }
        )

    run_ids = {run["run_id"] for run in runs}
    in_runs = lambda entries: [e for e in entries if e.get("run_id") in run_ids]  # noqa: E731
    totals = {"n_runs": len(runs), **run_stats(runs, in_runs(accepted), in_runs(rejected), in_runs(failed))}
    if len(runs) == 1:
        # Same run as runs[0]; per-run blocks already hold them.
        totals.pop("seed_check", None)
        totals.pop("feature_triggering", None)

    summary = {
        "prefix": args.prefix,
        "path": str(domain_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counting_rules": (
            "Rejections are 'feature_inactive' (SAE activation check) and 'rouge_duplicate' (ROUGE-L dedup); "
            "n_generated = n_accepted + n_rejected. 'target_reached' samples and their calls are reported under "
            "'target_reached' and excluded from api, samples and sae (target_reached.sae holds the SAE passes of "
            "exactly the target_reached samples, also those of calls that produced a checked sample). A feature attempt is a call that returned a "
            "response and was checked (not failed, not target_reached). mean_attempts_to_reach = mean over reached "
            "features of the 1-based attempt (within the pass) that produced the first accepted sample. SAE "
            "gpu_seconds = net wall-clock time of the SAE forward passes between CUDA synchronizations; waves "
            "interrupted before their checkpoint are not counted."
        ),
        "files": {key: str(path) for key, path in files.items() if path.exists()},
        "totals": totals,
        "runs": run_summaries,
    }

    output_path = log_dir / f"{args.prefix}_summary.json"
    bb.save_json(output_path, summary)

    api, samples, tr, sae = totals["api"], totals["samples"], totals["target_reached"], totals["sae"]
    tokens = api["tokens"]
    n_incomplete = sum(r["status"] == "incomplete" for r in runs)
    print(f"Prefix {args.prefix!r}: {len(runs)} run(s) ({n_incomplete} incomplete), "
          f"{api['n_calls']} call(s) ({api['n_failed_calls']} failed)")
    print(f"  generated {samples['n_generated']} = accepted {samples['n_accepted']} + rejected {samples['n_rejected']} "
          f"(feature inactive {samples['n_rejected_feature_inactive']}, ROUGE duplicate {samples['n_rejected_rouge_duplicate']}) "
          f"-> acceptance rate {samples['acceptance_rate']}")
    print(f"  tokens: prompt {tokens['prompt_tokens']} + completion {tokens['completion_tokens']} "
          f"= total {tokens['total_tokens']} (cost {tokens['cost']})")
    print(f"  excluded target_reached: {tr['n_samples']} sample(s) from {tr['n_calls']} call(s), "
          f"{tr['tokens']['total_tokens']} token(s)")
    for run in run_summaries:
        print(f"  run {run['run_id']} ({run['status']}):")
        seed = run.get("seed_check")
        if seed is not None:
            labels = ", ".join(f"{label} {v['n_missing_after_seed_check']}/{v['n_relevant']}"
                               for label, v in (seed["by_label"] or {}).items())
            print(f"    seed check: {seed['n_relevant']} relevant feature(s), {seed['n_covered_by_seeds']} covered, "
                  f"{seed['n_missing_after_seed_check']} missing" + (f" (missing/relevant: {labels})" if labels else ""))
        for name, block in (("pass 0", run["feature_triggering"]["pass_0_seed_missing_features"]),
                            ("all passes", run["feature_triggering"]["all_passes"])):
            print(f"    features {name}: attempted {block['n_features_attempted']}, reached {block['n_features_reached']}, "
                  f"exhausted {block['n_features_exhausted']}, never reached {block['n_features_never_reached']}, "
                  f"open {block['n_features_open']}; "
                  f"mean attempts to reach {block['mean_attempts_to_reach']} {block['attempts_to_reach_distribution']}")
    if sae["tracked"]:
        total = sae["total"]
        print(f"  SAE: {total['n_forward_passes']} forward pass(es), {total['n_forward_tokens']} token(s), "
              f"{total['gpu_seconds']:.2f} s ({total['gpu_time']}) net GPU time"
              + ("" if sae["tracking_complete"] else " [incomplete tracking]"))
        tr_sae = tr["sae"]
        print(f"  excluded SAE of target_reached samples: {tr_sae['n_forward_passes']} forward pass(es), "
              f"{tr_sae['gpu_seconds']:.2f} s")
    else:
        print("  SAE: not tracked for these run(s)")
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
