"""Summarize a hybrid generation run (all runs written under one --prefix) into a JSON file.

Hybrid counterpart of ../feature_guided/summarize_generation_run.py and
../blackbox/summarize_generation_run.py: merges the blackbox phase (<prefix>_bb_*) and the
feature-guided phase (<prefix>_fg_*) of every run. Requires hybrid/<domain>/output/<prefix>_bb_accepted.json
and _bb_rejected.json plus at least one of hybrid/<domain>/log/<prefix>_bb_log.json (runs whose blackbox
phase completed) and hybrid/<domain>/output/<prefix>_checkpoint.json (a run that has not finished yet;
reported with "status": "incomplete" and the phase it stopped in). The _fg_* files, _fg_log.json,
_log.json and both _failed.json files are read if present. Computes, in total and per run_id:
  - overall: API calls, token usage and samples of both phases together, plus the token usage
    split by phase ("tokens_by_phase": total / blackbox / feature_guided).
  - blackbox: the blackbox phase, counted as in the blackbox summary (only "rouge_duplicate" is
    a rejection).
  - feature_guided: the feature-guided phase, counted as in the feature-guided summary
    ("feature_inactive" + "rouge_duplicate"), with the feature triggering (pass 0 = the relevant
    features covered neither by the seeds nor by the blackbox examples) and the SAE compute of
    the candidate checks and of the seed + blackbox coverage check (not tracked by older runs).
  - coverage (per run): relevant features covered by the seeds, by the blackbox examples, by the
    blackbox examples only, and left uncovered (= the features of pass 0), in total and per label.
In every phase, "target_reached" samples (never checked, the phase's target already reached) are
reported separately under "target_reached" and left out of the sample counts and the SAE compute;
a call whose every sample was target_reached is likewise left out of the call count and tokens.
Writes hybrid/<domain>/log/<prefix>_summary.json.

Calls and samples are joined on (run_id, wave_idx, slot_index) WITHIN a phase: both phases share
the run_id and count waves/slots from 0, so the phases are always evaluated on their own files.

Usage:
    python summarize_generation_run.py --domain toxicity_detection --prefix toxicity_hybrid_350_50_llama_d0_6_t0_0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ARM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ARM_DIR.parent))

# The per-phase counting is the blackbox / feature-guided summaries' own.
from blackbox import summarize_generation_run as bbs  # noqa: E402
from feature_guided import summarize_generation_run as fgs  # noqa: E402
from shared.benchmarks import DOMAINS  # noqa: E402
from shared.generation import PHASE_BLACKBOX, PHASE_FEATURE_GUIDED  # noqa: E402
from shared.run_io import load_json_dict, load_json_list, save_json, utc_now  # noqa: E402
from shared.summary import is_generation_failure, slot_key, verification_summary  # noqa: E402

PHASES = (PHASE_BLACKBOX, PHASE_FEATURE_GUIDED)
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "cached_prompt_tokens", "cost")


def sum_tokens(blocks: list[dict]) -> dict:
    total = {key: sum(block[key] for block in blocks) for key in TOKEN_KEYS}
    total["cost"] = round(total["cost"], 10)
    return total


def counted_calls(call_records: list[dict], accepted: list[dict], rejected: list[dict]) -> list[dict]:
    """The calls both summaries count: all but those whose every sample was target_reached."""
    checked_keys = {e["_key"] for e in accepted + rejected if e.get("rejected_reason") != fgs.TARGET_REACHED_REASON}
    tr_keys = {e["_key"] for e in rejected if e.get("rejected_reason") == fgs.TARGET_REACHED_REASON} - checked_keys
    return [c for c in call_records if c["_key"] not in tr_keys]


# ---------------------------------------------------------------------------
# Run sources: logs (completed phases) and the checkpoint (incomplete run)
# ---------------------------------------------------------------------------

def collect_runs(logs: dict[str, dict | None], checkpoint: dict | None) -> list[dict]:
    """One entry per run_id with the call records of both phases, taken from the phase logs where
    the phase completed and from the checkpoint otherwise."""
    bb_runs = {r["run_id"]: r for r in (logs["bb_log"] or {}).get("runs", [])}
    fg_runs = {r["run_id"]: r for r in (logs["fg_log"] or {}).get("runs", [])}
    hybrid_runs = {r["run_id"]: r for r in (logs["log"] or {}).get("runs", [])}
    run_ids = list(dict.fromkeys([*hybrid_runs, *bb_runs, *fg_runs]))
    if checkpoint is not None and checkpoint["run_id"] not in run_ids:
        run_ids.append(checkpoint["run_id"])

    runs = []
    for run_id in run_ids:
        cp = checkpoint if checkpoint is not None and checkpoint["run_id"] == run_id else None
        bb_run, fg_run, hybrid_run = bb_runs.get(run_id), fg_runs.get(run_id), hybrid_runs.get(run_id)
        bb_state = (cp or {}).get("blackbox") or {}
        fg_state = (cp or {}).get("feature_guided") or {}
        schedule = fg_run if fg_run is not None else fg_state.get("schedule") or {}
        resolved = (cp or {}).get("resolved_args", {})
        # The combined log is written last, so a run is completed exactly when it is in there.
        completed = hybrid_run is not None
        meta_src = hybrid_run or fg_run or bb_run or {}
        runs.append({
            "run_id": run_id,
            "status": "completed" if completed else "incomplete",
            "stopped_in_phase": None if completed else (cp or {}).get("phase"),
            "model": meta_src.get("model", resolved.get("model")),
            "model_id": meta_src.get("model_id"),
            "seed_group": meta_src.get("seed_group", resolved.get("seed_group")),
            "n_requested_blackbox": (hybrid_run or {}).get("n_requested_blackbox", (bb_run or {}).get("n_requested", resolved.get("n_blackbox"))),
            "n_requested_feature_guided": (hybrid_run or {}).get("n_requested_feature_guided", (fg_run or {}).get("n_requested", resolved.get("n_feature_guided"))),
            "rouge_threshold": meta_src.get("rouge_threshold", resolved.get("rouge_threshold")),
            "threshold": (fg_run or hybrid_run or {}).get("threshold", resolved.get("threshold")),
            "attempts_per_feature": (fg_run or {}).get("attempts_per_feature", resolved.get("attempts_per_feature")),
            "feature_scores": (fg_run or {}).get("feature_scores", resolved.get("feature_scores")),
            "feature_labels": (fg_run or {}).get("feature_labels", resolved.get("feature_labels")),
            "model_params": meta_src.get("model_params"),
            "started_at": (hybrid_run or {}).get("started_at", (cp or {}).get("started_at", (bb_run or {}).get("started_at"))),
            "finished_at": (hybrid_run or {}).get("finished_at"),
            "wall_clock_time": (hybrid_run or {}).get("wall_clock_time"),
            "checkpoint_saved_at": (cp or {}).get("saved_at"),
            "blackbox_started_at": (bb_run or {}).get("started_at", bb_state.get("started_at")),
            "blackbox_finished_at": (bb_run or {}).get("finished_at", bb_state.get("finished_at")),
            "feature_guided_started_at": (fg_run or {}).get("started_at", fg_state.get("started_at")),
            "feature_guided_finished_at": (fg_run or {}).get("finished_at"),
            "_calls": {
                PHASE_BLACKBOX: bb_run["calls"] if bb_run is not None else bb_state.get("call_records", []),
                PHASE_FEATURE_GUIDED: fg_run["calls"] if fg_run is not None else fg_state.get("call_records", []),
            },
            # Runs before the shared feature-guidance code did not track the coverage check.
            "_sae_coverage": (fg_run or {}).get("sae_coverage_check", (fg_state.get("counters") or {}).get("sae_coverage_check")),
            "_schedule": {
                key: schedule.get(key)
                for key in ("covered", "uncovered", "seed_covered", "blackbox_covered", "blackbox_only_covered", "exhausted_in_pass")
            },
        })
    return runs


def fg_summary_run(run: dict) -> dict:
    """The run in the normalised form feature_guided/summarize_generation_run.py works on. SAE
    compute of the candidate checks is summed from the call records; the coverage check (seeds +
    blackbox examples) is taken from the run's counters where tracked."""
    calls = run["_calls"][PHASE_FEATURE_GUIDED]
    tracked = bool(calls) and all("sae_n_forward_passes" in c for c in calls)
    return {
        "run_id": run["run_id"],
        "calls": calls,
        "seed_covered": None,
        "seed_uncovered": None,
        "seed_coverage_by_label": None,
        "exhausted_in_pass": run["_schedule"]["exhausted_in_pass"],
        "attempts_per_feature": run["attempts_per_feature"],
        "feature_scores": run["feature_scores"],
        "feature_labels": run["feature_labels"],
        "sae_tracked": tracked,
        "sae_seed_check": run["_sae_coverage"],
        "sae_candidate_check": fgs.sae_from_calls(calls) if tracked else None,
        "sae_tracking_complete": tracked,
    }


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def coverage(run: dict) -> dict | None:
    """Relevant features covered by seeds / blackbox examples / either, and left uncovered."""
    schedule = run["_schedule"]
    if schedule["covered"] is None:
        return None  # the run has not reached the coverage check yet
    sets = {
        "n_covered_by_seeds": set(schedule["seed_covered"]),
        "n_covered_by_blackbox": set(schedule["blackbox_covered"]),
        "n_covered_by_blackbox_only": set(schedule["blackbox_only_covered"]),
        "n_covered": set(schedule["covered"]),
        "n_uncovered": set(schedule["uncovered"]),
    }

    def counts(ids: set[int]) -> dict:
        return {"n_relevant": len(ids), **{key: len(s & ids) for key, s in sets.items()}}

    label_of = fgs.feature_labels_by_id(run["feature_scores"], run["feature_labels"])
    return {
        **counts(sets["n_covered"] | sets["n_uncovered"]),
        "by_label": (
            {label: counts({fid for fid, fl in label_of.items() if fl == label}) for label in run["feature_labels"]}
            if label_of is not None else None
        ),
    }


# ---------------------------------------------------------------------------
# Per-phase / overall stats
# ---------------------------------------------------------------------------

def blackbox_stats(calls: list[dict], accepted: list[dict], rejected: list[dict], failed: list[dict]) -> dict:
    """The blackbox summary's counts, in the feature-guided summary's layout."""
    s = bbs.run_stats(calls, accepted, rejected, failed)
    return {
        "api": {
            "n_calls": s["n_calls"],
            "n_successful_calls": s["n_successful_calls"],
            "n_failed_calls": s["n_failed_calls"],
            "tokens": s["tokens"],
            "sampling_verification": s["sampling_verification"],
        },
        "samples": {
            "n_generated": s["n_generated"],
            "n_accepted": s["n_accepted"],
            "n_rejected": s["n_rejected"],
            "n_rejected_rouge_duplicate": s["n_rejected"],
            "acceptance_rate": s["acceptance_rate"],
        },
        "target_reached": s["target_reached"],
    }


def feature_guided_stats(runs: list[dict], accepted: list[dict], rejected: list[dict], failed: list[dict]) -> dict:
    stats = fgs.run_stats([fg_summary_run(r) for r in runs], accepted, rejected, failed)
    stats.pop("seed_check", None)
    triggering = stats.pop("feature_triggering", None)
    if triggering is not None:
        stats["feature_triggering"] = {
            "pass_0_uncovered_features": triggering["pass_0_seed_missing_features"],
            "all_passes": triggering["all_passes"],
            "by_pass": triggering["by_pass"],
        }
    sae = stats["sae"]
    if sae["tracked"]:
        sae["coverage_check"] = sae.pop("seed_check")
        if sae["coverage_check"] is None:
            sae["note"] = "SAE compute of the seed + blackbox coverage check was not tracked (older run); total = candidate checks only."
    else:
        sae["note"] = "SAE compute was not tracked by the generator version of these run(s)."
    return stats


def overall_stats(phase_stats: dict[str, dict], counted: list[dict]) -> dict:
    bb_s, fg_s = phase_stats[PHASE_BLACKBOX], phase_stats[PHASE_FEATURE_GUIDED]
    both = (bb_s, fg_s)
    samples = {
        key: sum(s["samples"].get(key, 0) for s in both)
        for key in ("n_generated", "n_accepted", "n_rejected", "n_rejected_feature_inactive", "n_rejected_rouge_duplicate")
    }
    samples["acceptance_rate"] = round(samples["n_accepted"] / samples["n_generated"], 4) if samples["n_generated"] else None
    tokens_by_phase = {phase: phase_stats[phase]["api"]["tokens"] for phase in PHASES}
    return {
        "api": {
            **{key: sum(s["api"][key] for s in both) for key in ("n_calls", "n_successful_calls", "n_failed_calls")},
            "n_calls_by_phase": {phase: phase_stats[phase]["api"]["n_calls"] for phase in PHASES},
            "tokens": sum_tokens(list(tokens_by_phase.values())),
            "sampling_verification": verification_summary(counted),
        },
        "tokens_by_phase": {"total": sum_tokens(list(tokens_by_phase.values())), **tokens_by_phase},
        "samples": samples,
        "target_reached": {
            "n_samples": sum(s["target_reached"]["n_samples"] for s in both),
            "n_calls": sum(s["target_reached"]["n_calls"] for s in both),
            "tokens": sum_tokens([s["target_reached"]["tokens"] for s in both]),
            # Only the feature-guided phase runs candidates through the SAE.
            "sae": fg_s["target_reached"]["sae"],
        },
    }


def hybrid_stats(runs: list[dict], data: dict[str, dict[str, list[dict]]]) -> dict:
    """Stats over the given runs; data[phase] holds that phase's accepted/rejected/failed entries."""
    run_ids = {r["run_id"] for r in runs}
    picked = {
        phase: {kind: [e for e in entries if e.get("run_id") in run_ids] for kind, entries in data[phase].items()}
        for phase in PHASES
    }
    calls = {phase: [c for r in runs for c in r["_calls"][phase]] for phase in PHASES}
    phase_stats = {
        PHASE_BLACKBOX: blackbox_stats(calls[PHASE_BLACKBOX], **picked[PHASE_BLACKBOX]),
        PHASE_FEATURE_GUIDED: feature_guided_stats(runs, **picked[PHASE_FEATURE_GUIDED]),
    }
    counted = [
        c for phase in PHASES
        for c in counted_calls(calls[phase], picked[phase]["accepted"], picked[phase]["rejected"])
    ]
    stats = {"overall": overall_stats(phase_stats, counted), **phase_stats}
    if len(runs) == 1:
        stats["coverage"] = coverage(runs[0])
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize token usage (total and per phase), accepted/rejected counts, coverage, feature "
        "triggering and SAE compute of a hybrid generation run (by --prefix)."
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
        **{f"{p}_{kind}": output_dir / f"{args.prefix}_{p}_{kind}.json"
           for p in ("bb", "fg") for kind in ("accepted", "rejected", "failed")},
        "accepted": output_dir / f"{args.prefix}_accepted.json",
        "checkpoint": output_dir / f"{args.prefix}_checkpoint.json",
        "bb_log": log_dir / f"{args.prefix}_bb_log.json",
        "fg_log": log_dir / f"{args.prefix}_fg_log.json",
        "log": log_dir / f"{args.prefix}_log.json",
    }
    missing = [str(files[key]) for key in ("bb_accepted", "bb_rejected") if not files[key].exists()]
    if missing:
        raise SystemExit(f"Missing required file(s) for prefix {args.prefix!r}: {missing}")
    if not files["bb_log"].exists() and not files["checkpoint"].exists():
        raise SystemExit(f"Neither {files['bb_log']} nor {files['checkpoint']} exists for prefix {args.prefix!r}.")

    data: dict[str, dict[str, list[dict]]] = {}
    for phase, short in zip(PHASES, ("bb", "fg")):
        data[phase] = {
            "accepted": [e for e in load_json_list(files[f"{short}_accepted"]) if e.get("type") == "synthetic"],
            "rejected": load_json_list(files[f"{short}_rejected"]),
            "failed": [e for e in load_json_list(files[f"{short}_failed"]) if is_generation_failure(e)],
        }
        for entries in data[phase].values():
            for entry in entries:
                entry["_key"] = slot_key(entry.get("run_id"), entry)
    runs = collect_runs({key: load_json_dict(files[key]) for key in ("bb_log", "fg_log", "log")},
                        load_json_dict(files["checkpoint"]))
    for run in runs:
        for records in run["_calls"].values():
            for record in records:
                record["_key"] = slot_key(run["run_id"], record)

    run_summaries = [
        {**{k: v for k, v in run.items() if not k.startswith("_")}, **hybrid_stats([run], data)}
        for run in runs
    ]
    totals = {"n_runs": len(runs)}
    if len(runs) > 1:
        # With a single run, the per-run block already holds everything.
        totals.update(hybrid_stats(runs, data))
        totals.pop("coverage", None)

    summary = {
        "prefix": args.prefix,
        "path": str(domain_dir),
        "generated_at": utc_now(),
        "counting_rules": (
            "Each phase is counted on its own files. Blackbox phase: only 'rouge_duplicate' is a rejection. "
            "Feature-guided phase: 'feature_inactive' (SAE activation check) and 'rouge_duplicate' (ROUGE-L dedup). "
            "n_generated = n_accepted + n_rejected. 'target_reached' samples and their calls are reported under "
            "'target_reached' and excluded from api, samples and sae (target_reached.sae holds the SAE passes of "
            "exactly the target_reached samples, also those of calls that produced a checked sample). overall = blackbox + feature_guided; "
            "tokens_by_phase splits overall.api.tokens. Feature triggering and mean_attempts_to_reach as in "
            "feature_guided/summarize_generation_run.py, with pass 0 = the relevant features covered neither by the "
            "seeds nor by the blackbox examples."
        ),
        "files": {key: str(path) for key, path in files.items() if path.exists()},
        "totals": totals,
        "runs": run_summaries,
    }

    output_path = log_dir / f"{args.prefix}_summary.json"
    save_json(output_path, summary)

    n_incomplete = sum(r["status"] == "incomplete" for r in runs)
    print(f"Prefix {args.prefix!r}: {len(runs)} run(s) ({n_incomplete} incomplete)")
    for run in run_summaries:
        overall = run["overall"]
        status = run["status"] + (f" in phase {run['stopped_in_phase']}" if run["stopped_in_phase"] else "")
        print(f"  run {run['run_id']} ({status}):")
        for name, tokens in overall["tokens_by_phase"].items():
            print(f"    tokens {name:<14}: prompt {tokens['prompt_tokens']} + completion {tokens['completion_tokens']} "
                  f"= total {tokens['total_tokens']} (cost {tokens['cost']})")
        for name, block in (("overall", overall), *((p, run[p]) for p in PHASES)):
            api, samples, tr = block["api"], block["samples"], block["target_reached"]
            print(f"    {name:<14}: {api['n_calls']} call(s) ({api['n_failed_calls']} failed), "
                  f"generated {samples['n_generated']} = accepted {samples['n_accepted']} + rejected {samples['n_rejected']} "
                  f"(feature inactive {samples.get('n_rejected_feature_inactive', 0)}, "
                  f"ROUGE duplicate {samples['n_rejected_rouge_duplicate']}) -> acceptance rate {samples['acceptance_rate']}; "
                  f"excluded target_reached {tr['n_samples']} sample(s) / {tr['n_calls']} call(s)")
        cov = run.get("coverage")
        if cov is not None:
            labels = ", ".join(f"{label} {v['n_uncovered']}/{v['n_relevant']}" for label, v in (cov["by_label"] or {}).items())
            print(f"    coverage: {cov['n_relevant']} relevant feature(s), seeds {cov['n_covered_by_seeds']}, "
                  f"blackbox {cov['n_covered_by_blackbox']} ({cov['n_covered_by_blackbox_only']} only by blackbox), "
                  f"covered {cov['n_covered']}, uncovered {cov['n_uncovered']}"
                  + (f" (uncovered/relevant: {labels})" if labels else ""))
        triggering = run[PHASE_FEATURE_GUIDED].get("feature_triggering")
        for name, block in (("pass 0", triggering["pass_0_uncovered_features"]), ("all passes", triggering["all_passes"])):
            print(f"    features {name}: attempted {block['n_features_attempted']}, reached {block['n_features_reached']}, "
                  f"exhausted {block['n_features_exhausted']}, never reached {block['n_features_never_reached']}, "
                  f"open {block['n_features_open']}; "
                  f"mean attempts to reach {block['mean_attempts_to_reach']} {block['attempts_to_reach_distribution']}")
        sae = run[PHASE_FEATURE_GUIDED]["sae"]
        if sae["tracked"]:
            cand = sae["candidate_check"]
            print(f"    SAE candidate checks: {cand['n_forward_passes']} forward pass(es), {cand['n_forward_tokens']} token(s), "
                  f"{cand['gpu_seconds']:.2f} s ({cand['gpu_time']}) net GPU time")
            cov_sae = sae["coverage_check"]
            print("    SAE coverage check: " + ("not tracked (older run)" if cov_sae is None else
                  f"{cov_sae['n_forward_passes']} forward pass(es), {cov_sae['gpu_seconds']:.2f} s"))
            tr_sae = run[PHASE_FEATURE_GUIDED]["target_reached"]["sae"]
            print(f"    excluded SAE of target_reached samples: {tr_sae['n_forward_passes']} forward pass(es), "
                  f"{tr_sae['gpu_seconds']:.2f} s")
        else:
            print("    SAE: not tracked for this run")
    if len(runs) > 1:
        tokens = totals["overall"]["tokens_by_phase"]
        print(f"  all runs: tokens total {tokens['total']['total_tokens']} = blackbox {tokens['blackbox']['total_tokens']} "
              f"+ feature-guided {tokens['feature_guided']['total_tokens']} (cost {tokens['total']['cost']})")
    print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
