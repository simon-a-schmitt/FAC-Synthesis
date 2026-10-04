"""Hybrid arm: first --n-blackbox examples exactly as in the blackbox arm, then --n-feature-guided
examples exactly as in the feature-guided arm, targeted at the relevant SAE features that neither
the seeds nor the blackbox examples cover.

Both phases run the same shared code as the standalone arms (shared/generation.run_phase, with
shared/feature_guidance.FeatureGuidance in phase 2) and share one run_id, generator model,
--rouge-threshold and accepted pool:
  Phase 1 (blackbox): BLACKBOX_TEMPLATE, until --n-blackbox candidates are accepted.
  Coverage (once, between the phases, checkpointed): seeds + accepted blackbox examples through
      Llama + SAE; a relevant feature is covered if any of them activates it.
  Phase 2 (feature-guided): FEATURE_GUIDED_TEMPLATE, pass 0 over the still uncovered features,
      until --n-feature-guided candidates are accepted. Its context examples are drawn from (and
      deduped against) a pool that already holds the blackbox examples.

Output (hybrid/<domain>/output and /log, named after --prefix):
  <prefix>_bb_accepted / _bb_rejected / _bb_discarded / _bb_failed.json, log/<prefix>_bb_log.json   phase 1
  <prefix>_fg_accepted / _fg_rejected / _fg_discarded / _fg_failed.json, log/<prefix>_fg_log.json   phase 2
  <prefix>_accepted.json, log/<prefix>_log.json   seeds + both phases (each synthetic entry tagged
      with "phase") and a run summary - same naming as the other arms, so labeling works on it.
A run always starts from the seeds alone (use a fresh --prefix); an interrupted run is continued
with --resume.

Usage: see hybrid_generation_job.sh.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARM_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ARM_DIR.parent))

from shared import feature_guidance as fgd  # noqa: E402
from shared.generation import (  # noqa: E402
    PHASE_BLACKBOX,
    PHASE_FEATURE_GUIDED,
    STRICT_RESUME_ARGS,
    Phase,
    Pool,
    add_generation_args,
    default_max_calls,
    new_phase_state,
    phase_counts_line,
    run_phase,
    setup_generation,
    start_run,
)
from shared.openrouter import summarize_sampling_verification  # noqa: E402
from shared.run_io import (  # noqa: E402
    append_run_log,
    format_wall_clock_slurm,
    load_checkpoint,
    load_json_list,
    save_checkpoint,
    save_json,
    utc_now,
)

ARM = "hybrid"
PHASE_TAGS = {PHASE_BLACKBOX: "bb", PHASE_FEATURE_GUIDED: "fg"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_generation_args(parser)
    parser.add_argument("--n-blackbox", type=int, required=True, help="ACCEPTED blackbox examples (phase 1).")
    parser.add_argument("--n-feature-guided", type=int, required=True, help="ACCEPTED feature-guided examples (phase 2).")
    parser.add_argument("--max-calls-blackbox", type=int, default=None, help="Cap on phase-1 calls (default: 50 x --n-blackbox).")
    parser.add_argument("--max-calls-feature-guided", type=int, default=None,
                        help="Cap on phase-2 calls (default: 50 x --n-feature-guided).")
    fgd.add_feature_args(parser)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.n_blackbox < 0 or args.n_feature_guided < 0:
        raise SystemExit("--n-blackbox and --n-feature-guided must be >= 0.")
    fgd.validate_feature_args(args)
    setup = setup_generation(args, ARM, ARM_DIR)
    max_calls = {
        PHASE_BLACKBOX: default_max_calls(args.max_calls_blackbox, args.n_blackbox),
        PHASE_FEATURE_GUIDED: default_max_calls(args.max_calls_feature_guided, args.n_feature_guided),
    }
    resolved_args = setup.resolved_args(
        n_blackbox=args.n_blackbox,
        n_feature_guided=args.n_feature_guided,
        max_calls_blackbox=max_calls[PHASE_BLACKBOX],
        max_calls_feature_guided=max_calls[PHASE_FEATURE_GUIDED],
        **fgd.feature_resolved_args(args),
    )
    checkpoint_path = setup.output_path("checkpoint")
    checkpoint = load_checkpoint(
        checkpoint_path, args.resume, resolved_args, STRICT_RESUME_ARGS + fgd.STRICT_RESUME_ARGS,
        ("n_blackbox", "n_feature_guided", "rouge_threshold", "max_calls_blackbox", "max_calls_feature_guided",
         "max_concurrent_requests", "requests_per_second"),
    )
    if checkpoint is None:
        existing = [str(setup.output_path(s)) for s in ("accepted", "bb_accepted", "fg_accepted") if setup.output_path(s).exists()]
        if existing:
            raise SystemExit(f"Output of an earlier run with --prefix {args.prefix!r} already exists ({existing}). "
                             "A hybrid run always starts from the seeds alone - use a new --prefix.")
        run_id, started_at, current = uuid.uuid4().hex, utc_now(), PHASE_BLACKBOX
        states = {PHASE_BLACKBOX: new_phase_state(started_at), PHASE_FEATURE_GUIDED: None}
    else:
        run_id, started_at, current = checkpoint["run_id"], checkpoint["started_at"], checkpoint["phase"]
        states = {PHASE_BLACKBOX: checkpoint["blackbox"], PHASE_FEATURE_GUIDED: checkpoint["feature_guided"]}

    pool = Pool.load(setup.output_path("accepted"), setup.seed_examples, started_at)
    rejected = {p: load_json_list(setup.output_path(f"{tag}_rejected")) for p, tag in PHASE_TAGS.items()}
    discarded = {p: load_json_list(setup.output_path(f"{tag}_discarded")) for p, tag in PHASE_TAGS.items()}
    failed = {p: load_json_list(setup.output_path(f"{tag}_failed")) for p, tag in PHASE_TAGS.items()}
    targets = {PHASE_BLACKBOX: args.n_blackbox, PHASE_FEATURE_GUIDED: args.n_feature_guided}
    templates = {PHASE_BLACKBOX: setup.prompts.blackbox_template, PHASE_FEATURE_GUIDED: setup.prompts.feature_guided_template}

    def make_phase(name: str) -> Phase:
        return Phase(name, states[name], targets[name], max_calls[name], templates[name],
                     rejected[name], discarded[name], failed[name], tag=PHASE_TAGS[name])

    def persist() -> None:
        pool.save(setup.output_path("accepted"))
        for name, tag in PHASE_TAGS.items():
            pool.save(setup.output_path(f"{tag}_accepted"), phase=name)
            save_json(setup.output_path(f"{tag}_rejected"), rejected[name])
            save_json(setup.output_path(f"{tag}_discarded"), discarded[name])
            save_json(setup.output_path(f"{tag}_failed"), failed[name])
        save_checkpoint(checkpoint_path, {
            "run_id": run_id, "started_at": started_at, "phase": current,
            "blackbox": states[PHASE_BLACKBOX], "feature_guided": states[PHASE_FEATURE_GUIDED],
            "resolved_args": resolved_args,
        })

    def phase_log_entry(phase: Phase, run, extra: dict) -> dict:
        state, tag = phase.state, phase.tag
        return {
            **setup.run_meta(run_id),
            "arm": f"{ARM}_{phase.name}",
            "phase": phase.name,
            **extra,
            **phase.stats(),
            "max_calls": phase.max_calls,
            "endpoint_catalog": run.endpoint_catalog.snapshot(),
            "endpoint_catalog_fetched_at": run.endpoint_catalog.fetched_at,
            "accepted_file": str(setup.output_path(f"{tag}_accepted")),
            "rejected_file": str(setup.output_path(f"{tag}_rejected")),
            "discarded_file": str(setup.output_path(f"{tag}_discarded")),
            "failed_file": str(setup.output_path(f"{tag}_failed")),
            "started_at": state["started_at"],
            "finished_at": state["finished_at"],
            "wall_clock_time": format_wall_clock_slurm(state["started_at"], state["finished_at"]),
            "calls": state["call_records"],
        }

    stop_reason = None
    with ThreadPoolExecutor(max_workers=args.max_concurrent_requests) as executor:
        run = start_run(setup, run_id, pool, executor, persist)

        bb = make_phase(PHASE_BLACKBOX)
        if current == PHASE_BLACKBOX:
            print(f"=== Phase 1: blackbox ({bb.counters['n_accepted_this_run']}/{bb.target_n} accepted so far) ===")
            stop_reason = run_phase(run, bb)
            if stop_reason is None:
                bb.state["finished_at"] = utc_now()
                append_run_log(setup.log_path("bb_log"), f"{args.prefix}_bb", str(setup.domain_dir),
                               phase_log_entry(bb, run, {}))
                current = PHASE_FEATURE_GUIDED
                persist()

        if stop_reason is None:
            if states[PHASE_FEATURE_GUIDED] is None:
                states[PHASE_FEATURE_GUIDED] = new_phase_state(utc_now())
            bb_entries = [e for e in pool.entries if e.get("phase") == PHASE_BLACKBOX]
            guidance = fgd.start_feature_guidance(args, states[PHASE_FEATURE_GUIDED], setup.seed_file, bb_entries)
            persist()  # coverage is checkpointed before the first phase-2 call
            fg = make_phase(PHASE_FEATURE_GUIDED)
            print(f"=== Phase 2: feature-guided ({fg.counters['n_accepted_this_run']}/{fg.target_n} accepted so far) ===")
            stop_reason = run_phase(run, fg, guidance)

    persist()
    if stop_reason is not None:
        print(f"[warn] {current} phase stopped without reaching its target ({stop_reason}); checkpoint retained at "
              f"{checkpoint_path}. Run again with --resume (e.g. with a higher call cap) to continue.", file=sys.stderr)
        return

    finished_at = fg.state["finished_at"] = utc_now()
    append_run_log(setup.log_path("fg_log"), f"{args.prefix}_fg", str(setup.domain_dir), phase_log_entry(
        fg, run, {"n_blackbox_examples": bb.counters["n_accepted_this_run"], **guidance.log_fields(),
                  "feature_stats": fg.state["feature_stats"]},
    ))
    phases = (bb, fg)
    prompt_tokens = sum(p.counters["total_prompt_tokens"] for p in phases)
    completion_tokens = sum(p.counters["total_completion_tokens"] for p in phases)
    append_run_log(setup.log_path(), args.prefix, str(setup.domain_dir), {
        **setup.run_meta(run_id),
        "n_requested_blackbox": bb.target_n,
        "n_requested_feature_guided": fg.target_n,
        "n_accepted_blackbox": bb.counters["n_accepted_this_run"],
        "n_accepted_feature_guided": fg.counters["n_accepted_this_run"],
        "n_accepted": sum(p.counters["n_accepted_this_run"] for p in phases),
        "n_rejected_blackbox": bb.counters["n_rejected_this_run"],
        "n_rejected_feature_guided": fg.counters["n_rejected_this_run"],
        "n_rejected": sum(p.counters["n_rejected_this_run"] for p in phases),
        "n_discarded_blackbox": bb.counters["n_discarded_this_run"],
        "n_discarded_feature_guided": fg.counters["n_discarded_this_run"],
        "n_discarded": sum(p.counters["n_discarded_this_run"] for p in phases),
        "n_failed_calls_blackbox": bb.counters["n_failed_calls"],
        "n_failed_calls_feature_guided": fg.counters["n_failed_calls"],
        "n_calls_blackbox": bb.counters["n_calls"],
        "n_calls_feature_guided": fg.counters["n_calls"],
        "n_relevant_features": len(guidance.features),
        "n_covered_before_feature_guided": len(guidance.schedule["covered"]),
        "threshold": args.threshold,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "sae_total": guidance.sae_totals(),
        "sampling_verification_summary": summarize_sampling_verification(
            bb.state["call_records"] + fg.state["call_records"]),
        "accepted_file": str(setup.output_path("accepted")),
        "bb_accepted_file": str(setup.output_path("bb_accepted")),
        "fg_accepted_file": str(setup.output_path("fg_accepted")),
        "bb_log_file": str(setup.log_path("bb_log")),
        "fg_log_file": str(setup.log_path("fg_log")),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_clock_time": format_wall_clock_slurm(started_at, finished_at),
    })
    checkpoint_path.unlink(missing_ok=True)
    print(f"Done: {bb.counters['n_accepted_this_run']} blackbox + {fg.counters['n_accepted_this_run']} feature-guided "
          f"accepted over {bb.counters['n_calls']} + {fg.counters['n_calls']} call(s); "
          f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
    print(f"  blackbox:       {phase_counts_line(bb)}")
    print(f"  feature-guided: {phase_counts_line(fg)}")
    guidance.print_sae_summary()


if __name__ == "__main__":
    main()
