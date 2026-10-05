"""State of every stage of a run, read off the file system - the single implementation of the done
checks, used by the job scripts (experiments/slurm/*.sh, via the CLI below) and by
experiments/status.py / experiments/submit.py.

  stage        applies to          done when
  gen          bb / fg / hybrid    GEN_ACCEPTED_JSON exists, GEN_CHECKPOINT_JSON does not,
                                   GEN_LOG_JSON has prefix == run_id and a completed run, and the
                                   accepted pool holds the run's seeds + exactly n_synthetic synthetic
                                   entries (hybrid: n_blackbox + n_feature_guided, also per phase)
  label_build  bb / fg / hybrid    LLaMA-Factory dataset prepared and passes the gate
                                   (experiments/prepare_lf_dataset.py is_prepared)
  ft           gold / bb/fg/hybrid LORA_DIR/adapter_model.safetensors + FT_LOG_JSON whose
                                   ft_fingerprint == FT_FINGERPRINT (runs.py)
  bench        all                 BENCH_DONE whose bench_fingerprint == BENCH_FINGERPRINT

stage_state(): "done", "stale" (marker written under another config), "missing", or "-" if the
stage does not apply to the arm.

The label checks also fail if more than MAX_MAJORITY_FALLBACK_RATE of the synthetic examples got
their label by run_labeling.py's majority fallback (summed over all runs in LABEL_LOG_JSON): those
rows look complete, and a rerun keeps them - run_labeling.py only re-requests blank labels.

CLI (exit 0 = check passed; reasons on stdout, the state word for *-state on stdout):
  gen-done <run_id>          generation complete
  gen-state <run_id>         done | resume (checkpoint exists) | fresh (no output) | partial
                             (outputs but neither done nor a checkpoint; exit 1)
  labels <run_id>            LABEL_TSV complete (n_total rows, every text labeled, fallback rate)
  fallbacks <run_id>         number of majority-fallback labels
  label-build-done <run_id>  dataset prepared and gate passed
  ft-state <run_id>          done | stale | missing; exit 0 iff done
  bench-state <run_id>       done | stale | missing; exit 0 iff done
  bench-resumable <run_id>   exit 0 if BENCH_JSONL is absent or was started (BENCH_STARTED) under the
                             current bench fingerprint, i.e. --resume may continue it
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from runs import GEN_ARMS, load_config, parse_run_id, resolve

MAX_MAJORITY_FALLBACK_RATE = 0.02
PHASES = {"blackbox": "n_blackbox", "feature_guided": "n_feature_guided"}  # hybrid entry "phase" -> resolve key
STAGE_NAMES = ("gen", "label_build", "ft", "bench")


# ---------------------------------------------------------------------------
# gen
# ---------------------------------------------------------------------------

def load_accepted(r: dict) -> tuple[list[dict], list[dict]]:
    entries = json.loads(Path(r["gen_accepted_json"]).read_text(encoding="utf-8"))
    return [e for e in entries if e.get("type") == "seed"], [e for e in entries if e.get("type") != "seed"]


def check_gen_done(r: dict) -> list[str]:
    accepted, checkpoint, log = (Path(r[k]) for k in ("gen_accepted_json", "gen_checkpoint_json", "gen_log_json"))
    if not accepted.is_file():
        return [f"generation output {accepted} does not exist (generation not run yet)"]
    errors = []
    if checkpoint.exists():
        errors.append(f"checkpoint {checkpoint} exists (generation incomplete; continue it with --resume)")
    if not log.is_file():
        errors.append(f"generation log {log} does not exist")
    else:
        data = json.loads(log.read_text(encoding="utf-8"))
        if data.get("prefix") != r["run_id"] or not data.get("runs"):
            errors.append(f"generation log {log} has no completed run for prefix {r['run_id']!r} "
                          f"(prefix={data.get('prefix')!r}, runs={len(data.get('runs', []))})")

    seeds, synthetic = load_accepted(r)
    if len(seeds) != r["seed_n_examples"]:
        errors.append(f"{len(seeds)} seed entries in {accepted.name}, expected {r['seed_n_examples']}")
    if r["arm"] == "hybrid":
        expected = r["n_blackbox"] + r["n_feature_guided"]
        per_phase = Counter(e.get("phase") for e in synthetic)
        for phase, key in PHASES.items():
            if per_phase.get(phase, 0) != r[key]:
                errors.append(f"{per_phase.get(phase, 0)} synthetic entries of phase {phase!r}, expected {r[key]}")
        unknown = {p: n for p, n in per_phase.items() if p not in PHASES}
        if unknown:
            errors.append(f"synthetic entries with unknown phase: {unknown}")
    else:
        expected = r["n_synthetic"]
    if len(synthetic) != expected:
        errors.append(f"{len(synthetic)} synthetic entries in {accepted.name}, expected {expected}")
    return errors


def gen_state(r: dict) -> tuple[str, list[str]]:
    if Path(r["gen_checkpoint_json"]).exists():
        return "resume", []
    outputs = [r[k] for k in r if k.startswith("gen_") and k.endswith("_json") and Path(r[k]).exists()]
    if not outputs:
        return "fresh", []
    errors = check_gen_done(r)
    if not errors:
        return "done", []
    return "partial", errors + [f"existing outputs without checkpoint: {', '.join(Path(o).name for o in outputs)}"]


# ---------------------------------------------------------------------------
# label_build
# ---------------------------------------------------------------------------

def majority_fallbacks(r: dict) -> int:
    log = Path(r["label_log_json"])
    if not log.is_file():
        return 0
    return sum(run.get("n_majority_fallback") or 0 for run in json.loads(log.read_text(encoding="utf-8"))["runs"])


def check_labels(r: dict, n_total: int) -> list[str]:
    tsv = Path(r["label_tsv"])
    if not tsv.is_file():
        return [f"label TSV {tsv} does not exist"]
    with open(tsv, "r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.reader(f, delimiter="\t") if row and row[0]]
    labeled = {row[0] for row in rows if len(row) > 1 and row[1].strip()}

    seeds, synthetic = load_accepted(r)
    errors = []
    missing_seeds = [i for i, e in enumerate(seeds) if e["text"] not in labeled]
    if missing_seeds:
        errors.append(f"{len(missing_seeds)} seed(s) without label (seed index {missing_seeds})")
    missing = [i for i, e in enumerate(synthetic) if e["text"] not in labeled]
    if missing:
        shown = ", ".join(map(str, missing[:50])) + (", ..." if len(missing) > 50 else "")
        errors.append(f"{len(missing)} synthetic example(s) without label (synthetic index {shown})")
    if len(rows) != n_total:
        errors.append(f"{len(rows)} rows in {tsv.name}, expected n_total={n_total}")
    n_fallback = majority_fallbacks(r)
    if synthetic and n_fallback / len(synthetic) > MAX_MAJORITY_FALLBACK_RATE:
        errors.append(f"{n_fallback}/{len(synthetic)} synthetic labels set by majority fallback "
                      f"({n_fallback / len(synthetic):.1%} > {MAX_MAJORITY_FALLBACK_RATE:.0%}; see {r['label_log_json']})")
    return errors


def label_build_done(r: dict, n_total: int, verbose: bool = False) -> bool:
    from prepare_lf_dataset import is_prepared  # tokenizer etc. only when needed
    return is_prepared(r, n_total, verbose=verbose)


# ---------------------------------------------------------------------------
# ft / bench (fingerprinted done markers)
# ---------------------------------------------------------------------------

def marker_fingerprint(path: str, key: str) -> str | None:
    """The fingerprint a JSON done marker stored ("" if it has none); None if there is no marker."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get(key) or ""
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return ""


def ft_state(r: dict) -> tuple[str, str]:
    stored = marker_fingerprint(r["ft_log_json"], "ft_fingerprint")
    if stored is None or not (Path(r["lora_dir"]) / "adapter_model.safetensors").is_file():
        return "missing", ""
    if stored != r["ft_fingerprint"]:
        return "stale", f"ft_fingerprint in {r['ft_log_json']}: {stored!r}, current: {r['ft_fingerprint']!r}"
    return "done", ""


def bench_state(r: dict) -> tuple[str, str]:
    stored = marker_fingerprint(r["bench_done"], "bench_fingerprint")
    if stored is None:
        return "missing", ""
    if stored != r["bench_fingerprint"]:
        return "stale", f"bench_fingerprint in {r['bench_done']}: {stored!r}, current: {r['bench_fingerprint']!r}"
    return "done", ""


def bench_resumable(r: dict) -> bool:
    return (not Path(r["bench_jsonl"]).exists()
            or marker_fingerprint(r["bench_started"], "bench_fingerprint") == r["bench_fingerprint"])


# ---------------------------------------------------------------------------
# All stages of a run
# ---------------------------------------------------------------------------

def applies(r: dict, stage: str) -> bool:
    return {"gen": "gen" in r["stages"], "label_build": "label_build" in r["stages"],
            "ft": "ft_bench" in r["stages"], "bench": True}[stage]


def stage_state(r: dict, stage: str, n_total: int) -> str:
    if not applies(r, stage):
        return "-"
    if stage == "gen":
        return "done" if gen_state(r)[0] == "done" else "missing"
    if stage == "label_build":
        return "done" if label_build_done(r, n_total) else "missing"
    if stage == "ft":
        return ft_state(r)[0]
    return bench_state(r)[0]


def run_states(r: dict, n_total: int) -> dict[str, str]:
    return {stage: stage_state(r, stage, n_total) for stage in STAGE_NAMES}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CHECKS = ("gen-done", "gen-state", "labels", "fallbacks", "label-build-done", "ft-state", "bench-state",
          "bench-resumable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("check", choices=CHECKS)
    parser.add_argument("run_id")
    args = parser.parse_args()

    config = load_config()
    n_total = config["global"]["n_total"]
    try:
        r = resolve(parse_run_id(args.run_id), config)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    needs = {"gen-done": "gen", "gen-state": "gen", "labels": "label_build", "fallbacks": "label_build",
             "label-build-done": "label_build", "ft-state": "ft"}.get(args.check)
    if needs and not applies(r, needs):
        raise SystemExit(f"error: stage {needs!r} does not apply to {args.run_id} (arm {r['arm']!r}).")

    if args.check in ("ft-state", "bench-state"):
        state, detail = (ft_state if args.check == "ft-state" else bench_state)(r)
        if detail:
            print(f"[{args.check}] {args.run_id}: {detail}", file=sys.stderr)
        print(state)
        sys.exit(0 if state == "done" else 1)
    if args.check == "bench-resumable":
        sys.exit(0 if bench_resumable(r) else 1)
    if args.check == "label-build-done":
        sys.exit(0 if label_build_done(r, n_total, verbose=True) else 1)
    if args.check == "gen-state":
        state, errors = gen_state(r)
        for e in errors:
            print(f"[gen-state] {args.run_id}: {e}", file=sys.stderr)
        print(state)
        sys.exit(1 if state == "partial" else 0)
    if args.check == "fallbacks":
        print(majority_fallbacks(r))
        return
    if args.check == "gen-done":
        errors = check_gen_done(r)
        ok_msg = f"generation complete ({r['gen_accepted_json']})"
    else:
        errors = check_labels(r, n_total)
        ok_msg = f"labels complete ({n_total} rows in {r['label_tsv']})"
        n_fallback = majority_fallbacks(r)
        if n_fallback:
            print(f"[{args.check}] WARNUNG: {n_fallback} label(s) set by majority fallback "
                  f"(run_labeling.py; see {r['label_log_json']}).")
    for e in errors:
        print(f"[{args.check}] {args.run_id}: {e}")
    if errors:
        sys.exit(1)
    print(f"[{args.check}] {args.run_id}: {ok_msg}")


if __name__ == "__main__":
    main()
