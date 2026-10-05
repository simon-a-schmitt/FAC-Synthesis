"""Checks of the label_build stage (experiments/slurm/label_build_job.sh) for a bb/fg/hybrid run.
Exit 0 = check passed, 1 = failed (reason on stdout).

  gen-done <run_id>   generation finished: GEN_ACCEPTED_JSON exists, GEN_CHECKPOINT_JSON does not,
                      GEN_LOG_JSON has prefix == run_id and at least one completed run, and the
                      accepted pool holds exactly n_synthetic synthetic entries (hybrid: n_blackbox +
                      n_feature_guided, and per phase n_blackbox / n_feature_guided) after its seeds.
  gen-state <run_id>  prints the generation state of the run (for experiments/slurm/gen_job.sh):
                      done (gen-done passes), resume (GEN_CHECKPOINT_JSON exists), fresh (no output
                      at all) or partial (outputs but neither done nor a checkpoint - exit 1).
  labels <run_id>     LABEL_TSV is complete: every seed + synthetic text of the accepted pool has a
                      non-empty label and the TSV has n_total rows. Missing ones are listed by their
                      synthetic index (position among the synthetic entries of the accepted pool).

labels also fails if more than MAX_MAJORITY_FALLBACK_RATE of the synthetic examples got their
label by run_labeling.py's majority fallback (summed over all runs in LABEL_LOG_JSON): those rows
look complete, and a rerun keeps them - run_labeling.py only re-requests blank labels.
  fallbacks <run_id>  prints only that count (for run_meta.json).

Usage:
    python experiments/label_build_checks.py gen-done <run_id>
    python experiments/label_build_checks.py gen-state <run_id>
    python experiments/label_build_checks.py labels <run_id>
    python experiments/label_build_checks.py fallbacks <run_id>
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("check", choices=("gen-done", "gen-state", "labels", "fallbacks"))
    parser.add_argument("run_id")
    args = parser.parse_args()

    config = load_config()
    try:
        r = resolve(parse_run_id(args.run_id), config)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    if r["arm"] not in GEN_ARMS:
        raise SystemExit(f"error: {args.run_id} is not a generation run (arm {r['arm']!r}).")

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
        errors = check_labels(r, config["global"]["n_total"])
        ok_msg = f"labels complete ({config['global']['n_total']} rows in {r['label_tsv']})"
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
