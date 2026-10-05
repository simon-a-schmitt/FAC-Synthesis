"""Projects the OpenRouter cost (USD) and tokens of all synthetic runs (bb / fg / hybrid, every
benchmark, setup, seed group) from the logged calls of finished runs. Every logged call carries the
cost OpenRouter actually billed (usage.cost), so the rates below are measured, not list prices.

Rates (per accepted / labeled example) per (benchmark, phase, model):
  phase       bb, fg (standalone arms), hybrid_bb, hybrid_fg (hybrid phase logs), label
  source      "current"  runs of the current generation config (gen_fingerprint in the log, e.g. a pilot)
              "legacy"   older runs of the same benchmark/phase/model (other sampling, e.g. T=1.0)
              "scaled"   no run of this benchmark: the toxicity_detection rate x the benchmark's cost ratio
                         to toxicity_detection, measured on the phase "bb" (gen) / "label" (labeling;
                         "bb" if the benchmark has no labeling run yet) of the models measured on both
Projection: bb/fg n_synthetic x rate, hybrid n_bb x rate(hybrid_bb) + n_fg x rate(hybrid_fg),
labeling n_synthetic x rate(label, label model of the setup); summed over all runs of experiments.yaml.

Usage:
    python experiments/cost_projection.py            # -> experiments/results/cost_projection.md
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from runs import GEN_ARMS, PROJECT_DIR, expand, load_config, resolve
from shared.openrouter import GENERATION_MODEL_PRESETS

DS = PROJECT_DIR / "data_synthesis"
OUT = PROJECT_DIR / "experiments" / "results" / "cost_projection.md"
REF_BENCH = "toxicity_detection"


def label_model_ids() -> dict[str, str]:
    """label_model preset -> model id (data_synthesis/labeling/run_labeling.py MODEL_PRESETS)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_labeling", DS / "labeling" / "run_labeling.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses in the module look themselves up there
    spec.loader.exec_module(module)
    return {k: v[0] for k, v in module.MODEL_PRESETS.items()}


def observations() -> list[dict]:
    """One record per logged run with calls: bench, phase, model_id, n (accepted/labeled), cost, tokens."""
    obs = []
    sources = [("blackbox", "bb"), ("feature_guided", "fg"), ("hybrid", None), ("labeling", "label")]
    for arm_dir, phase in sources:
        for log in sorted((DS / arm_dir).glob("*/log/*_log.json")):
            if arm_dir == "hybrid":
                if log.name.endswith("_bb_log.json"):
                    phase_ = "hybrid_bb"
                elif log.name.endswith("_fg_log.json"):
                    phase_ = "hybrid_fg"
                else:
                    continue  # main log: no calls of its own
            else:
                phase_ = phase
            data = json.loads(log.read_text(encoding="utf-8"))
            for run in data.get("runs", []):
                calls = run.get("calls") or []
                n = run.get("n_to_label") if phase_ == "label" else run.get("n_accepted")
                if not calls or not n or n < 50:
                    continue
                usage = [((c.get("openrouter_response") or {}).get("usage") or {}) for c in calls]
                obs.append({
                    "bench": log.parents[1].name, "phase": phase_, "model_id": run.get("model_id"), "n": n,
                    "cost": sum(u.get("cost") or 0.0 for u in usage),
                    "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
                    "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
                    "current": bool(run.get("gen_fingerprint")) or (phase_ == "label" and "__" in log.name),
                    "log": log.name,
                })
    return obs


def rates(obs: list[dict]) -> dict[tuple, dict]:
    """(bench, phase, model_id) -> per-example cost/tokens from current runs if any, else legacy."""
    grouped: dict[tuple, dict[bool, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for o in obs:
        grouped[(o["bench"], o["phase"], o["model_id"])][o["current"]].append(o)
    out = {}
    for key, by_cur in grouped.items():
        runs = by_cur.get(True) or by_cur.get(False)
        n = sum(o["n"] for o in runs)
        out[key] = {"cost": sum(o["cost"] for o in runs) / n,
                    "prompt_tokens": sum(o["prompt_tokens"] for o in runs) / n,
                    "completion_tokens": sum(o["completion_tokens"] for o in runs) / n,
                    "source": "current" if by_cur.get(True) else "legacy", "n_runs": len(runs)}
    return out


def bench_factor(rate_table: dict, bench: str, phase: str) -> float | None:
    """Cost ratio bench / REF_BENCH on `phase`, median over the models measured on both."""
    ratios = [r["cost"] / rate_table[(REF_BENCH, ph, m)]["cost"]
              for (b, ph, m), r in rate_table.items()
              if b == bench and ph == phase and (REF_BENCH, ph, m) in rate_table and rate_table[(REF_BENCH, ph, m)]["cost"]]
    return statistics.median(ratios) if ratios else None


def lookup(rate_table: dict, bench: str, phase: str, model_id: str) -> dict | None:
    if (bench, phase, model_id) in rate_table:
        return rate_table[(bench, phase, model_id)]
    ref = rate_table.get((REF_BENCH, phase, model_id))
    basis = "label" if phase == "label" else "bb"
    factor = bench_factor(rate_table, bench, basis)
    if factor is None and basis == "label":  # no labeling run of the benchmark yet: generation ratio
        basis, factor = "bb", bench_factor(rate_table, bench, "bb")
    if ref is None or factor is None:
        return None
    return {k: (v * factor if k != "source" and k != "n_runs" else v) for k, v in ref.items()} | {
        "source": f"scaled x{factor:.2f} ({basis})"}


def main() -> None:
    config = load_config()
    label_ids = label_model_ids()
    rate_table = rates(observations())

    totals: dict[tuple, dict] = defaultdict(lambda: {"cost": 0.0, "prompt_tokens": 0.0, "completion_tokens": 0.0,
                                                     "runs": 0, "sources": set()})
    missing = set()
    for spec in expand(config):
        if spec.arm not in GEN_ARMS:
            continue
        r = resolve(spec, config)
        gen_model = GENERATION_MODEL_PRESETS[r["gen_model"]][0]
        n_syn = (spec.n_bb + spec.n_fg) if spec.arm == "hybrid" else r["n_synthetic"]
        parts = ([("hybrid_bb", gen_model, spec.n_bb), ("hybrid_fg", gen_model, spec.n_fg)] if spec.arm == "hybrid"
                 else [(spec.arm, gen_model, n_syn)])
        parts.append(("label", label_ids[r["label_model"]], n_syn))
        for phase, model_id, n in parts:
            rate = lookup(rate_table, spec.bench, phase, model_id)
            key = (spec.bench, spec.arm, spec.setup, phase)
            if rate is None:
                missing.add((spec.bench, phase, model_id))
                continue
            t = totals[key]
            t["runs"] += 1
            t["sources"].add(rate["source"])
            for k in ("cost", "prompt_tokens", "completion_tokens"):
                t[k] += n * rate[k]

    lines = ["# Cost projection (OpenRouter, USD)", "",
             "All synthetic runs of experiments.yaml (bb / fg / hybrid x setups x seed groups), per benchmark, "
             "arm, setup and phase. Rates per example from the billed cost of logged calls (usage.cost); "
             "source: current = runs of the current generation config, legacy = older runs of the benchmark "
             "(e.g. T=1.0), scaled = toxicity_detection rate x the benchmark's measured cost ratio.", "",
             "| bench | arm | setup | phase | runs | prompt tok (M) | completion tok (M) | cost USD | rate source |",
             "|---|---|---|---|---|---|---|---|---|"]
    grand = 0.0
    by_bench: dict[str, float] = defaultdict(float)
    for (bench, arm, setup, phase), t in sorted(totals.items()):
        grand += t["cost"]
        by_bench[bench] += t["cost"]
        lines.append(f"| {bench} | {arm} | {setup} | {phase} | {t['runs']} | {t['prompt_tokens'] / 1e6:.2f} | "
                     f"{t['completion_tokens'] / 1e6:.2f} | {t['cost']:.2f} | {', '.join(sorted(t['sources']))} |")
    lines += ["", "| bench | cost USD |", "|---|---|"]
    lines += [f"| {b} | {c:.2f} |" for b, c in by_bench.items()]
    lines += [f"| **total** | **{grand:.2f}** |", ""]
    lines += ["Measured rates (USD per 1000 examples):", "", "| bench | phase | model | USD/1k | runs | source |",
              "|---|---|---|---|---|---|"]
    for (bench, phase, model), r in sorted(rate_table.items()):
        lines.append(f"| {bench} | {phase} | {model} | {r['cost'] * 1000:.3f} | {r['n_runs']} | {r['source']} |")
    if missing:
        lines += ["", "No rate (not projected): " + ", ".join(f"{b}/{p}/{m}" for b, p, m in sorted(missing))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
