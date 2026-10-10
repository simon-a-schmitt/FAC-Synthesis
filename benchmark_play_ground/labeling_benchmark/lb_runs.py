"""Runs of the labeling benchmark and every path/parameter of one, derived from its run_id + the
configs alone (as experiments/runs.py does for experiments).

run_ids:
    every run_id of experiments/runs.py expand() (plain, icl, gold, bb, fg, hybrid - same names,
    same threshold tags), evaluated here with free generation on the model experiments built:
        plain             base model
        icl               base model + the run's seed examples as few-shot turns (icl_k)
        gold/bb/fg/hybrid base model + LoRA adapter experiments/runs/<run_id>/lora (needs ft done there)
    {bench}__api__{model}  OpenRouter model (config.yaml api_models), plain prompt
experiments is only read: its run_ids, resolve() and stage checks; nothing there is written.

Outputs per run under labeling_benchmark/runs/<run_id>/:
    raw_outputs.jsonl   one line per example (raw model text, lb_fingerprint); resumable
    predictions.jsonl   exact + lenient parse per example, rewritten on every run
    summary.json        metrics of both parsers
    lbench.done         done marker with lb_fingerprint, written last

lb_fingerprint: everything that shapes an answer - benchmark, mode, test TSV + max_prompts,
max_new_tokens, sha256 of the system prompt and user prefix, icl_k + sha256 of the few-shot TSV,
ft_fingerprint of the adapter (experiments/runs.py; a retrained adapter makes the run stale), api:
model id + request parameters. batch_size is not part of it.

Usage:
    python benchmark_play_ground/labeling_benchmark/lb_runs.py list [--bench B] [--arm A] [--setup S] [--seed-set K] [--group G]
    python benchmark_play_ground/labeling_benchmark/lb_runs.py show <run_id> [--format json|shell]
    python benchmark_play_ground/labeling_benchmark/lb_runs.py state <run_id>   (done|stale|missing|waiting_ft; exit 0 iff done)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

LB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LB_DIR.parents[1]
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
LB_CONFIG = LB_DIR / "config.yaml"
LB_RUNS_DIR = LB_DIR / "runs"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "data_synthesis"))
# experiments/ goes to the END of sys.path, so its runs/status/submit never shadow modules here.
sys.path.append(str(EXPERIMENTS_DIR))
import runs as exp_runs  # noqa: E402  experiments/runs.py (puts data_synthesis/ and the repo root on sys.path)
from stages import bench_state as exp_bench_state, ft_state as exp_ft_state  # noqa: E402

from benchmark_play_ground.api_prediction import API_BASE_PARAMS, API_MODEL_PRESETS  # noqa: E402
from prompts import load_labeling_prompt  # noqa: E402
from shared.openrouter import build_model_params  # noqa: E402

ARMS = (*exp_runs.ARMS, "api")
FT_ARMS = tuple(arm for arm, stages in exp_runs.STAGES.items() if "ft_bench" in stages)
API_RUN_ID_RE = re.compile(rf"^(?P<bench>{exp_runs._NAME})__api__(?P<api_model>{exp_runs._NAME})$")
# Bump when the generation or parsing of a run changes in a way the fingerprint fields do not show.
LB_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LbSpec:
    """An experiments RunSpec (exp) or an api run (bench + api_model)."""
    bench: str
    arm: str
    exp: exp_runs.RunSpec | None = None
    api_model: str | None = None

    @property
    def run_id(self) -> str:
        return self.exp.run_id if self.exp else f"{self.bench}__api__{self.api_model}"

    def __getattr__(self, name):  # setup, seed_set, group, n_bb, n_fg, ... of the experiments spec
        exp = self.__dict__.get("exp")
        if exp is None:
            if name in exp_runs.RunSpec.__dataclass_fields__:
                return None
            raise AttributeError(name)
        return getattr(exp, name)


def load_configs() -> tuple[dict, dict]:
    """(experiments config, labeling-benchmark config)."""
    exp_config = exp_runs.load_config()
    lb_config = yaml.safe_load(LB_CONFIG.read_text(encoding="utf-8"))
    unknown = [m for m in lb_config["api_models"] if m not in API_MODEL_PRESETS]
    if unknown:
        raise ValueError(f"api_models {unknown} unknown to api_prediction.API_MODEL_PRESETS.")
    missing = [b for b in exp_config["benchmarks"] if b not in lb_config["batch_size"]]
    if missing:
        raise ValueError(f"{LB_CONFIG}: no batch_size for {missing}.")
    return exp_config, lb_config


def parse_run_id(run_id: str) -> LbSpec:
    m = API_RUN_ID_RE.match(run_id)
    if m:
        return LbSpec(m["bench"], "api", api_model=m["api_model"])
    exp = exp_runs.parse_run_id(run_id)
    return LbSpec(exp.bench, exp.arm, exp=exp)


def expand(exp_config: dict, lb_config: dict) -> list[LbSpec]:
    """Per benchmark: the api runs first, then the experiments runs in their order."""
    specs = []
    exp_specs = exp_runs.expand(exp_config)
    for bench in exp_config["benchmarks"]:
        specs += [LbSpec(bench, "api", api_model=m) for m in lb_config["api_models"]]
        specs += [LbSpec(s.bench, s.arm, exp=s) for s in exp_specs if s.bench == bench]
    return specs


def select(exp_config: dict, lb_config: dict, run_ids: list[str] | None = None, **filters) -> list[LbSpec]:
    if run_ids:
        return [parse_run_id(r) for r in run_ids]
    return [s for s in expand(exp_config, lb_config)
            if all(v is None or getattr(s, k) == v for k, v in filters.items())]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def api_model_params(api_model: str, max_new_tokens: int) -> dict:
    """The request parameters ApiPredictor sends (built the same way, without an API key);
    run_lbench.py checks that both agree."""
    model_id, provider = API_MODEL_PRESETS[api_model]
    no_overrides = argparse.Namespace(temperature=None, top_p=None, max_tokens=None, extra_params=None)
    params = build_model_params(model_id, provider, dict(API_BASE_PARAMS, max_tokens=max_new_tokens), no_overrides)
    params["provider"]["allow_fallbacks"] = True
    return params


def resolve_full(spec: LbSpec, exp_config: dict, lb_config: dict) -> tuple[dict, dict | None]:
    """(r, exp_r): r = all paths/parameters of the labeling-benchmark run (str values), exp_r = the
    experiments resolve() of the run (None for api)."""
    if spec.bench not in exp_config["benchmarks"]:
        raise ValueError(f"Benchmark {spec.bench!r} not in {exp_runs.DEFAULT_CONFIG}.")
    b = exp_config["benchmarks"][spec.bench]
    run_dir = LB_RUNS_DIR / spec.run_id
    prompt = load_labeling_prompt(spec.bench)
    r: dict = {
        "run_id": spec.run_id,
        "bench": spec.bench,
        "arm": spec.arm,
        "lb_run_dir": run_dir,
        "raw_jsonl": run_dir / "raw_outputs.jsonl",
        "predictions_jsonl": run_dir / "predictions.jsonl",
        "summary_json": run_dir / "summary.json",
        "lb_done": run_dir / "lbench.done",
        "test_tsv": PROJECT_DIR / b["test_tsv"],
        "max_prompts": b["max_prompts"],
        "max_new_tokens": lb_config["max_new_tokens"],
        "batch_size": lb_config["batch_size"][spec.bench],
        "resource": "cpu" if spec.arm == "api" else "gpu",
    }
    fields = {
        "version": LB_FORMAT_VERSION, "bench": spec.bench, "test_tsv": b["test_tsv"], "max_prompts": b["max_prompts"],
        "max_new_tokens": r["max_new_tokens"], "system_prompt_sha256": sha256_text(prompt.system),
        "user_prefix": prompt.user_prefix,
    }
    exp_r = None
    if spec.arm == "api":
        if spec.api_model not in lb_config["api_models"]:
            raise ValueError(f"api model {spec.api_model!r} not in {LB_CONFIG} api_models.")
        params = api_model_params(spec.api_model, r["max_new_tokens"])
        r.update(mode="api", api_model=spec.api_model, api_model_id=API_MODEL_PRESETS[spec.api_model][0],
                 api_max_concurrent=lb_config["api_max_concurrent_requests"])
        fields.update(mode="api", model_id=r["api_model_id"], model_params=params)
    else:
        exp_r = exp_runs.resolve(spec.exp, exp_config)
        mode = "fine_tuned" if spec.arm in FT_ARMS else spec.arm
        r.update(mode=mode, base_model=exp_r["base_model"], exp_run_dir=exp_r["run_dir"],
                 exp_bench_jsonl=exp_r["bench_jsonl"])
        for key in ("setup", "seed_set", "seed_group", "n_bb", "n_fg", "rouge_threshold", "activation_threshold",
                    "threshold_tag"):
            if key in exp_r:
                r[key] = exp_r[key]
        fields.update(mode=mode, base_model=exp_config["global"]["base_model"])
        if mode == "icl":
            r.update(icl_k=exp_r["icl_k"], few_shot_tsv=exp_r["few_shot_tsv"])
            fields.update(icl_k=exp_r["icl_k"], few_shot_sha256=exp_runs.file_sha256(exp_r["few_shot_tsv"]))
        if mode == "fine_tuned":
            r.update(lora_dir=exp_r["lora_dir"], ft_fingerprint=exp_r["ft_fingerprint"])
            fields.update(ft_fingerprint=exp_r["ft_fingerprint"])
    r["lb_fingerprint"] = hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()
    r["lb_fingerprint_fields"] = json.dumps(fields, sort_keys=True)
    return {k: str(v) if isinstance(v, Path) else v for k, v in r.items()}, exp_r


def resolve(spec: LbSpec, exp_config: dict, lb_config: dict) -> dict:
    return resolve_full(spec, exp_config, lb_config)[0]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def marker_fingerprint(path: str) -> str | None:
    """lb_fingerprint of a done marker ("" if unreadable), None if there is none."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("lb_fingerprint") or ""
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return ""


def prerequisite(r: dict, exp_r: dict | None) -> str | None:
    """None if the run can be evaluated now; else the reason (fine_tuned: experiments ft not done)."""
    if r["mode"] == "fine_tuned":
        state, _ = exp_ft_state(exp_r)
        if state != "done":
            return f"experiments ft {state}"
    return None


def lb_state(r: dict, exp_r: dict | None) -> str:
    """done | stale (done under another fingerprint) | missing | waiting_ft (adapter not ready)."""
    stored = marker_fingerprint(r["lb_done"])
    if stored is not None and stored == r["lb_fingerprint"]:
        return "done"
    if prerequisite(r, exp_r):
        return "waiting_ft"
    return "missing" if stored is None else "stale"


def experiments_bench_done(exp_r: dict | None) -> bool:
    return exp_r is not None and exp_bench_state(exp_r)[0] == "done"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def to_shell(resolved: dict) -> str:
    return "\n".join(f"{k.upper()}={shlex.quote(str(v))}" for k, v in resolved.items())


def add_filter_args(parser: argparse.ArgumentParser, run_ids: bool = True) -> None:
    if run_ids:
        parser.add_argument("run_ids", nargs="*", help="Explicit run_ids (instead of filters).")
    parser.add_argument("--bench")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--setup")
    parser.add_argument("--seed-set")
    parser.add_argument("--group")


def filters_of(args: argparse.Namespace) -> dict:
    return {"bench": args.bench, "arm": args.arm, "setup": args.setup, "seed_set": args.seed_set, "group": args.group}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    add_filter_args(sub.add_parser("list", help="run_ids, one per line"), run_ids=False)
    p_show = sub.add_parser("show", help="all paths/parameters of one run")
    p_show.add_argument("run_id")
    p_show.add_argument("--format", choices=("json", "shell"), default="json")
    sub.add_parser("state", help="done|stale|missing|waiting_ft").add_argument("run_id")
    args = parser.parse_args()

    exp_config, lb_config = load_configs()
    if args.command == "list":
        for spec in select(exp_config, lb_config, **filters_of(args)):
            print(spec.run_id)
        return
    try:
        r, exp_r = resolve_full(parse_run_id(args.run_id), exp_config, lb_config)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    if args.command == "show":
        print(to_shell(r) if args.format == "shell" else json.dumps(r, indent=2))
    else:
        state = lb_state(r, exp_r)
        print(state)
        sys.exit(0 if state == "done" else 1)


if __name__ == "__main__":
    main()
