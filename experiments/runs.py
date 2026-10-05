"""Expands experiments/config/experiments.yaml into runs with a canonical run_id and derives every
path and parameter of a run from its run_id + the config alone (single source of truth for the
job scripts). Nothing here stores state: whether a run is done is read off the file system later.

run_id schema (separator "__", characters [a-z0-9_-] only):
    {bench}__plain
    {bench}__icl__{seedset}-{gg}
    {bench}__gold__{seedset}-{gg}
    {bench}__bb__{setup}__{seedset}-{gg}
    {bench}__fg__{setup}__{seedset}-{gg}
    {bench}__hybrid__{setup}__{seedset}-{gg}__r{nbb}-{nfg}
e.g. claudette_tos__hybrid__llama__k10-03__r300-100

Paths of generation/labeling outputs mirror the existing logic of
data_synthesis/{blackbox,feature_guided,hybrid}/run_generation.py (shared/generation.py) and
data_synthesis/labeling/run_labeling.py, with --prefix = run_id. New per-run artifacts live under
experiments/runs/<run_id>/.

Done markers carry a config fingerprint (ft_fingerprint: FT hyperparameters, base model, ft_seed,
n_total; bench_fingerprint: bench script, test TSV, mode, icl_k, max_prompts, extra args and - for
fine_tuned - ft_fingerprint), so a marker written under another config is detected as stale.

Usage:
    python experiments/runs.py list [--bench B] [--arm A] [--setup S] [--seed-set K] [--group G]
    python experiments/runs.py show <run_id> [--format json|shell]
    eval "$(python experiments/runs.py show <run_id> --format shell)"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENTS_DIR.parent
DEFAULT_CONFIG = EXPERIMENTS_DIR / "config" / "experiments.yaml"
RUNS_DIR = EXPERIMENTS_DIR / "runs"

if os.environ.get("WS_PATH"):
    WS_PATH = Path(os.environ["WS_PATH"])
else:
    WS_PATH = PROJECT_DIR.parents[1]  # <ws>/code/FAC-Synthesis
    print(f"[runs.py] warning: $WS_PATH not set, falling back to {WS_PATH}", file=sys.stderr)

sys.path.insert(0, str(PROJECT_DIR / "data_synthesis"))
sys.path.insert(0, str(PROJECT_DIR / "benchmarks"))
from shared import seed_derivation  # noqa: E402
from shared.benchmarks import SEED_SETS, find_seed_file  # noqa: E402
from sample_gold_ft_sets import SEED_GROUP_PATTERN  # noqa: E402

ARMS = ("plain", "icl", "gold", "bb", "fg", "hybrid")
GEN_ARMS = ("bb", "fg", "hybrid")
# run_id arm -> directory under data_synthesis/ (= run_labeling.py --source)
GEN_ARM_DIRS = {"bb": "blackbox", "fg": "feature_guided", "hybrid": "hybrid"}
STAGES = {
    "plain": ["bench"],
    "icl": ["bench"],
    "gold": ["ft_bench"],
    "bb": ["gen", "label_build", "ft_bench"],
    "fg": ["gen", "label_build", "ft_bench"],
    "hybrid": ["gen", "label_build", "ft_bench"],
}
GEN_RESOURCE = {"bb": "cpu", "fg": "gpu", "hybrid": "gpu"}

SEP = "__"
_NAME = r"[a-z0-9]+(?:_[a-z0-9]+)*"  # no "__" inside a field
_SEED = r"(?P<seed_set>k\d+)-(?P<group>\d{2})"
RUN_ID_PATTERNS = {
    "plain": re.compile(rf"^(?P<bench>{_NAME})__plain$"),
    "icl": re.compile(rf"^(?P<bench>{_NAME})__icl__{_SEED}$"),
    "gold": re.compile(rf"^(?P<bench>{_NAME})__gold__{_SEED}$"),
    "bb": re.compile(rf"^(?P<bench>{_NAME})__bb__(?P<setup>{_NAME})__{_SEED}$"),
    "fg": re.compile(rf"^(?P<bench>{_NAME})__fg__(?P<setup>{_NAME})__{_SEED}$"),
    "hybrid": re.compile(rf"^(?P<bench>{_NAME})__hybrid__(?P<setup>{_NAME})__{_SEED}__r(?P<n_bb>\d+)-(?P<n_fg>\d+)$"),
}


@dataclass(frozen=True)
class RunSpec:
    bench: str
    arm: str
    setup: str | None = None
    seed_set: str | None = None
    group: str | None = None
    n_bb: int | None = None  # hybrid only
    n_fg: int | None = None  # hybrid only

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"Unknown arm {self.arm!r}; expected one of {ARMS}.")
        needs_seed = self.arm != "plain"
        needs_setup = self.arm in GEN_ARMS
        needs_ratio = self.arm == "hybrid"
        for name, value, needed in (("seed_set", self.seed_set, needs_seed), ("group", self.group, needs_seed),
                                    ("setup", self.setup, needs_setup), ("n_bb", self.n_bb, needs_ratio),
                                    ("n_fg", self.n_fg, needs_ratio)):
            if (value is not None) != needed:
                raise ValueError(f"Arm {self.arm!r}: {name} must be {'set' if needed else 'None'}, got {value!r}.")

    @property
    def run_id(self) -> str:
        return format_run_id(self)


def format_run_id(spec: RunSpec) -> str:
    parts = [spec.bench, spec.arm]
    if spec.setup is not None:
        parts.append(spec.setup)
    if spec.seed_set is not None:
        parts.append(f"{spec.seed_set}-{spec.group}")
    if spec.arm == "hybrid":
        parts.append(f"r{spec.n_bb}-{spec.n_fg}")
    run_id = SEP.join(parts)
    if parse_run_id(run_id) != spec:  # rejects field values that would not parse back
        raise ValueError(f"RunSpec {spec} does not format to a valid run_id ({run_id!r}).")
    return run_id


def parse_run_id(run_id: str) -> RunSpec:
    fields = run_id.split(SEP)
    arm = fields[1] if len(fields) > 1 else None
    match = RUN_ID_PATTERNS[arm].match(run_id) if arm in RUN_ID_PATTERNS else None
    if match is None:
        raise ValueError(f"Invalid run_id {run_id!r}.")
    g = match.groupdict()
    return RunSpec(bench=g["bench"], arm=arm, setup=g.get("setup"), seed_set=g.get("seed_set"),
                   group=g.get("group"), n_bb=int(g["n_bb"]) if "n_bb" in g else None,
                   n_fg=int(g["n_fg"]) if "n_fg" in g else None)


# ---------------------------------------------------------------------------
# Config + expansion
# ---------------------------------------------------------------------------

def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for seed_set, spec in config["global"]["seed_sets"].items():
        if seed_set not in SEED_SETS:
            raise ValueError(f"Seed set {seed_set!r} unknown to shared/benchmarks.py SEED_SETS.")
        if len(set(spec["groups"])) != len(spec["groups"]):
            raise ValueError(f"Seed set {seed_set!r} lists a group twice: {spec['groups']}.")
        for n_bb, n_fg in hybrid_splits(config, seed_set):
            if n_bb <= 0 or n_fg <= 0:
                raise ValueError(f"Seed set {seed_set!r}: hybrid split {n_bb}/{n_fg} must be > 0 on both sides.")
    return config


def n_synthetic(config: dict, seed_set: str) -> int:
    """Generated (or, for gold, drawn) examples of a training set: n_total minus its seeds."""
    return config["global"]["n_total"] - config["global"]["seed_sets"][seed_set]["n_examples"]


def hybrid_splits(config: dict, seed_set: str) -> list[tuple[int, int]]:
    """(n_bb, n_fg) of every hybrid variant of a seed set."""
    n = n_synthetic(config, seed_set)
    return [(n - n_fg, n_fg) for n_fg in config["hybrid_n_feature_guided"]]


def expand(config: dict) -> list[RunSpec]:
    """Every run of the config, in a fixed order: benchmarks, setups, seed sets, groups and ratios
    in config order; arms in ARMS order."""
    seed_sets = config["global"]["seed_sets"]
    seeds = [(s, g) for s, spec in seed_sets.items() for g in spec["groups"]]
    specs = []
    for bench in config["benchmarks"]:
        specs.append(RunSpec(bench, "plain"))
        specs += [RunSpec(bench, "icl", seed_set=s, group=g) for s, g in seeds]
        specs += [RunSpec(bench, "gold", seed_set=s, group=g) for s, g in seeds]
        for arm in ("bb", "fg"):
            specs += [RunSpec(bench, arm, setup, s, g) for setup in config["setups"] for s, g in seeds]
        specs += [RunSpec(bench, "hybrid", setup, s, g, n_bb, n_fg)
                  for setup in config["setups"] for s, g in seeds for n_bb, n_fg in hybrid_splits(config, s)]
    return specs


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def gold_tsv_path(bench: str, seed_set: str, seed_file: Path) -> Path:
    """Naming scheme of benchmarks/sample_gold_ft_sets.py: {prefix}_gold_ft_set_{infix}{id}.tsv."""
    prefix, group_id = SEED_GROUP_PATTERN.match(seed_file.name).groups()
    _, infix = SEED_SETS[seed_set]
    return PROJECT_DIR / "benchmarks" / bench / "gold_ft_sets" / f"{prefix}_gold_ft_set_{infix}{group_id}.tsv"


def generation_paths(arm_dir: str, domain: str, prefix: str) -> dict:
    """Output of <arm>/run_generation.py --prefix <prefix> (SetupGeneration.output_path/log_path).
    hybrid writes rejected/discarded/failed and an extra accepted file + log per phase (bb_, fg_)."""
    domain_dir = PROJECT_DIR / "data_synthesis" / arm_dir / domain
    out, log = domain_dir / "output", domain_dir / "log"
    paths = {
        "gen_domain_dir": domain_dir,
        "gen_accepted_json": out / f"{prefix}_accepted.json",
        "gen_checkpoint_json": out / f"{prefix}_checkpoint.json",
        "gen_log_json": log / f"{prefix}_log.json",
    }
    if arm_dir == "hybrid":
        for tag in ("bb", "fg"):
            paths.update({
                f"gen_{tag}_accepted_json": out / f"{prefix}_{tag}_accepted.json",
                f"gen_{tag}_rejected_json": out / f"{prefix}_{tag}_rejected.json",
                f"gen_{tag}_discarded_json": out / f"{prefix}_{tag}_discarded.json",
                f"gen_{tag}_failed_json": out / f"{prefix}_{tag}_failed.json",
                f"gen_{tag}_log_json": log / f"{prefix}_{tag}_log.json",
            })
    else:
        paths.update(gen_rejected_json=out / f"{prefix}_rejected.json", gen_discarded_json=out / f"{prefix}_discarded.json",
                     gen_failed_json=out / f"{prefix}_failed.json")
    return paths


def labeling_paths(domain: str, prefix: str) -> dict:
    """Output of labeling/run_labeling.py for <prefix>_accepted.json (prefix = input stem without
    "_accepted"; the output dir is per domain, not per source)."""
    label_dir = PROJECT_DIR / "data_synthesis" / "labeling" / domain
    return {
        "label_tsv": label_dir / f"{prefix}.tsv",
        "label_failed_json": label_dir / f"{prefix}_failed.json",
        "label_log_json": label_dir / "log" / f"{prefix}_log.json",
    }


def resolve(spec: RunSpec, config: dict) -> dict:
    """All paths/parameters of a run; only the keys relevant to its arm are present."""
    if spec.bench not in config["benchmarks"]:
        raise ValueError(f"Benchmark {spec.bench!r} not in config.")
    if spec.setup is not None and spec.setup not in config["setups"]:
        raise ValueError(f"Setup {spec.setup!r} not in config.")
    if spec.seed_set is not None and spec.group not in config["global"]["seed_sets"].get(spec.seed_set, {}).get("groups", []):
        raise ValueError(f"Seed group {spec.seed_set}-{spec.group} not in config.")
    if spec.arm == "hybrid" and (spec.n_bb, spec.n_fg) not in hybrid_splits(config, spec.seed_set):
        raise ValueError(f"Hybrid ratio {spec.n_bb}/{spec.n_fg} does not match seed set {spec.seed_set} "
                         f"(valid: {', '.join(f'r{a}-{b}' for a, b in hybrid_splits(config, spec.seed_set))}).")

    g, b = config["global"], config["benchmarks"][spec.bench]
    run_id = spec.run_id
    run_dir = RUNS_DIR / run_id
    is_ft = "ft_bench" in STAGES[spec.arm]

    r: dict = {
        "run_id": run_id,
        "bench": spec.bench,
        "arm": spec.arm,
        "stages": list(STAGES[spec.arm]),
        "ws_path": WS_PATH,
        "project_dir": PROJECT_DIR,
        "run_dir": run_dir,
        "run_meta_json": run_dir / "run_meta.json",
        "base_model": WS_PATH / g["base_model"],
        # benchmark
        "bench_script": PROJECT_DIR / b["bench_script"],
        "bench_mode": "fine_tuned" if is_ft else spec.arm,
        "test_tsv": PROJECT_DIR / b["test_tsv"],
        "max_prompts": b["max_prompts"],
        "bench_extra_args": list(b["bench_extra_args"]),
        "bench_jsonl": run_dir / "bench.jsonl",
    }
    if spec.seed_set is not None:
        seed_file = find_seed_file(spec.bench, spec.seed_set, spec.group)
        n_examples = g["seed_sets"][spec.seed_set]["n_examples"]
        n_found = len(seed_derivation.load_seed_texts(seed_file))
        if n_found != n_examples:
            raise ValueError(f"{seed_file} has {n_found} seed examples, seed set {spec.seed_set} expects {n_examples}.")
        r.update(seed_set=spec.seed_set, seed_group=spec.group, seed_file=seed_file, seed_n_examples=n_examples)
    if spec.arm == "icl":
        r.update(icl_k=r["seed_n_examples"], few_shot_tsv=r["seed_file"])
    if spec.arm == "gold":
        r.update(gold_tsv=gold_tsv_path(spec.bench, spec.seed_set, r["seed_file"]),
                 gold_pool_tsv=PROJECT_DIR / b["gold_pool_tsv"], gold_n=n_synthetic(config, spec.seed_set))
    if spec.arm in GEN_ARMS:
        setup = config["setups"][spec.setup]
        arm_dir = GEN_ARM_DIRS[spec.arm]
        r.update(setup=spec.setup, gen_resource=GEN_RESOURCE[spec.arm], gen_arm_dir=arm_dir,
                 gen_prefix=run_id, gen_model=setup["gen_model"], rouge_threshold=b["rouge_threshold"])
        if spec.arm == "hybrid":
            r.update(n_blackbox=spec.n_bb, n_feature_guided=spec.n_fg)
        else:
            r.update(n_synthetic=n_synthetic(config, spec.seed_set))
        if spec.arm in ("fg", "hybrid"):
            r.update(activation_threshold=b["activation_threshold"], sae_ckpt=WS_PATH / g["sae_ckpt"])
        r.update(generation_paths(arm_dir, spec.bench, run_id))
        r.update(label_source=arm_dir, label_model=setup["label_model"],
                 gen_max_concurrent=config["api"]["generation"]["max_concurrent_requests"],
                 label_max_concurrent=config["api"]["labeling"]["max_concurrent_requests"])
        r.update(labeling_paths(spec.bench, run_id))
    if is_ft:
        r.update(
            ft_seed=seed_derivation.derive_seed_from_file(spec.bench, r["seed_file"],
                                                          seed_derivation.PURPOSE_FINE_TUNING),
            lf_data_dir=run_dir / "lf_data",
            lf_dataset_name=run_id,
            lf_dataset_json=run_dir / "lf_data" / f"{run_id}.json",
            lf_dataset_info_json=run_dir / "lf_data" / "dataset_info.json",
            lora_dir=run_dir / "lora",
            ft_log_json=run_dir / "ft_log.json",
        )
        r.update({f"ft_{k}": v for k, v in g["fine_tuning"].items()})
        r["ft_fingerprint"] = fingerprint({
            "fine_tuning": g["fine_tuning"], "base_model": g["base_model"], "ft_seed": r["ft_seed"],
            "n_total": g["n_total"],
        })
    r["bench_fingerprint"] = fingerprint({
        "bench_script": b["bench_script"], "test_tsv": b["test_tsv"], "mode": r["bench_mode"],
        "icl_k": r.get("icl_k"), "max_prompts": b["max_prompts"], "bench_extra_args": list(b["bench_extra_args"]),
        "ft_fingerprint": r.get("ft_fingerprint"),
    })
    return {k: str(v) if isinstance(v, Path) else v for k, v in r.items()}


def fingerprint(fields: dict) -> str:
    """sha256 over the config values a done marker depends on (ft_log.json, bench.done). Paths are
    the config's relative ones, so the fingerprint does not depend on where the workspace lives.
    A marker whose stored fingerprint differs from the current one is stale."""
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def to_shell(resolved: dict) -> str:
    """KEY=value lines for eval in bash; lists become bash arrays."""
    lines = []
    for key, value in resolved.items():
        if isinstance(value, list):
            rendered = "(" + " ".join(shlex.quote(str(v)) for v in value) + ")"
        else:
            rendered = shlex.quote(str(value))
        lines.append(f"{key.upper()}={rendered}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    p_list = sub.add_parser("list", help="run_ids, one per line")
    p_list.add_argument("--bench")
    p_list.add_argument("--arm", choices=ARMS)
    p_list.add_argument("--setup")
    p_list.add_argument("--seed-set")
    p_list.add_argument("--group")
    p_show = sub.add_parser("show", help="all paths/parameters of one run")
    p_show.add_argument("run_id")
    p_show.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.command == "list":
        filters = {"bench": args.bench, "arm": args.arm, "setup": args.setup,
                   "seed_set": args.seed_set, "group": args.group}
        for spec in expand(config):
            if all(v is None or getattr(spec, k) == v for k, v in filters.items()):
                print(spec.run_id)
    else:
        try:
            spec = parse_run_id(args.run_id)
            resolved = resolve(spec, config)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}")
        print(to_shell(resolved) if args.format == "shell" else json.dumps(resolved, indent=2))


if __name__ == "__main__":
    main()
