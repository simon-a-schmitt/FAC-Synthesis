"""Measures the prompt length (tokens of the rendered chat text, as generate_batch() encodes it)
of benchmark runs without a GPU: per run, the benchmark script's own parse_args() is fed the
command line ft_bench_job.sh would build from runs.py, and its own build_chat_messages /
render_chat_text render every test prompt. Compared against the run's --max-input-tokens.

Usage:
    python experiments/measure_prompt_lengths.py                      # plain + icl of every benchmark
    python experiments/measure_prompt_lengths.py --bench toxicity_detection
    python experiments/measure_prompt_lengths.py <run_id> [<run_id> ...]
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from collections import defaultdict

from runs import PROJECT_DIR, expand, load_config, parse_run_id, resolve

sys.path.insert(0, str(PROJECT_DIR))


def bench_argv(r: dict) -> list[str]:
    """The benchmark command line of ft_bench_job.sh (phase_benchmark) for a plain/icl run."""
    argv = [r["bench_script"], "--model-path", r["base_model"], "--data-tsv", r["test_tsv"],
            "--mode", r["bench_mode"], "--max-prompts", str(r["max_prompts"]), "--device", "cpu"]
    if r["bench_mode"] == "icl":
        argv += ["--few-shot-tsv", r["few_shot_tsv"], "--icl-k", str(r["icl_k"])]
    return argv + list(r["bench_extra_args"]) + ["--output-jsonl", "/dev/null"]


def measure(r: dict, tokenizers: dict) -> dict:
    module_name = "benchmark_play_ground." + r["bench_script"].rsplit("/", 1)[1].removesuffix(".py")
    mod = importlib.import_module(module_name)
    sys.argv = bench_argv(r)
    args = mod.parse_args()

    loader = next(getattr(mod, n) for n in dir(mod) if n.startswith("load_") and n.endswith("_tsv"))
    records = loader(args.data_tsv)
    if args.max_prompts > 0:
        records = records[: args.max_prompts]
    few_shots = loader(args.few_shot_tsv) if args.few_shot_tsv and args.mode == "icl" else []

    if args.model_path not in tokenizers:
        from transformers import AutoTokenizer
        tokenizers[args.model_path] = AutoTokenizer.from_pretrained(args.model_path)
    tok = tokenizers[args.model_path]
    lengths = sorted(
        len(tok(mod.render_chat_text(tok, mod.build_chat_messages(args, rec["prompt"], few_shots)),
                add_special_tokens=False)["input_ids"])
        for rec in records
    )
    limit = args.max_input_tokens
    return {"lengths": lengths, "limit": limit,
            "n_over": sum(1 for n in lengths if limit and n > limit)}


def p99(lengths: list[int]) -> int:
    return lengths[min(len(lengths) - 1, math.ceil(0.99 * len(lengths)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_ids", nargs="*")
    parser.add_argument("--bench")
    args = parser.parse_args()

    config = load_config()
    if args.run_ids:
        specs = [parse_run_id(r) for r in args.run_ids]
    else:
        specs = [s for s in expand(config) if s.arm in ("plain", "icl") and args.bench in (None, s.bench)]

    tokenizers: dict = {}
    by_cell: dict = defaultdict(list)  # (bench, seed set) -> [(run_id, result)]
    print(f"{'run_id':40s} {'n':>5s} {'max':>6s} {'p99':>6s} {'limit':>6s} {'>limit':>7s}")
    for spec in specs:
        r = resolve(spec, config)
        m = measure(r, tokenizers)
        by_cell[(spec.bench, spec.seed_set or "plain")].append(m)
        print(f"{r['run_id']:40s} {len(m['lengths']):5d} {m['lengths'][-1]:6d} {p99(m['lengths']):6d} "
              f"{str(m['limit']):>6s} {m['n_over']:7d}")

    print("\nPer benchmark x seed set (over all groups):")
    print(f"{'bench':20s} {'set':6s} {'max':>6s} {'p99':>6s} {'limit':>6s} {'>limit (sum)':>13s} {'>limit (worst group)':>21s}")
    for (bench, seed_set), ms in by_cell.items():
        lengths = sorted(n for m in ms for n in m["lengths"])
        print(f"{bench:20s} {seed_set:6s} {lengths[-1]:6d} {p99(lengths):6d} {str(ms[0]['limit']):>6s} "
              f"{sum(m['n_over'] for m in ms):13d} {max(m['n_over'] for m in ms):21d}")


if __name__ == "__main__":
    main()
