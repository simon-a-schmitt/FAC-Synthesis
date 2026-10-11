#!/usr/bin/env python3
"""Runs one labeling-benchmark run (lb_runs.py): free generation on the test set, then the labeling
parser (lb_parsers.py; an answer it cannot parse is invalid as a whole) and the metrics (lb_metrics.py).

Prompts: build_chat_messages() of the experiments benchmark script of the domain
(run_{toxicity,claudette,cti_vsp}_benchmark.py) - the same messages experiments sends, incl. the ICL
turns; the api arm gets the plain messages. Their system prompt must equal the labeling prompt
(prompts/<domain>/labeling.py), checked at the start.

Generation: greedy, at most max_new_tokens (config.yaml), no stop strings.
  local  base model (+ LoRA adapter of experiments/runs/<run_id>/lora for gold/bb/fg/hybrid, which
         must be done there), batches of batch_size, halved on CUDA OOM
  api    OpenRouter (api_prediction.ApiPredictor, temperature 0, no logprobs)

Resumable: raw_outputs.jsonl is appended per batch; batches are fixed blocks of the example index
(so the left-padded batch of an example does not depend on interruptions), a block that is not
complete on disk is generated again. Every line carries the lb_fingerprint; a run directory with
outputs of another fingerprint (or a stale done marker) is moved to <name>.stale.<timestamp> first.
A run whose done marker carries the current fingerprint is skipped (--force: evaluate again from
raw_outputs.jsonl).

Usage (benchmark env fac_env, from the repo root):
    python benchmark_play_ground/labeling_benchmark/run_lbench.py <run_id>
    python benchmark_play_ground/labeling_benchmark/run_lbench.py <run_id> --max-prompts 50 --output-dir <dir>
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from datetime import datetime
from pathlib import Path

from lb_metrics import evaluate
from lb_parsers import SLOTS, LabelingParser
from lb_experiments_eval import load_test
from lb_runs import api_model_params, load_configs, parse_run_id, prerequisite, resolve_full

from benchmark_play_ground.api_prediction import ApiPredictor, add_api_cli_args, first_messages_dump
from prompts import load_labeling_prompt

BENCH_MODULES = {
    "toxicity_detection": ("benchmark_play_ground.run_toxicity_benchmark", "TOXICITY_SYSTEM_PROMPT"),
    "claudette_tos": ("benchmark_play_ground.run_claudette_benchmark", "CLAUDETTE_SYSTEM_PROMPT"),
    "cti_vsp": ("benchmark_play_ground.run_cti_vsp_benchmark", "CTI_VSP_SYSTEM_PROMPT"),
}
OUTPUT_KEYS = ("raw_jsonl", "predictions_jsonl", "summary_json", "lb_done")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_id")
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Override max_prompts (smoke test; needs --output-dir, part of the fingerprint).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Write the outputs here instead of labeling_benchmark/runs/<run_id>/.")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch_size of config.yaml.")
    p.add_argument("--force", action="store_true", help="Evaluate again even if the run is done.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    add_api_cli_args(p)
    p.set_defaults(max_concurrent_requests=None)  # default: api_max_concurrent_requests of config.yaml
    args = p.parse_args()
    if args.max_prompts is not None and args.output_dir is None:
        p.error("--max-prompts needs --output-dir (the run directory holds full runs only).")
    return args


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def line_fingerprints(path: Path) -> set:
    fps = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                fps.add(json.loads(line).get("fingerprint"))
            except json.JSONDecodeError:
                continue
    return fps


def move_stale(r: dict) -> None:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    for key in OUTPUT_KEYS:
        path = Path(r[key])
        if path.exists():
            target = path.with_name(f"{path.name}.stale.{ts}")
            path.rename(target)
            print(f"[STALE] moved {path} -> {target}")


def done_fingerprint(r: dict) -> str | None:
    try:
        return json.loads(Path(r["lb_done"]).read_text(encoding="utf-8")).get("lb_fingerprint") or ""
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return ""


def load_raw(r: dict, records: list[dict], block: int) -> dict[int, dict]:
    """index -> raw line, complete blocks only (the file is rewritten without the others)."""
    path = Path(r["raw_jsonl"])
    if not path.exists():
        return {}
    done: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # cut off by a killed job; its block is generated again
        i = rec["index"]
        if rec.get("fingerprint") != r["lb_fingerprint"] or i >= len(records) or rec["prompt"] != records[i]["prompt"]:
            raise SystemExit(f"{path}: line for index {i} belongs to another setup / test set.")
        done[i] = rec
    n_blocks = (len(records) + block - 1) // block
    keep = {}
    for b in range(n_blocks):
        idx = range(b * block, min((b + 1) * block, len(records)))
        if all(i in done for i in idx):
            keep.update({i: done[i] for i in idx})
    if len(keep) != len(done):
        print(f"[RESUME] dropping {len(done) - len(keep)} line(s) of incomplete batches")
        with path.open("w", encoding="utf-8") as fh:
            for i in sorted(keep):
                fh.write(json.dumps(keep[i], ensure_ascii=False) + "\n")
    return keep


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class LocalGenerator:
    def __init__(self, r: dict, args):
        import torch
        from benchmark_play_ground.model_wrapper import LocalModel

        self.torch = torch
        self.device = args.device
        lora = r.get("lora_dir") if r["mode"] == "fine_tuned" else None
        model = LocalModel(r["base_model"], device=args.device, dtype=args.dtype, lora_path=lora)
        model.load()
        self.tokenizer = model.tokenizer
        self.model = model._gen._model
        self.max_new_tokens = r["max_new_tokens"]
        eos = self.model.generation_config.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.pad_token_id}
        self.dumped = False
        print(f"Local model: {r['base_model']}" + (f" + LoRA {lora}" if lora else ""))

    def _generate(self, texts: list[str]) -> list[dict]:
        torch = self.torch
        encoded = [self.tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
        if not self.dumped:
            self.dumped = True
            print("=" * 88 + "\nRAW MODEL INPUT -- first example (printed once)\n" + "=" * 88)
            print(texts[0] + "\n" + "=" * 88, flush=True)
        padded = self.tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt")
        input_ids = padded["input_ids"].to(self.device)
        attention_mask = padded["attention_mask"].to(self.device)
        outputs = None
        try:
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=self.max_new_tokens,
                do_sample=False, use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            if len(texts) <= 1:
                raise
        # Retry outside the except block, so the failed attempt's tensors can be freed (as in
        # run_cti_vsp_benchmark.generate_batch).
        if outputs is None:
            del input_ids, attention_mask, padded
            gc.collect()
            torch.cuda.empty_cache()
            mid = len(texts) // 2
            print(f"CUDA OOM at batch size {len(texts)}. Splitting into {mid} + {len(texts) - mid}...", flush=True)
            return self._generate(texts[:mid]) + self._generate(texts[mid:])
        out = []
        for row in outputs[:, input_ids.shape[1]:].tolist():
            n_gen = next((k for k, t in enumerate(row) if t in self.stop_ids), None)
            out.append({
                "raw_output": self.tokenizer.decode(row, skip_special_tokens=True),
                "hit_max_new_tokens": n_gen is None,
            })
        return out

    def generate(self, chat_messages: list[list[dict]]) -> list[dict]:
        texts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in chat_messages]
        out = self._generate(texts)
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        return out


class ApiGenerator:
    def __init__(self, r: dict, args):
        args.api_model = r["api_model"]
        if args.max_concurrent_requests is None:
            args.max_concurrent_requests = r["api_max_concurrent"]
        self.predictor = ApiPredictor(args, max_tokens=r["max_new_tokens"], logprobs=False)
        if self.predictor.model_params != api_model_params(r["api_model"], r["max_new_tokens"]):
            raise SystemExit("lb_runs.api_model_params() no longer builds the parameters ApiPredictor sends.")
        self.dumped = False

    def generate(self, chat_messages: list[list[dict]]) -> list[dict]:
        if not self.dumped:
            self.dumped = True
            first_messages_dump(chat_messages[0])
        return [{"raw_output": resp["text"], "hit_max_new_tokens": resp["openrouter"].get("finish_reason") == "length",
                 "openrouter": resp["openrouter"]}
                for resp in self.predictor.predict_many(chat_messages)]


def message_builder(r: dict, records_loader):
    """chat messages of a query, from the experiments benchmark script of the domain."""
    module_name, prompt_attr = BENCH_MODULES[r["bench"]]
    module = importlib.import_module(module_name)
    if getattr(module, prompt_attr) != load_labeling_prompt(r["bench"]).system:
        raise SystemExit(f"{module_name}.{prompt_attr} differs from prompts/{r['bench']}/labeling.py SYSTEM_PROMPT.")
    few_shots = []
    if r["mode"] == "icl":
        few_shots = records_loader(r["few_shot_tsv"])
        if len(few_shots) < r["icl_k"]:
            raise SystemExit(f"{r['few_shot_tsv']}: {len(few_shots)} few-shot examples < icl_k={r['icl_k']}.")
    ns = argparse.Namespace(mode="plain" if r["mode"] == "api" else r["mode"], icl_k=r.get("icl_k", 0))
    return lambda prompt: module.build_chat_messages(ns, prompt, few_shots)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def parse_and_evaluate(r: dict, records: list[dict], gold: list[dict], raw: dict[int, dict]) -> tuple[list[dict], dict]:
    domain, slots = r["bench"], SLOTS[r["bench"]]
    parser = LabelingParser(domain)
    rows = []
    for i, rec in enumerate(records):
        text = raw[i]["raw_output"]
        values = parser.parse(text)
        rows.append({"index": i, "gt": rec["gt"], "gt_values": gold[i], "raw_output": text,
                     "hit_max_new_tokens": raw[i].get("hit_max_new_tokens"),
                     "values": values, "valid": all(values[s] is not None for s in slots)})
    metrics = evaluate(domain, gold, [row["values"] for row in rows])
    counts = {
        "n_examples": len(rows),
        "n_valid": sum(row["valid"] for row in rows),
        "n_hit_max_new_tokens": sum(bool(row["hit_max_new_tokens"]) for row in rows),
    }
    return rows, {"metrics": metrics, "counts": counts}


def main() -> None:
    args = parse_args()
    exp_config, lb_config = load_configs()
    r, exp_r = resolve_full(parse_run_id(args.run_id), exp_config, lb_config)
    if args.output_dir is not None:
        out = args.output_dir.resolve()
        r.update(lb_run_dir=str(out), **{k: str(out / Path(r[k]).name) for k in OUTPUT_KEYS})
    if args.max_prompts is not None:
        r["max_prompts"] = args.max_prompts
        r["lb_fingerprint"] = hashlib.sha256(f"{r['lb_fingerprint']}|max_prompts={args.max_prompts}".encode()).hexdigest()
    if args.batch_size is not None:
        r["batch_size"] = args.batch_size
    started_at = now()
    print(f"Run: {r['run_id']} ({r['mode']}) -> {r['lb_run_dir']}\nFingerprint: {r['lb_fingerprint']}", flush=True)

    stored = done_fingerprint(r)
    if stored == r["lb_fingerprint"] and not args.force:
        print(f"[SKIP] {r['lb_done']} is current.")
        return
    if (stored is not None and stored != r["lb_fingerprint"]) or (line_fingerprints(Path(r["raw_jsonl"])) - {r["lb_fingerprint"]}):
        move_stale(r)
    reason = prerequisite(r, exp_r)
    if reason:
        raise SystemExit(f"{r['run_id']}: not ready ({reason}).")
    Path(r["lb_run_dir"]).mkdir(parents=True, exist_ok=True)

    from benchmark_play_ground.data_loader import load_claudette_tsv, load_cti_vsp_tsv, load_toxicity_tsv
    loader = {"toxicity_detection": load_toxicity_tsv, "claudette_tos": load_claudette_tsv,
              "cti_vsp": load_cti_vsp_tsv}[r["bench"]]
    records, gold = load_test(r["bench"], r["test_tsv"], r["max_prompts"])
    build_messages = message_builder(r, loader)

    block = max(1, r["batch_size"])
    raw = load_raw(r, records, block)
    todo = [b for b in range(0, len(records), block) if b not in raw]
    print(f"{len(raw)}/{len(records)} answers on disk, {len(todo)} batch(es) of {block} to generate", flush=True)
    if todo:
        generator = ApiGenerator(r, args) if r["mode"] == "api" else LocalGenerator(r, args)
        with Path(r["raw_jsonl"]).open("a", encoding="utf-8") as fh:
            for k, start in enumerate(todo, 1):
                idx = list(range(start, min(start + block, len(records))))
                outputs = generator.generate([build_messages(records[i]["prompt"]) for i in idx])
                for i, o in zip(idx, outputs):
                    rec = {"index": i, "fingerprint": r["lb_fingerprint"], "prompt": records[i]["prompt"],
                           "gt": records[i]["gt"], **o}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    raw[i] = rec
                fh.flush()
                print(f"  [{k}/{len(todo)}] batch @ {start} done", flush=True)

    rows, result = parse_and_evaluate(r, records, gold, raw)
    with Path(r["predictions_jsonl"]).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "run_id": r["run_id"], "bench": r["bench"], "arm": r["arm"], "mode": r["mode"],
        "lb_fingerprint": r["lb_fingerprint"], "fingerprint_fields": json.loads(r["lb_fingerprint_fields"]),
        "model": r.get("api_model_id") or r.get("base_model"), "lora_dir": r.get("lora_dir"),
        "max_new_tokens": r["max_new_tokens"], "batch_size": r["batch_size"], "max_prompts": r["max_prompts"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_at": started_at, "finished_at": now(),
        **result,
    }
    Path(r["summary_json"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    c = result["counts"]
    m = result["metrics"]
    print(f"== {r['run_id']}: n={c['n_examples']}  valid={c['n_valid']}  hit_max_new_tokens={c['n_hit_max_new_tokens']}  "
          f"{m['metric']}={m['value']:.4f}")

    tmp = Path(r["lb_done"] + ".tmp")
    tmp.write_text(json.dumps({"run_id": r["run_id"], "lb_fingerprint": r["lb_fingerprint"], "finished_at": now(),
                               "slurm_job_id": os.environ.get("SLURM_JOB_ID")}) + "\n", encoding="utf-8")
    os.replace(tmp, r["lb_done"])
    print(f"[DONE] {r['lb_done']}")


if __name__ == "__main__":
    main()
