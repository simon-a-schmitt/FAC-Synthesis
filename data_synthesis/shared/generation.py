"""Generation machinery shared by all three arms (blackbox/, feature_guided/, hybrid/).

Context examples: each generation call is given 3 context examples - 2 seed examples plus 1
example from the accepted pool (or 3 seed examples while the pool is still empty). Seed examples
come from benchmarks/<domain>/<seed set>/ (see shared/benchmarks.py); the accepted pool holds every
synthetic example accepted so far (including earlier runs with the same --prefix).

Dedup: every candidate is compared via ROUGE-L F-measure against seeds + accepted pool and
rejected as "rouge_duplicate" if its best score exceeds --rouge-threshold. The feature-guided arm
additionally rejects candidates that do not activate their target SAE feature ("feature_inactive",
see shared/feature_guidance.py).

Waves: requests run concurrently in waves of 2 * --max-concurrent-requests calls. All prompts of
a wave are built up front from the SAME frozen snapshot of the accepted pool, each slot drawing its
context examples with an RNG seeded from SHA-256(generation_seed, wave_idx, slot), where
generation_seed is derived from the seed group's content (shared/seed_derivation.py) - so every
arm on the same seed group draws identical contexts. After the whole wave has returned (barrier),
the results are applied strictly in slot order (never completion order), so a candidate is deduped
against the snapshot plus whatever earlier slots of the same wave accepted. Outputs and the
checkpoint are written after every wave; --resume continues from there.

Discarded: once a phase has reached its target, the remaining candidates of the same wave are not
checked at all. They are neither accepted nor rejected but "discarded" (reason "target_reached"),
kept with their text in <prefix>_discarded.json (their API cost was paid) and counted separately
(n_discarded), never in n_rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rouge_score import tokenizers
from rouge_score.rouge_scorer import _score_lcs

from prompts import GenerationPrompts, load_generation_prompts
from shared import seed_derivation
from shared.benchmarks import DEFAULT_SEED_SET, DOMAINS, SEED_GROUPS, SEED_SETS, find_seed_file
from shared.openrouter import (
    GENERATION_MODEL_PRESETS,
    EndpointCatalog,
    RateLimiter,
    add_api_args,
    build_model_params,
    call_openrouter_chat,
    expected_provider_name,
    require_api_key,
    response_meta,
    summarize_sampling_verification,
    verify_call,
)
from shared.run_io import (
    append_run_log,
    collect_hardware,
    format_wall_clock_slurm,
    load_checkpoint,
    load_json_list,
    save_checkpoint,
    save_json,
    utc_now,
)
from shared.summary import DISCARD_REASON
from shared.text_cleaning import assert_clean, load_seed_examples

BASE_GENERATION_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 2048,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "usage": {"include": True},
}

# 2 distinct seeds per prompt normally, 3 while the accepted pool is still empty.
MIN_SEED_EXAMPLES = 3
WAVE_SLOTS_PER_WORKER = 2
# Default call cap per phase: --max-calls defaults to this many calls per requested sample.
DEFAULT_MAX_CALLS_PER_SAMPLE = 50

PHASE_BLACKBOX = "blackbox"
PHASE_FEATURE_GUIDED = "feature_guided"

# Args whose change on --resume would mix models or change the context universe.
STRICT_RESUME_ARGS = ("model", "domain", "seed_set", "seed_group", "generation_seed")


# ---------------------------------------------------------------------------
# CLI + setup
# ---------------------------------------------------------------------------

def add_generation_args(parser: argparse.ArgumentParser) -> None:
    """The arguments every arm shares; --n / --max-calls are added by the arm itself."""
    parser.add_argument("--model", type=str, required=True, choices=sorted(GENERATION_MODEL_PRESETS),
                        help="Generator model: " + ", ".join(f"{k}={v[0]}" for k, v in GENERATION_MODEL_PRESETS.items()))
    parser.add_argument("--domain", type=str, required=True, choices=DOMAINS)
    parser.add_argument("--seed-set", type=str, default=DEFAULT_SEED_SET, choices=sorted(SEED_SETS),
                        help="benchmarks/<domain>/seed_groups (k5) or seed_groups_k10 (k10) (default: %(default)s).")
    parser.add_argument("--seed-group", type=str, required=True, choices=SEED_GROUPS)
    parser.add_argument("--prefix", type=str, required=True,
                        help="Filename prefix of the output (<prefix>_accepted/_rejected/_discarded/_failed.json) and log files.")
    parser.add_argument("--rouge-threshold", type=float, required=True,
                        help="Reject a candidate whose best ROUGE-L F-measure against seeds + accepted pool exceeds this.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run from <prefix>_checkpoint.json.")
    add_api_args(parser)


@dataclass
class GenerationSetup:
    """Everything a run derives from its arguments before the first call."""

    args: argparse.Namespace
    arm: str
    domain_dir: Path  # <arm>/<domain>, holds output/ and log/
    seed_file: Path
    seed_examples: list[str]
    generation_seed: int
    prompts: GenerationPrompts
    model_id: str
    provider: str
    model_params: dict
    api_key: str

    def output_path(self, suffix: str) -> Path:
        return self.domain_dir / "output" / f"{self.args.prefix}_{suffix}.json"

    def log_path(self, suffix: str = "log") -> Path:
        return self.domain_dir / "log" / f"{self.args.prefix}_{suffix}.json"

    def resolved_args(self, **extra) -> dict:
        """Snapshot of the args stored in the checkpoint (see load_checkpoint)."""
        args = self.args
        return {
            "model": args.model,
            "domain": args.domain,
            "seed_set": args.seed_set,
            "seed_group": args.seed_group,
            "generation_seed": self.generation_seed,
            "rouge_threshold": args.rouge_threshold,
            "max_concurrent_requests": args.max_concurrent_requests,
            "requests_per_second": args.requests_per_second,
            **extra,
        }

    def run_meta(self, run_id: str) -> dict:
        """The fields every run log entry starts with."""
        args = self.args
        return {
            "run_id": run_id,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hardware": collect_hardware(),
            "arm": self.arm,
            "model": args.model,
            "model_id": self.model_id,
            "domain": args.domain,
            "path": str(self.domain_dir),
            "seed_set": args.seed_set,
            "seed_group": args.seed_group,
            "seed_file": str(self.seed_file),
            "generation_seed": self.generation_seed,
            "n_seeds": len(self.seed_examples),
            "rouge_threshold": args.rouge_threshold,
            "model_params": self.model_params,
        }


def setup_generation(args: argparse.Namespace, arm: str, arm_dir: Path) -> GenerationSetup:
    seed_file = find_seed_file(args.domain, args.seed_set, args.seed_group)
    seed_examples = load_seed_examples(seed_file)
    if len(seed_examples) < MIN_SEED_EXAMPLES:
        raise SystemExit(f"Need at least {MIN_SEED_EXAMPLES} seed examples, but {seed_file} has {len(seed_examples)}.")
    model_id, provider = GENERATION_MODEL_PRESETS[args.model]
    setup = GenerationSetup(
        args=args,
        arm=arm,
        domain_dir=arm_dir / args.domain,
        seed_file=seed_file,
        seed_examples=seed_examples,
        generation_seed=seed_derivation.derive_seed_from_file(args.domain, seed_file, seed_derivation.PURPOSE_GENERATION),
        prompts=load_generation_prompts(args.domain),
        model_id=model_id,
        provider=provider,
        model_params=build_model_params(model_id, provider, BASE_GENERATION_PARAMS, args),
        api_key=require_api_key(args.env_file),
    )
    print(f"Domain: {args.domain} | seed file: {seed_file} ({len(seed_examples)} seeds) | "
          f"generation seed: {setup.generation_seed}")
    print(f"Model: {model_id} | ROUGE-L threshold: {args.rouge_threshold} | output: {setup.domain_dir}")
    return setup


def default_max_calls(max_calls: int | None, n: int) -> int:
    return max_calls if max_calls is not None else DEFAULT_MAX_CALLS_PER_SAMPLE * n


# ---------------------------------------------------------------------------
# Context examples, prompts, candidates
# ---------------------------------------------------------------------------

_ROUGE_TOKENIZER = tokenizers.DefaultTokenizer(use_stemmer=False)


def rouge_tokenize(text: str) -> list[str]:
    return _ROUGE_TOKENIZER.tokenize(text)


def rouge_l_fmeasure(tokens_a: list[str], tokens_b: list[str]) -> float:
    return _score_lcs(tokens_a, tokens_b).fmeasure


def derive_slot_rng(generation_seed: int, wave_idx: int, slot_index: int) -> random.Random:
    """A slot's RNG depends only on (seed group content, wave, slot) - not on the arm, run id,
    worker thread or completion timing."""
    key = f"{generation_seed}|{wave_idx}|{slot_index}"
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))


def pick_context_examples(rng: random.Random, seed_examples: list[str], pool_snapshot: list[dict]) -> list[dict]:
    """2 distinct seed examples + 1 accepted-pool entry, or 3 seed examples while the pool is empty."""
    n_seeds = len(seed_examples)
    if not pool_snapshot:
        return [{"type": "seed", "seed_example_id": i, "text": seed_examples[i]} for i in rng.sample(range(n_seeds), 3)]
    seed_indices = rng.sample(range(n_seeds), 2)
    pool_entry = pool_snapshot[rng.randrange(len(pool_snapshot))]
    context = [{"type": "seed", "seed_example_id": i, "text": seed_examples[i]} for i in seed_indices]
    context.append({"type": "accepted", "accepted_id": pool_entry["id"], "text": pool_entry["text"]})
    return context


def render_examples_block(context_examples: list[dict]) -> str:
    return "\n".join(f"{i + 1}. {entry['text']}" for i, entry in enumerate(context_examples))


def build_user_prompt(template: str, context_examples: list[dict]) -> str:
    user_prompt = template.replace("{{SEED_EXAMPLES}}", render_examples_block(context_examples))
    assert_clean(user_prompt, "user prompt")
    return user_prompt


def parse_single_candidate(text: str) -> list[str]:
    """Every template asks for exactly one output line, so only the first non-empty line is the
    candidate (an optional leading "1." / "1)" stripped); further lines are model commentary and
    are logged as discarded rather than deduped as additional candidates."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return []
    match = re.match(r"^\d+[.)]\s*(.+)$", lines[0])
    candidate = match.group(1).strip() if match else lines[0]
    if len(lines) > 1:
        print(f"  [warn] Model returned {len(lines) - 1} extra line(s) beyond the first; discarding as "
              f"commentary: {lines[1:]!r}", file=sys.stderr)
    return [candidate]


def make_call_id(run_id: str, call_number: int, tag: str | None = None) -> str:
    """Join key between a call's log record and the accepted/rejected/failed entries it produced;
    `tag` namespaces the call counters of hybrid's two phases."""
    return f"{run_id}_{tag}_{call_number:06d}" if tag else f"{run_id}_{call_number:06d}"


def print_raw_first_call(model_id: str, system_prompt: str, user_prompt: str, model_params: dict) -> None:
    """The exact request body of a run's first call (repr shows hidden whitespace)."""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    print("\n" + "=" * 100)
    print("[debug] RAW FIRST CALL OF THIS RUN - what actually goes to the OpenRouter API")
    print("=" * 100)
    print("--- system message (repr) ---")
    print(repr(system_prompt))
    print("--- user message (repr) ---")
    print(repr(user_prompt))
    print("--- full JSON request body (model/messages/params) ---")
    print(json.dumps({"model": model_id, "messages": messages, **model_params}, ensure_ascii=False, indent=2))
    print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# API calls (worker threads)
# ---------------------------------------------------------------------------

@dataclass
class WorkerContext:
    """Read-only bundle shared across worker threads."""

    api_key: str
    model_id: str
    model_params: dict
    system_prompt: str
    max_retry_seconds: float
    request_timeout: float
    rate_limiter: RateLimiter
    expected_provider: str | None  # only set when the provider is pinned


@dataclass
class CallResult:
    call_id: str
    call_number: int
    wave_idx: int
    slot_index: int
    context_examples: list[dict]
    ok: bool
    error: str | None = None
    candidates: list[str] = field(default_factory=list)
    openrouter_meta: dict = field(default_factory=dict)


def fetch_one(ctx: WorkerContext, call_id: str, call_number: int, wave_idx: int, slot_index: int,
              user_prompt: str, context_examples: list[dict]) -> CallResult:
    """Runs on a worker thread: one API call with an already-built prompt. Never touches the pool."""
    result = CallResult(call_id, call_number, wave_idx, slot_index, context_examples, ok=False)
    messages = [{"role": "system", "content": ctx.system_prompt}, {"role": "user", "content": user_prompt}]
    try:
        response = call_openrouter_chat(ctx.api_key, ctx.model_id, messages, ctx.model_params,
                                        ctx.max_retry_seconds, ctx.request_timeout, ctx.rate_limiter)
    except RuntimeError as exc:
        result.error = str(exc)
        return result
    routed = response.get("provider")
    if ctx.expected_provider is not None:
        assert routed is not None and routed.lower() == ctx.expected_provider.lower(), (
            f"Expected provider {ctx.expected_provider!r}, but OpenRouter routed call {call_number} to {routed!r}."
        )
    raw_text = response["choices"][0]["message"].get("content")
    if raw_text is None:
        print("  [warn] Empty message content; treating as no candidates.", file=sys.stderr)
    result.ok = True
    result.candidates = parse_single_candidate(raw_text or "")
    result.openrouter_meta = response_meta(response)
    return result


# ---------------------------------------------------------------------------
# Accepted pool
# ---------------------------------------------------------------------------

class Pool:
    """Seed entries + accepted synthetic entries; the dedup and context universe of a run.

    Persisted as <prefix>_accepted.json (seed entries first, type "seed" / "synthetic"), so that
    file is a complete record of every prompt-eligible text."""

    def __init__(self, seed_entries: list[dict], entries: list[dict]):
        self.seed_entries = seed_entries
        self.entries = entries
        self.seed_tokens = [rouge_tokenize(e["text"]) for e in seed_entries]
        self.tokens = [rouge_tokenize(e["text"]) for e in entries]

    @classmethod
    def load(cls, accepted_path: Path, seed_examples: list[str], started_at: str) -> "Pool":
        stored = load_json_list(accepted_path)
        stored_seeds = [e for e in stored if e.get("type") == "seed"]
        entries = [e for e in stored if e.get("type") == "synthetic"]
        if [e["text"] for e in stored_seeds] == seed_examples:
            seed_entries = stored_seeds
        else:
            if stored_seeds:
                print(f"[warn] Seed entries in {accepted_path} differ from the seed file; replacing them.")
            seed_entries = [
                {"id": i, "type": "seed", "text": text, "seed_example_id": i, "run_id": None, "added_at": started_at}
                for i, text in enumerate(seed_examples)
            ]
        print(f"Loaded {len(entries)} accepted synthetic entrie(s) from {accepted_path}")
        return cls(seed_entries, entries)

    def next_id(self) -> int:
        return max((e["id"] for e in self.entries), default=len(self.seed_entries) - 1) + 1

    def best_match(self, tokens: list[str]) -> tuple[float, dict | None]:
        best_score, best = -1.0, None
        for i, seed_tokens in enumerate(self.seed_tokens):
            score = rouge_l_fmeasure(tokens, seed_tokens)
            if score > best_score:
                best_score, best = score, {"type": "seed", "id": i}
        for entry, entry_tokens in zip(self.entries, self.tokens):
            score = rouge_l_fmeasure(tokens, entry_tokens)
            if score > best_score:
                best_score, best = score, {"type": "synthetic", "id": entry["id"]}
        return best_score, best

    def add(self, entry: dict, tokens: list[str]) -> None:
        self.entries.append(entry)
        self.tokens.append(tokens)

    def save(self, path: Path, phase: str | None = None) -> None:
        """Seeds + all synthetic entries, or only those of one phase (hybrid's per-phase files)."""
        entries = self.entries if phase is None else [e for e in self.entries if e.get("phase") == phase]
        save_json(path, self.seed_entries + entries)


# ---------------------------------------------------------------------------
# Phases and the wave loop
# ---------------------------------------------------------------------------

def new_phase_state(started_at: str) -> dict:
    """Checkpointed state of one phase (a standalone arm has exactly one)."""
    return {
        "wave_idx": 0,
        "counters": {
            "n_calls": 0,
            "n_failed_calls": 0,
            "n_accepted_this_run": 0,
            "n_rejected_this_run": 0,
            "n_discarded_this_run": 0,
            "n_rouge_duplicate": 0,
            "n_feature_inactive": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
        },
        "call_records": [],
        "started_at": started_at,
        "finished_at": None,
    }


@dataclass
class Phase:
    name: str  # PHASE_BLACKBOX / PHASE_FEATURE_GUIDED
    state: dict  # see new_phase_state
    target_n: int
    max_calls: int
    template: str
    rejected: list[dict]
    discarded: list[dict]
    failed: list[dict]
    tag: str | None = None  # hybrid: "bb" / "fg" (call ids, file names, log lines)

    def __post_init__(self):
        # Checkpoints written before "discarded" existed lack the counter.
        self.counters.setdefault("n_discarded_this_run", 0)

    @property
    def counters(self) -> dict:
        return self.state["counters"]

    def target_reached(self) -> bool:
        return self.counters["n_accepted_this_run"] >= self.target_n

    def stats(self) -> dict:
        """The phase's counts for its run log entry."""
        c = self.counters
        return {
            "n_requested": self.target_n,
            "n_calls": c["n_calls"],
            "n_failed_calls": c["n_failed_calls"],
            "n_accepted": c["n_accepted_this_run"],
            "n_rejected": c["n_rejected_this_run"],
            "n_discarded": c["n_discarded_this_run"],
            "n_rouge_duplicate": c["n_rouge_duplicate"],
            "n_feature_inactive": c["n_feature_inactive"],
            "prompt_tokens": c["total_prompt_tokens"],
            "completion_tokens": c["total_completion_tokens"],
            "total_tokens": c["total_prompt_tokens"] + c["total_completion_tokens"],
            "sampling_verification_summary": summarize_sampling_verification(self.state["call_records"]),
        }


@dataclass
class GenerationRun:
    run_id: str
    setup: GenerationSetup
    pool: Pool
    worker: WorkerContext
    endpoint_catalog: EndpointCatalog
    executor: ThreadPoolExecutor
    wave_size: int
    persist: Callable[[], None]  # writes outputs + checkpoint; called after every wave


def start_run(setup: GenerationSetup, run_id: str, pool: Pool, executor: ThreadPoolExecutor,
              persist: Callable[[], None]) -> GenerationRun:
    args = setup.args
    worker = WorkerContext(
        api_key=setup.api_key,
        model_id=setup.model_id,
        model_params=setup.model_params,
        system_prompt=setup.prompts.system,
        max_retry_seconds=args.max_retry_seconds,
        request_timeout=args.request_timeout,
        rate_limiter=RateLimiter(args.requests_per_second),
        expected_provider=expected_provider_name(setup.provider),
    )
    catalog = EndpointCatalog(setup.model_id, args.request_timeout)
    catalog.refresh()
    print(f"Run id: {run_id}")
    return GenerationRun(run_id, setup, pool, worker, catalog, executor,
                         WAVE_SLOTS_PER_WORKER * args.max_concurrent_requests, persist)


def process_call_result(run: GenerationRun, phase: Phase, result: CallResult, slot_fields: dict,
                        call_fields: dict, checks: list[dict] | None) -> tuple[str, list[dict]]:
    """Applies one call's result to the pool. Main thread only, strictly in slot order.

    `checks` (feature-guided only) holds the SAE check of each candidate: {"active", "fields",
    "summary"}. Returns (outcome, per-candidate outcomes); outcome is "failed" (API error),
    "accepted" (>= 1 candidate accepted), "rejected" (no candidate, or all checked ones rejected) or
    "discarded" (every candidate discarded unchecked because the phase's target was already reached)."""
    counters = phase.counters
    context_texts = [c["text"] for c in result.context_examples]
    base = {
        "run_id": run.run_id,
        "call_id": result.call_id,
        "call_number": result.call_number,
        "wave_idx": result.wave_idx,
        "slot_index": result.slot_index,
        **slot_fields,
    }
    if not result.ok:
        print(f"  [error] {result.error}", file=sys.stderr)
        counters["n_failed_calls"] += 1
        phase.failed.append({**base, "context_examples": context_texts, "error": result.error, "timestamp": utc_now()})
        return "failed", []

    usage = result.openrouter_meta.get("usage") or {}
    counters["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
    counters["total_completion_tokens"] += usage.get("completion_tokens", 0)
    call_outcomes: list[dict] = []
    phase.state["call_records"].append({
        **{k: v for k, v in base.items() if k != "run_id"},
        "context_examples": result.context_examples,
        "n_parsed_candidates": len(result.candidates),
        "outcomes": call_outcomes,
        **call_fields,
        "openrouter_response": result.openrouter_meta,
        "sampling_verification": verify_call(result.call_id, run.worker.model_params,
                                             result.openrouter_meta.get("provider"), run.endpoint_catalog),
    })
    base["generation_id"] = result.openrouter_meta.get("id")

    outcome = "rejected"
    for i, text in enumerate(result.candidates):
        check = checks[i] if checks is not None else None
        check_fields = check["fields"] if check else {}
        summary = f"{check['summary']} " if check else ""
        if phase.target_reached():
            phase.discarded.append({**base, "discarded_text": text, "discarded_reason": DISCARD_REASON, **check_fields,
                                    "context_examples": context_texts, "timestamp": utc_now()})
            counters["n_discarded_this_run"] += 1
            call_outcomes.append({"status": "discarded", "discarded_reason": DISCARD_REASON})
            if outcome == "rejected" and all(o["status"] == "discarded" for o in call_outcomes):
                outcome = "discarded"
            continue

        tokens = rouge_tokenize(text)
        score, match = run.pool.best_match(tokens)
        active = check is None or check["active"]
        if not active or score > run.setup.args.rouge_threshold:
            reason = "rouge_duplicate" if active else "feature_inactive"
            phase.rejected.append({
                **base, "rejected_text": text, "rejected_reason": reason, **check_fields,
                "rouge_l_score": score, "matched_pool_id": match["id"], "matched_pool_type": match["type"],
                "context_examples": context_texts, "timestamp": utc_now(),
            })
            counters["n_rejected_this_run"] += 1
            counters[f"n_{reason}"] += 1
            call_outcomes.append({"status": "rejected", "rejected_reason": reason})
            print(f"  [reject:{reason}] {summary}rouge={score:.3f} vs {match['type']} id={match['id']}: {text[:80]!r}")
        else:
            entry_id = run.pool.next_id()
            run.pool.add({
                "id": entry_id, "type": "synthetic", "phase": phase.name, "text": text, **slot_fields, **check_fields,
                "context_examples": context_texts, **{k: v for k, v in base.items() if k not in slot_fields},
                "added_at": utc_now(),
            }, tokens)
            counters["n_accepted_this_run"] += 1
            call_outcomes.append({"status": "accepted", "accepted_id": entry_id})
            outcome = "accepted"
            print(f"  [accept] ({counters['n_accepted_this_run']}/{phase.target_n}) {summary}rouge={score:.3f}: {text[:80]!r}")
    return outcome, call_outcomes


def phase_counts_line(phase: Phase) -> str:
    """accepted / rejected / discarded / failed of a phase, for the "Done:" lines."""
    c = phase.counters
    return (f"{c['n_accepted_this_run']} accepted / {c['n_rejected_this_run']} rejected "
            f"({c['n_rouge_duplicate']} ROUGE duplicate, {c['n_feature_inactive']} feature inactive) / "
            f"{c['n_discarded_this_run']} discarded ({DISCARD_REASON}) / {c['n_failed_calls']} failed call(s) "
            f"over {c['n_calls']} call(s)")


def run_phase(run: GenerationRun, phase: Phase, guidance=None) -> str | None:
    """Runs waves until the phase's target is reached. Returns None on success, else the reason it
    stopped (call cap, all features exhausted, a wave without a single successful call); outputs
    and checkpoint are persisted after every wave either way.

    `guidance` (shared/feature_guidance.FeatureGuidance, feature-guided phase only) picks one target
    feature per slot, extends the prompt, SAE-checks every candidate as soon as its call returns
    (while the rest of the wave is still in flight) and updates the feature schedule."""
    counters = phase.counters
    setup = run.setup
    tag = f"{phase.tag} " if phase.tag else ""
    while not phase.target_reached():
        budget = phase.max_calls - counters["n_calls"]
        if budget <= 0:
            return f"reached the call cap ({phase.max_calls})"
        n_slots = min(run.wave_size, budget)
        slots = guidance.wave_slots(n_slots) if guidance else [None] * n_slots
        if not slots:
            return "all relevant features are exhausted"

        wave_idx = phase.state["wave_idx"]
        pool_snapshot = list(run.pool.entries)  # frozen for this wave's prompts
        print(f"[{tag}wave {wave_idx}] dispatching {len(slots)} call(s) (accepted "
              f"{counters['n_accepted_this_run']}/{phase.target_n}, pool size {len(pool_snapshot)})"
              + (f" {guidance.wave_info()}" if guidance else "") + "...")
        futures = {}
        for slot_index, slot in enumerate(slots):
            counters["n_calls"] += 1
            call_number = counters["n_calls"]
            rng = derive_slot_rng(setup.generation_seed, wave_idx, slot_index)
            context = pick_context_examples(rng, setup.seed_examples, pool_snapshot)
            user_prompt = guidance.build_prompt(phase.template, context, slot) if guidance else build_user_prompt(phase.template, context)
            call_id = make_call_id(run.run_id, call_number, phase.tag)
            if call_number == 1:
                print_raw_first_call(setup.model_id, setup.prompts.system, user_prompt, setup.model_params)
            context_desc = ", ".join(f"{c['type']}:{c.get('seed_example_id', c.get('accepted_id'))}" for c in context)
            print(f"  [{tag}wave {wave_idx} slot {slot_index}] call {call_number} ({call_id})"
                  + (f" {guidance.describe(slot)}" if guidance else "") + f" context=[{context_desc}]")
            future = run.executor.submit(fetch_one, run.worker, call_id, call_number, wave_idx, slot_index, user_prompt, context)
            futures[future] = slot_index

        # Barrier: wait for the whole wave (SAE checks run on the main thread as calls complete).
        results, checks = {}, {}
        for future in as_completed(futures):
            slot_index = futures[future]
            results[slot_index] = future.result()
            if guidance:
                checks[slot_index] = guidance.check(results[slot_index], slots[slot_index])

        outcomes = []
        for slot_index, slot in enumerate(slots):
            result = results[slot_index]
            slot_fields = guidance.slot_fields(slot) if guidance else {}
            call_fields = guidance.call_fields(checks[slot_index]) if guidance else {}
            outcome, call_outcomes = process_call_result(run, phase, result, slot_fields, call_fields, checks.get(slot_index))
            if guidance:
                guidance.after_call(slot, outcome, call_outcomes, checks[slot_index])
            outcomes.append(outcome)
        if guidance:
            guidance.after_wave()
        phase.state["wave_idx"] += 1
        run.persist()
        print(f"[checkpoint] Saved progress after {tag}wave {wave_idx} "
              f"({counters['n_accepted_this_run']}/{phase.target_n} accepted so far).")
        if all(outcome == "failed" for outcome in outcomes):
            return f"every API call of {tag}wave {wave_idx} failed"
    return None


# ---------------------------------------------------------------------------
# Standalone arm (blackbox/ and feature_guided/)
# ---------------------------------------------------------------------------

def run_standalone_arm(args: argparse.Namespace, arm: str, arm_dir: Path, feature_guided: bool = False) -> None:
    """The complete run of the blackbox arm, or of the feature-guided arm. Output:
    <arm>/<domain>/output/<prefix>_accepted.json (seeds + every accepted sample of every run with
    this prefix) / _rejected.json / _failed.json and log/<prefix>_log.json (one entry per completed run)."""
    if feature_guided:
        from shared import feature_guidance as fgd

        fgd.validate_feature_args(args)
    max_calls = default_max_calls(args.max_calls, args.n)
    setup = setup_generation(args, arm, arm_dir)
    resolved_args = setup.resolved_args(n=args.n, max_calls=max_calls,
                                        **(fgd.feature_resolved_args(args) if feature_guided else {}))
    checkpoint_path = setup.output_path("checkpoint")
    checkpoint = load_checkpoint(
        checkpoint_path, args.resume, resolved_args,
        STRICT_RESUME_ARGS + (fgd.STRICT_RESUME_ARGS if feature_guided else ()),
        ("n", "rouge_threshold", "max_calls", "max_concurrent_requests", "requests_per_second"),
    )
    if checkpoint is None:
        run_id = uuid.uuid4().hex
        state = new_phase_state(utc_now())
    else:
        run_id = checkpoint["run_id"]
        state = {k: v for k, v in checkpoint.items() if k not in ("run_id", "resolved_args", "saved_at")}

    pool = Pool.load(setup.output_path("accepted"), setup.seed_examples, state["started_at"])
    phase = Phase(
        PHASE_FEATURE_GUIDED if feature_guided else PHASE_BLACKBOX, state, args.n, max_calls,
        setup.prompts.feature_guided_template if feature_guided else setup.prompts.blackbox_template,
        load_json_list(setup.output_path("rejected")), load_json_list(setup.output_path("discarded")),
        load_json_list(setup.output_path("failed")),
    )

    def persist() -> None:
        pool.save(setup.output_path("accepted"))
        save_json(setup.output_path("rejected"), phase.rejected)
        save_json(setup.output_path("discarded"), phase.discarded)
        save_json(setup.output_path("failed"), phase.failed)
        save_checkpoint(checkpoint_path, {"run_id": run_id, **state, "resolved_args": resolved_args})

    guidance = fgd.start_feature_guidance(args, state, setup.seed_file) if feature_guided else None
    with ThreadPoolExecutor(max_workers=args.max_concurrent_requests) as executor:
        run = start_run(setup, run_id, pool, executor, persist)
        stop_reason = run_phase(run, phase, guidance)

    persist()
    if stop_reason is not None:
        print(f"[warn] Stopped without reaching the target ({stop_reason}; {phase.counters['n_accepted_this_run']}/"
              f"{args.n} accepted); checkpoint retained at {checkpoint_path}. Run again with --resume (e.g. with a "
              "higher --max-calls) to continue.", file=sys.stderr)
        return

    state["finished_at"] = utc_now()
    append_run_log(setup.log_path(), args.prefix, str(setup.domain_dir), {
        **setup.run_meta(run_id),
        **phase.stats(),
        **(guidance.log_fields() if guidance else {}),
        "max_calls": max_calls,
        "endpoint_catalog": run.endpoint_catalog.snapshot(),
        "endpoint_catalog_fetched_at": run.endpoint_catalog.fetched_at,
        "accepted_file": str(setup.output_path("accepted")),
        "rejected_file": str(setup.output_path("rejected")),
        "discarded_file": str(setup.output_path("discarded")),
        "failed_file": str(setup.output_path("failed")),
        "started_at": state["started_at"],
        "finished_at": state["finished_at"],
        "wall_clock_time": format_wall_clock_slurm(state["started_at"], state["finished_at"]),
        **({"feature_stats": state["feature_stats"]} if guidance else {}),
        "calls": state["call_records"],
    })
    checkpoint_path.unlink(missing_ok=True)
    print(f"Done: {phase_counts_line(phase)}; prompt_tokens={phase.counters['total_prompt_tokens']}, "
          f"completion_tokens={phase.counters['total_completion_tokens']}")
    if guidance:
        guidance.print_sae_summary()
