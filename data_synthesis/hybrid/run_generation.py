"""Hybrid generation for one domain: first --n-blackbox examples exactly as in the blackbox arm,
then --n-feature-guided examples exactly as in the feature-guided arm, targeted at the relevant
SAE features that neither the seeds nor the blackbox examples cover.

Sibling of ../blackbox/run_generation.py and ../feature_guided/run_generation.py. All generation,
dedup, SAE and scheduling machinery is imported from those two scripts unchanged (the
feature-guided module in turn imports the blackbox one), so each phase behaves identically to its
arm. Both phases share one run_id, one generator model (set below), one --rouge-threshold and one
accepted pool.

Phase 1 - blackbox (arm "hybrid_blackbox"):
  Waves of blackbox calls (prompt: blackbox/<domain>/prompts/prompt_step_1.py,
  STEP_1_PROMPT_TEMPLATE; context: 2 seeds + 1 accepted-pool entry, or 3 seeds while the pool is
  empty; ROUGE-L dedup against seeds + pool) until --n-blackbox candidates are accepted.

Coverage (once, between the phases, stored in the checkpoint):
  The seeds are run through Llama + SAE exactly as the feature-guided arm's seed coverage does
  (active_feature_identification/get_active_features/run_get_active_features.py). The accepted
  blackbox examples go through the same function, with the task set to the domain (they have no
  label column to detect it from). A relevant feature counts as covered if its raw activation
  exceeds --threshold on at least one content token of at least one seed OR blackbox example.

Phase 2 - feature-guided (arm "hybrid_feature_guided"):
  The feature-guided schedule, unchanged: pass 0 over the relevant features NOT covered by
  seeds + blackbox examples, later passes over all relevant features, up to
  --attempts-per-feature attempts per feature and pass, until --n-feature-guided candidates are
  accepted (prompt: feature_guided/<domain>/prompts/prompt_step_1.py, STEP_1_FG_PROMPT_TEMPLATE).
  The accepted pool the context examples are drawn from and deduped against already contains the
  blackbox examples.

Output (in hybrid/<domain>/output and hybrid/<domain>/log, named after --prefix):
  <prefix>_bb_accepted.json / _bb_rejected.json / _bb_failed.json, log/<prefix>_bb_log.json
      the blackbox phase (accepted = seeds + blackbox examples)
  <prefix>_fg_accepted.json / _fg_rejected.json / _fg_failed.json, log/<prefix>_fg_log.json
      the feature-guided phase (accepted = seeds + feature-guided examples), incl. the coverage
  <prefix>_accepted.json, log/<prefix>_log.json
      every accepted example of both phases (seeds + blackbox + feature-guided, each synthetic
      entry tagged with "phase"), plus a run summary. Same naming as the other arms, so
      labeling/run_labeling.py works on all three accepted files.
A run cannot append to existing output of the same --prefix (use a fresh prefix); an interrupted
run is continued with --resume from '<prefix>_checkpoint.json'.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import uuid
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

BASE_DIR = Path(__file__).parent
DATA_SYNTHESIS_DIR = BASE_DIR.parent
BLACKBOX_DIR = DATA_SYNTHESIS_DIR / "blackbox"
FEATURE_GUIDED_DIR = DATA_SYNTHESIS_DIR / "feature_guided"
FEATURE_GUIDED_SCRIPT = FEATURE_GUIDED_DIR / "run_generation.py"


def _load_feature_guided_module():
    spec = importlib.util.spec_from_file_location("hybrid_fg_run_generation", FEATURE_GUIDED_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Must be registered before exec: its @dataclass definitions look themselves up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Shared machinery - imported, not copied, so it can never drift from either arm.
fg = _load_feature_guided_module()
bb = fg.bb

# ---------------------------------------------------------------------------
# Generator model - one for both phases, selected via --model
# ---------------------------------------------------------------------------

# --model selects one of these by keyword: keyword -> (model id, provider), as in
# labeling/run_labeling.py.
MODEL_PRESETS = {
    "llama": ("meta-llama/llama-3.1-8b-instruct", "deepinfra/fp8"),
    "deepseek": ("deepseek/deepseek-v4-flash-0731", "baseten/fp8"),
}
DEFAULT_MODEL = "deepseek"

# See feature_guided/run_generation.py.
PIN_PROVIDER = False

GENERATION_MODEL_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 2048,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "usage": {"include": True},
}

# Set by configure_model() from --model.
MODEL_ID: str = ""
PROVIDER: str = ""


def configure_model(model_keyword: str) -> None:
    """Sets the generator model for both phases. The imported helpers read these as module globals
    at call time (fg.build_model_params, and bb.fetch_one / bb.derive_seed / bb.print_raw_first_call).
    Both modules are private instances loaded just for this script, so this never affects the
    blackbox or feature-guided arms."""
    global MODEL_ID, PROVIDER
    MODEL_ID, PROVIDER = MODEL_PRESETS[model_keyword]
    base_model_params = dict(GENERATION_MODEL_PARAMS)
    if fg.DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT in MODEL_ID:
        base_model_params["reasoning"] = {"enabled": False}
    for module in (fg, bb):
        module.MODEL_ID = MODEL_ID
        module.PROVIDER = PROVIDER
        module.PIN_PROVIDER = PIN_PROVIDER
        # OpenRouter's response echoes back just the provider's display name.
        module.EXPECTED_PROVIDER_NAME = PROVIDER.split("/")[0]
        module.BASE_MODEL_PARAMS = base_model_params

# One arm id per phase, so the two phases' per-slot RNGs never collide under the shared run_id.
ARM = "hybrid"
ARM_BLACKBOX = "hybrid_blackbox"
ARM_FEATURE_GUIDED = "hybrid_feature_guided"
PHASE_BLACKBOX = "blackbox"
PHASE_FEATURE_GUIDED = "feature_guided"
# call_id namespaces (see bb.make_call_id), short so the ids stay readable.
CALL_ID_PHASE_BLACKBOX = "bb"
CALL_ID_PHASE_FEATURE_GUIDED = "fg"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def resolve_domain_dir(path_arg: str) -> Path:
    """The domain's hybrid/<domain> directory (holds output/ and log/)."""
    candidate = Path(path_arg)
    for option in (candidate, BASE_DIR / candidate):
        if option.is_dir() and option.resolve().parent == BASE_DIR.resolve():
            return option.resolve()
    raise SystemExit(f"--path '{path_arg}' is not a domain directory under {BASE_DIR}.")


def load_prompts(domain_name: str) -> dict:
    """The blackbox prompt of blackbox/<domain>, and the feature-guided + SAE classification
    prompts of feature_guided/<domain> / labeling/<domain>, so each phase uses its arm's prompt."""
    bb_step_1 = fg._load_module(  # noqa: SLF001
        BLACKBOX_DIR / domain_name / "prompts" / "prompt_step_1.py",
        f"hybrid_bb_prompt_step_1_{domain_name}",
        ("SYSTEM_PROMPT", "STEP_1_PROMPT_TEMPLATE"),
    )
    fg_system, fg_template, sae_system = fg.load_prompts(FEATURE_GUIDED_DIR / domain_name)
    return {
        "bb_system": bb_step_1.SYSTEM_PROMPT,
        "bb_template": bb_step_1.STEP_1_PROMPT_TEMPLATE,
        "fg_system": fg_system,
        "fg_template": fg_template,
        "sae_system": sae_system,
    }


def build_blackbox_user_prompt(template: str, context_examples: list[dict]) -> str:
    user_prompt = template.replace("{{SEED_EXAMPLES}}", bb.render_seed_examples_block(context_examples))
    fg.assert_clean(user_prompt, "blackbox prompt")
    return user_prompt


def load_get_active_features_module():
    """Must be called after fg.load_sae_context() (offline env vars, sys.path)."""
    spec = importlib.util.spec_from_file_location("hybrid_get_active_features", fg.GET_ACTIVE_FEATURES_SCRIPT)
    gaf = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(gaf)
    return gaf


def compute_text_peak_activations(texts: list[str], task: str, sae_ctx) -> list[dict[int, float]]:
    """Per text: {feature_id: max raw SAE activation over its content tokens}, computed exactly as
    fg.compute_seed_peak_activations() does for a seed row of the given task."""
    gaf = load_get_active_features_module()
    if task not in (gaf.TASK_TOXICITY, gaf.TASK_CTI_VSP, gaf.TASK_CLAUDETTE):
        raise SystemExit(f"Domain {task!r} is not a task known to {fg.GET_ACTIVE_FEATURES_SCRIPT}.")
    peaks = []
    with sae_ctx.fs.tc.no_grad():
        for text in texts:
            system, user_content, opening_phrase = gaf.build_prompt_for_row(task, text)
            peaks.append(
                gaf.compute_max_raw_activation_per_feature(
                    user_content, sae_ctx.model, sae_ctx.collector, sae_ctx.sae,
                    opening_phrase=opening_phrase, system=system,
                )
            )
    return peaks


# ---------------------------------------------------------------------------
# Coverage + feature schedule
# ---------------------------------------------------------------------------

def new_schedule(
    features: list[dict],
    seed_peaks: dict[int, float],
    bb_entries: list[dict],
    bb_text_peaks: list[dict[int, float]],
    threshold: float,
) -> dict:
    """Checkpointed schedule state in the format fg.advance_pass_if_done / fg.update_schedule
    expect, plus the coverage breakdown: pass 0 = relevant features covered neither by the seeds
    nor by the blackbox examples, in priority order."""
    relevant_ids = [f["feature_id"] for f in features]
    bb_peaks: dict[int, float] = {}
    for text_peaks in bb_text_peaks:
        for feature_id, peak in text_peaks.items():
            if peak > bb_peaks.get(feature_id, float("-inf")):
                bb_peaks[feature_id] = peak

    seed_covered = {fid for fid in relevant_ids if seed_peaks.get(fid, 0.0) > threshold}
    bb_covered = {fid for fid in relevant_ids if bb_peaks.get(fid, 0.0) > threshold}
    uncovered = [fid for fid in relevant_ids if fid not in seed_covered | bb_covered]
    relevant_set = set(relevant_ids)
    return {
        "covered": [fid for fid in relevant_ids if fid in seed_covered | bb_covered],
        "uncovered": uncovered,
        "seed_covered": [fid for fid in relevant_ids if fid in seed_covered],
        "blackbox_covered": [fid for fid in relevant_ids if fid in bb_covered],
        "blackbox_only_covered": [fid for fid in relevant_ids if fid in bb_covered - seed_covered],
        "seed_peak_activations": {str(fid): round(seed_peaks[fid], 6) for fid in relevant_ids if fid in seed_peaks},
        "blackbox_peak_activations": {str(fid): round(bb_peaks[fid], 6) for fid in relevant_ids if fid in bb_peaks},
        # Relevant features each accepted blackbox example activates above the threshold.
        "blackbox_entry_active_features": {
            str(entry["id"]): sorted(
                (fid for fid, peak in text_peaks.items() if fid in relevant_set and peak > threshold),
                key=relevant_ids.index,
            )
            for entry, text_peaks in zip(bb_entries, bb_text_peaks)
        },
        "pass_idx": 0,
        "pass_queue": list(uncovered),
        "pass_attempts": {},
        "exhausted": [],
        "exhausted_in_pass": {},
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def new_counters(next_id: int, feature_guided: bool) -> dict:
    counters = {
        "n_calls": 0,
        "n_failed_calls": 0,
        "n_accepted_this_run": 0,
        "n_rejected_this_run": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "next_id": next_id,
    }
    if feature_guided:
        counters.update(n_feature_inactive=0, n_rouge_duplicate=0)
    return counters


def tag_new_entries(pool: list[dict], start: int, phase: str) -> None:
    for entry in pool[start:]:
        entry.setdefault("phase", phase)


def save_pool_files(paths: dict, seed_entries: list[dict], pool: list[dict]) -> None:
    bb.save_json(paths["bb_accepted"], seed_entries + [e for e in pool if e["phase"] == PHASE_BLACKBOX])
    bb.save_json(paths["fg_accepted"], seed_entries + [e for e in pool if e["phase"] == PHASE_FEATURE_GUIDED])
    bb.save_json(paths["accepted"], seed_entries + pool)


def append_run_log(log_path: Path, prefix: str, domain_dir: Path, run_entry: dict) -> None:
    """Adds run_entry to the log (replacing an earlier entry of the same run, so a crash right
    after writing the log never double-counts a run on --resume)."""
    existing_log = bb.load_json_dict(log_path) or {}
    runs = [run for run in existing_log.get("runs", []) if run.get("run_id") != run_entry["run_id"]]
    runs.append(run_entry)
    prompt_tokens = sum(run.get("prompt_tokens", 0) for run in runs)
    completion_tokens = sum(run.get("completion_tokens", 0) for run in runs)
    bb.save_json(
        log_path,
        {
            "prefix": prefix,
            "path": str(domain_dir),
            "cumulative_prompt_tokens": prompt_tokens,
            "cumulative_completion_tokens": completion_tokens,
            "cumulative_total_tokens": prompt_tokens + completion_tokens,
            "runs": runs,
        },
    )
    print(f"Wrote run log to {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid generation: --n-blackbox blackbox examples, then --n-feature-guided "
        "feature-guided examples for the features still uncovered, via the OpenRouter API."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, choices=sorted(MODEL_PRESETS),
        help="Generator model keyword (both phases), mapped to an OpenRouter model id + provider via "
        f"MODEL_PRESETS ({', '.join(f'{k}={v[0]}' for k, v in MODEL_PRESETS.items())}; default: %(default)s).",
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Domain subfolder under hybrid/ (e.g. 'toxicity_detection'); prompts are taken from "
        "blackbox/<domain> and feature_guided/<domain>.",
    )
    parser.add_argument(
        "--seed-group", type=str, required=True, choices=sorted(bb.VALID_SEED_GROUPS),
        help="Seed group id (01-05); selects data/seed_groups/<domain>/*_seed_group_<id>.tsv.",
    )
    parser.add_argument("--n-blackbox", type=int, required=True, help="Number of ACCEPTED blackbox examples (phase 1).")
    parser.add_argument(
        "--n-feature-guided", type=int, required=True, help="Number of ACCEPTED feature-guided examples (phase 2)."
    )
    parser.add_argument(
        "--prefix", type=str, required=True,
        help="Filename prefix for the output ('<prefix>[_bb|_fg]_accepted.json' / '_rejected.json' / "
        "'_failed.json') and log ('<prefix>[_bb|_fg]_log.json') files.",
    )
    parser.add_argument(
        "--rouge-threshold", type=float, required=True,
        help="Reject a candidate (in both phases) if its highest ROUGE-L F-measure against seeds + "
        "accepted pool exceeds this value.",
    )

    # Feature guidance (phase 2)
    parser.add_argument(
        "--feature-scores", type=Path, default=None,
        help="Feature relevance JSONL (default: data/feature_scores/<domain>_feature_relevance_scores.jsonl).",
    )
    parser.add_argument(
        "--feature-labels", type=str, nargs="+", default=fg.DEFAULT_FEATURE_LABELS,
        choices=["Yes", "Probably", "Maybe", "No"],
        help="Task-relevant feature labels, in scheduling priority order (default: %(default)s).",
    )
    parser.add_argument(
        "--attempts-per-feature", type=int, default=fg.DEFAULT_ATTEMPTS_PER_FEATURE,
        help="Max. generation calls per feature and pass (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold", type=float, required=True,
        help="A feature counts as active (on a seed, blackbox example or candidate) if its RAW SAE "
        "activation exceeds this value on at least one content token.",
    )

    # SAE verification model
    parser.add_argument("--model-name", type=str, required=True, help="Local Llama-3.1-8B-Instruct directory.")
    parser.add_argument("--sae-ckpt-path", type=str, required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--hf-cache-dir", type=str, default=os.environ.get("TRANSFORMERS_CACHE", ""))

    # Generation API
    parser.add_argument("--temperature", type=float, default=None, help="Override BASE_MODEL_PARAMS['temperature'].")
    parser.add_argument("--top-p", type=float, default=None, help="Override BASE_MODEL_PARAMS['top_p'].")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override BASE_MODEL_PARAMS['max_tokens'].")
    parser.add_argument(
        "--extra-params", type=str, default=None,
        help="Additional OpenRouter request body parameters as a JSON object string.",
    )
    parser.add_argument(
        "--max-calls-blackbox", type=int, default=None,
        help="Safety cap on phase-1 API calls (default: 50x --n-blackbox, as in the blackbox arm).",
    )
    parser.add_argument(
        "--max-calls-feature-guided", type=int, default=None,
        help="Safety cap on phase-2 API calls (default: unlimited, as in the feature-guided arm).",
    )
    parser.add_argument("--env-file", type=Path, default=bb.DEFAULT_ENV_FILE, help="Path to a .env file providing OPENROUTER_API_KEY.")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-retry-seconds", type=float, default=600.0,
        help="Total wall-clock budget (seconds) to keep retrying a single failed HTTP request (default: %(default)s).",
    )
    parser.add_argument(
        "--max-concurrent-requests", type=int, default=1,
        help="Number of worker threads (default: 1). Requests run in waves of 2 * --max-concurrent-requests "
        "prompts in both phases.",
    )
    parser.add_argument(
        "--requests-per-second", type=float, default=None,
        help="Optional cap on request START rate shared across all concurrent workers (default: unlimited).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run from its checkpoint ('<prefix>_checkpoint.json' in <path>/output).",
    )
    return parser


# Changing any of these on --resume would mix generator models within one run, or invalidate the
# context universe or the checkpointed coverage / feature schedule.
SCHEDULE_ARGS = ("model", "seed_group", "feature_scores", "feature_labels", "attempts_per_feature", "threshold")
# Changing any of these on --resume is allowed, but noted.
NOTED_ARGS = (
    "n_blackbox", "n_feature_guided", "rouge_threshold", "max_calls_blackbox", "max_calls_feature_guided",
    "max_concurrent_requests", "requests_per_second",
)


def checkpoint_resolved_args(
    domain_dir: Path, args: argparse.Namespace, feature_scores: Path, max_calls_bb: int, max_calls_fg: int | None
) -> dict:
    return {
        "model": args.model,
        "path": str(domain_dir),
        "seed_group": args.seed_group,
        "feature_scores": str(feature_scores),
        "feature_labels": list(args.feature_labels),
        "attempts_per_feature": args.attempts_per_feature,
        "threshold": args.threshold,
        "n_blackbox": args.n_blackbox,
        "n_feature_guided": args.n_feature_guided,
        "rouge_threshold": args.rouge_threshold,
        "max_calls_blackbox": max_calls_bb,
        "max_calls_feature_guided": max_calls_fg,
        "max_concurrent_requests": args.max_concurrent_requests,
        "requests_per_second": args.requests_per_second,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    configure_model(args.model)
    if args.n_blackbox < 0 or args.n_feature_guided < 0:
        raise SystemExit("--n-blackbox and --n-feature-guided must be >= 0.")
    if args.attempts_per_feature < 1:
        raise SystemExit("--attempts-per-feature must be >= 1.")
    # Fail now rather than after the (API-only) blackbox phase.
    if not os.path.isdir(os.path.abspath(args.model_name)):
        raise SystemExit(f"--model-name: local model directory not found: {os.path.abspath(args.model_name)}")
    if args.sae_ckpt_path and not os.path.exists(args.sae_ckpt_path):
        raise SystemExit(f"--sae-ckpt-path: {args.sae_ckpt_path} does not exist.")

    bb.load_dotenv(args.env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(f"OPENROUTER_API_KEY is not set. Set it in the environment or in {args.env_file}.")

    domain_dir = resolve_domain_dir(args.path)
    domain_name = domain_dir.name
    prompts = load_prompts(domain_name)
    seed_file = fg.find_seed_file(domain_name, args.seed_group)
    seed_examples = fg.load_seed_examples(seed_file)
    n_seeds = len(seed_examples)
    if n_seeds < bb.MIN_SEED_EXAMPLES:
        raise SystemExit(
            f"Need at least {bb.MIN_SEED_EXAMPLES} seed examples for the context-drawing scheme, "
            f"but {seed_file} only has {n_seeds}."
        )

    feature_scores = args.feature_scores or fg.FEATURE_SCORES_DIR / f"{domain_name}_feature_relevance_scores.jsonl"
    features = fg.load_features(feature_scores, args.feature_labels)
    features_by_id = {feature["feature_id"]: feature for feature in features}

    model_params = fg.build_model_params(args)
    max_calls_bb = args.max_calls_blackbox if args.max_calls_blackbox is not None else 50 * args.n_blackbox
    max_calls_fg = args.max_calls_feature_guided

    output_dir = domain_dir / "output"
    log_dir = domain_dir / "log"
    paths = {
        "accepted": output_dir / f"{args.prefix}_accepted.json",
        "bb_accepted": output_dir / f"{args.prefix}_bb_accepted.json",
        "bb_rejected": output_dir / f"{args.prefix}_bb_rejected.json",
        "bb_failed": output_dir / f"{args.prefix}_bb_failed.json",
        "fg_accepted": output_dir / f"{args.prefix}_fg_accepted.json",
        "fg_rejected": output_dir / f"{args.prefix}_fg_rejected.json",
        "fg_failed": output_dir / f"{args.prefix}_fg_failed.json",
        "checkpoint": output_dir / f"{args.prefix}_checkpoint.json",
        "log": log_dir / f"{args.prefix}_log.json",
        "bb_log": log_dir / f"{args.prefix}_bb_log.json",
        "fg_log": log_dir / f"{args.prefix}_fg_log.json",
    }

    resolved_args = checkpoint_resolved_args(domain_dir, args, feature_scores, max_calls_bb, max_calls_fg)
    checkpoint = bb.load_json_dict(paths["checkpoint"])

    if args.resume:
        if checkpoint is None:
            raise SystemExit(f"--resume was given but no checkpoint found at {paths['checkpoint']}; nothing to resume.")
        for key in SCHEDULE_ARGS:
            old_value = checkpoint["resolved_args"].get(key)
            if old_value != resolved_args[key]:
                raise SystemExit(
                    f"Checkpoint at {paths['checkpoint']} was created with {key}={old_value!r}, but this invocation "
                    f"has {key}={resolved_args[key]!r}. That would change the context/feature schedule - use the "
                    "original value, or delete the checkpoint and output files to start fresh."
                )
        for key in NOTED_ARGS:
            old_value = checkpoint["resolved_args"].get(key)
            if old_value != resolved_args[key]:
                print(f"[resume] Note: --{key.replace('_', '-')} changed since the checkpoint was saved ({old_value!r} -> {resolved_args[key]!r}).")
        run_id = checkpoint["run_id"]
        started_at = checkpoint["started_at"]
        phase = checkpoint["phase"]
        bb_state = checkpoint["blackbox"]
        fg_state = checkpoint["feature_guided"]
        print(f"[resume] Continuing run {run_id} in the {phase} phase (checkpoint saved at {checkpoint['saved_at']}).")
    else:
        if checkpoint is not None:
            raise SystemExit(
                f"Found an incomplete checkpoint at {paths['checkpoint']} (run_id={checkpoint['run_id']}, "
                f"phase {checkpoint['phase']}). Pass --resume to continue it, or delete it (and the run's output "
                "files) to start a fresh run."
            )
        existing = [str(paths[key]) for key in ("accepted", "bb_accepted", "fg_accepted") if paths[key].exists()]
        if existing:
            raise SystemExit(
                f"Output of an earlier run with --prefix {args.prefix!r} already exists ({existing}). A hybrid run "
                "always starts from the seeds alone - use a new --prefix."
            )
        run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        phase = PHASE_BLACKBOX
        bb_state = {
            "wave_idx": 0,
            "counters": new_counters(n_seeds, feature_guided=False),
            "call_records": [],
            "started_at": started_at,
            "finished_at": None,
        }
        fg_state = None  # set up once the blackbox phase is done and Llama + SAE are loaded

    print(f"Domain: {domain_dir} | seed group: {args.seed_group} | seed file: {seed_file} ({n_seeds} seeds)")
    label_counts = {label: sum(f["label"] == label for f in features) for label in args.feature_labels}
    print(f"Relevant features: {len(features)} {label_counts} from {feature_scores}")
    print(
        f"Run id: {run_id} | model: {MODEL_ID} | target accepted: {args.n_blackbox} blackbox + "
        f"{args.n_feature_guided} feature-guided | max calls: {max_calls_bb} / {max_calls_fg} "
        f"| SAE raw threshold: {args.threshold} | ROUGE-L threshold: {args.rouge_threshold}"
    )

    # Seeds are not part of the accepted pool (same split as in both arms); the pool holds the
    # synthetic entries of both phases, each tagged with its "phase".
    seed_entries = [
        {"id": i, "type": "seed", "text": text, "seed_example_id": i, "run_id": None, "added_at": started_at}
        for i, text in enumerate(seed_examples)
    ]
    pool = [e for e in bb.load_json_list(paths["accepted"]) if e.get("type") == "synthetic"]
    seed_tokens = [bb.rouge_tokenize(e["text"]) for e in seed_entries]
    pool_tokens = [bb.rouge_tokenize(e["text"]) for e in pool]
    print(f"Loaded {len(pool)} accepted synthetic entrie(s) from {paths['accepted']}")

    bb_rejected = bb.load_json_list(paths["bb_rejected"])
    bb_failed = bb.load_json_list(paths["bb_failed"])
    fg_rejected = bb.load_json_list(paths["fg_rejected"])
    fg_failed = bb.load_json_list(paths["fg_failed"])

    n_workers = args.max_concurrent_requests
    wave_batch_size = 2 * n_workers
    # Shared by both phases (same generator model); per-call verification, see blackbox.
    endpoint_catalog = bb.EndpointCatalog(MODEL_ID, args.request_timeout)
    endpoint_catalog.refresh()

    def save_checkpoint() -> None:
        bb.save_json(
            paths["checkpoint"],
            {
                "run_id": run_id,
                "started_at": started_at,
                "phase": phase,
                "blackbox": bb_state,
                "feature_guided": fg_state,
                "resolved_args": resolved_args,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def save_outputs() -> None:
        save_pool_files(paths, seed_entries, pool)
        bb.save_json(paths["bb_rejected"], bb_rejected)
        bb.save_json(paths["bb_failed"], bb_failed)
        bb.save_json(paths["fg_rejected"], fg_rejected)
        bb.save_json(paths["fg_failed"], fg_failed)

    executor = ThreadPoolExecutor(max_workers=n_workers)
    try:
        # -------------------------------------------------------------------
        # Phase 1: blackbox (the wave loop of blackbox/run_generation.py main())
        # -------------------------------------------------------------------
        if phase == PHASE_BLACKBOX:
            counters = bb_state["counters"]
            bb_args = SimpleNamespace(n=args.n_blackbox, rouge_threshold=args.rouge_threshold)
            ctx = bb.WorkerContext(
                api_key=api_key,
                model_params=model_params,
                system_prompt=prompts["bb_system"],
                max_retry_seconds=args.max_retry_seconds,
                request_timeout=args.request_timeout,
                rate_limiter=bb.RateLimiter(args.requests_per_second),
            )
            print(f"=== Phase 1: blackbox ({counters['n_accepted_this_run']}/{args.n_blackbox} accepted so far) ===")
            while counters["n_accepted_this_run"] < args.n_blackbox:
                remaining_budget = max_calls_bb - counters["n_calls"]
                if remaining_budget <= 0:
                    break
                slots_this_wave = min(wave_batch_size, remaining_budget)
                wave_idx = bb_state["wave_idx"]
                pool_snapshot = list(pool)  # frozen for this wave's prompt construction
                print(
                    f"[bb wave {wave_idx}] dispatching {slots_this_wave} call(s) "
                    f"(accepted {counters['n_accepted_this_run']}/{args.n_blackbox}, pool snapshot size {len(pool_snapshot)})..."
                )

                futures: dict = {}
                for slot_index in range(slots_this_wave):
                    counters["n_calls"] += 1
                    call_number = counters["n_calls"]
                    slot_rng = bb.derive_slot_rng(run_id, args.seed_group, ARM_BLACKBOX, wave_idx, slot_index)
                    context_examples = bb.pick_context_examples(slot_rng, seed_examples, pool_snapshot)
                    user_prompt = build_blackbox_user_prompt(prompts["bb_template"], context_examples)
                    derived_seed = bb.derive_seed(f"{run_id}|{ARM_BLACKBOX}", call_number)
                    # Phase-namespaced: both phases count calls from 1 and share one accepted file.
                    call_id = bb.make_call_id(run_id, call_number, CALL_ID_PHASE_BLACKBOX)
                    if call_number == 1:
                        bb.print_raw_first_call(prompts["bb_system"], user_prompt, model_params, derived_seed)
                    context_desc = ", ".join(
                        f"{c['type']}:{c.get('seed_example_id', c.get('accepted_id'))}" for c in context_examples
                    )
                    print(f"  [bb wave {wave_idx} slot {slot_index}] call {call_number} ({call_id}) context=[{context_desc}]")
                    future = executor.submit(
                        bb.fetch_one, ctx, call_id, call_number, wave_idx, slot_index, user_prompt, context_examples, derived_seed
                    )
                    futures[future] = slot_index

                wait(futures.keys(), return_when=ALL_COMPLETED)  # barrier
                results_by_slot = {futures[future]: future.result() for future in futures}
                pool_size_before = len(pool)
                for slot_index in range(slots_this_wave):
                    bb.process_call_result(
                        results_by_slot[slot_index],
                        run_id=run_id,
                        args=bb_args,
                        seed_tokens=seed_tokens,
                        accepted_pool=pool,
                        pool_tokens=pool_tokens,
                        rejected_entries=bb_rejected,
                        failed_entries=bb_failed,
                        call_records=bb_state["call_records"],
                        counters=counters,
                        endpoint_catalog=endpoint_catalog,
                    )
                tag_new_entries(pool, pool_size_before, PHASE_BLACKBOX)
                bb_state["wave_idx"] += 1

                save_outputs()
                save_checkpoint()
                print(f"[checkpoint] Saved progress after bb wave {wave_idx} ({counters['n_accepted_this_run']}/{args.n_blackbox} accepted so far).")

            save_outputs()
            if counters["n_accepted_this_run"] < args.n_blackbox:
                save_checkpoint()
                print(
                    f"[warn] Blackbox phase stopped at --max-calls-blackbox ({max_calls_bb}) with "
                    f"{counters['n_accepted_this_run']}/{args.n_blackbox} accepted; checkpoint retained at "
                    f"{paths['checkpoint']}. Run again with --resume (and a higher --max-calls-blackbox) to continue.",
                    file=sys.stderr,
                )
                return

            bb_state["finished_at"] = datetime.now(timezone.utc).isoformat()
            append_run_log(
                paths["bb_log"], f"{args.prefix}_bb", domain_dir,
                {
                    "run_id": run_id,
                    "arm": ARM_BLACKBOX,
                    "phase": PHASE_BLACKBOX,
                    "model": args.model,
                    "model_id": MODEL_ID,
                    "path": str(domain_dir),
                    "seed_group": args.seed_group,
                    "seed_file": str(seed_file),
                    "n_seeds": n_seeds,
                    "n_requested": args.n_blackbox,
                    "rouge_threshold": args.rouge_threshold,
                    "n_calls": counters["n_calls"],
                    "n_failed_calls": counters["n_failed_calls"],
                    "n_accepted": counters["n_accepted_this_run"],
                    "n_rejected": counters["n_rejected_this_run"],
                    "prompt_tokens": counters["total_prompt_tokens"],
                    "completion_tokens": counters["total_completion_tokens"],
                    "total_tokens": counters["total_prompt_tokens"] + counters["total_completion_tokens"],
                    "model_params": model_params,
                    "sampling_verification_summary": bb.summarize_sampling_verification(bb_state["call_records"]),
                    "endpoint_catalog": endpoint_catalog.snapshot(),
                    "endpoint_catalog_fetched_at": endpoint_catalog.fetched_at,
                    "accepted_file": str(paths["bb_accepted"]),
                    "rejected_file": str(paths["bb_rejected"]),
                    "failed_file": str(paths["bb_failed"]),
                    "started_at": bb_state["started_at"],
                    "finished_at": bb_state["finished_at"],
                    "wall_clock_time": bb.format_wall_clock_slurm(bb_state["started_at"], bb_state["finished_at"]),
                    "calls": bb_state["call_records"],
                },
            )
            phase = PHASE_FEATURE_GUIDED
            save_checkpoint()
            print(
                f"Blackbox phase done: {counters['n_accepted_this_run']} accepted / "
                f"{counters['n_rejected_this_run']} rejected over {counters['n_calls']} call(s)."
            )

        # -------------------------------------------------------------------
        # Coverage of seeds + blackbox examples
        # -------------------------------------------------------------------
        print("Loading Llama + SAE for coverage + feature verification...")
        sae_ctx = fg.load_sae_context(args, prompts["sae_system"])

        if fg_state is None:
            bb_entries = [e for e in pool if e["phase"] == PHASE_BLACKBOX]
            print(f"Computing coverage of the relevant features on {n_seeds} seed(s) + {len(bb_entries)} blackbox example(s) (threshold {args.threshold})...")
            seed_peaks = fg.compute_seed_peak_activations(seed_file, sae_ctx)
            bb_text_peaks = compute_text_peak_activations([e["text"] for e in bb_entries], domain_name, sae_ctx)
            fg_state = {
                "wave_idx": 0,
                "counters": new_counters(bb_state["counters"]["next_id"], feature_guided=True),
                "call_records": [],
                "feature_stats": {},
                "schedule": new_schedule(features, seed_peaks, bb_entries, bb_text_peaks, args.threshold),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            save_checkpoint()
        schedule = fg_state["schedule"]
        seed_set, bb_set, covered_set = (set(schedule[key]) for key in ("seed_covered", "blackbox_covered", "covered"))
        for label in args.feature_labels:
            label_ids = [f["feature_id"] for f in features if f["label"] == label]
            n_seed = sum(fid in seed_set for fid in label_ids)
            n_bb = sum(fid in bb_set for fid in label_ids)
            n_covered = sum(fid in covered_set for fid in label_ids)
            print(
                f"  [coverage] {label:<9} seeds {n_seed:>4} / blackbox {n_bb:>4} / covered {n_covered:>4} / "
                f"not covered {len(label_ids) - n_covered:>4} / total {len(label_ids):>4}"
            )
        fg.advance_pass_if_done(schedule, features)

        # -------------------------------------------------------------------
        # Phase 2: feature-guided (the wave loop of feature_guided/run_generation.py main())
        # -------------------------------------------------------------------
        counters = fg_state["counters"]
        fg_args = SimpleNamespace(n=args.n_feature_guided, threshold=args.threshold, rouge_threshold=args.rouge_threshold)
        ctx = bb.WorkerContext(
            api_key=api_key,
            model_params=model_params,
            system_prompt=prompts["fg_system"],
            max_retry_seconds=args.max_retry_seconds,
            request_timeout=args.request_timeout,
            rate_limiter=bb.RateLimiter(args.requests_per_second),
        )
        print(
            f"=== Phase 2: feature-guided ({counters['n_accepted_this_run']}/{args.n_feature_guided} accepted so far; "
            f"pass {schedule['pass_idx']}: {len(schedule['pass_queue'])} feature(s) queued, {len(schedule['exhausted'])} exhausted) ==="
        )
        stop_reason = None
        while counters["n_accepted_this_run"] < args.n_feature_guided:
            remaining_budget = None if max_calls_fg is None else max_calls_fg - counters["n_calls"]
            if remaining_budget is not None and remaining_budget <= 0:
                stop_reason = f"reached --max-calls-feature-guided ({max_calls_fg})"
                break
            if not schedule["pass_queue"]:
                stop_reason = f"all {len(features)} relevant features are exhausted"
                break

            wave_idx = fg_state["wave_idx"]
            wave_feature_ids = schedule["pass_queue"][:wave_batch_size]
            if remaining_budget is not None:
                wave_feature_ids = wave_feature_ids[:remaining_budget]
            slots_this_wave = len(wave_feature_ids)
            pool_snapshot = list(pool)
            print(
                f"[fg wave {wave_idx}] pass {schedule['pass_idx']}: dispatching {slots_this_wave} call(s) "
                f"({len(schedule['pass_queue'])} feature(s) left in pass, {len(schedule['exhausted'])} exhausted; "
                f"accepted {counters['n_accepted_this_run']}/{args.n_feature_guided}, pool snapshot size {len(pool_snapshot)})..."
            )

            futures = {}
            slot_infos: dict[int, fg.SlotInfo] = {}
            for slot_index, feature_id in enumerate(wave_feature_ids):
                counters["n_calls"] += 1
                call_number = counters["n_calls"]
                feature = features_by_id[feature_id]
                slot_info = fg.SlotInfo(
                    feature=feature,
                    pass_idx=schedule["pass_idx"],
                    feature_attempt=schedule["pass_attempts"].get(str(feature_id), 0),
                )
                slot_infos[slot_index] = slot_info
                slot_rng = bb.derive_slot_rng(run_id, args.seed_group, ARM_FEATURE_GUIDED, wave_idx, slot_index)
                context_examples = bb.pick_context_examples(slot_rng, seed_examples, pool_snapshot)
                user_prompt = fg.build_user_prompt(prompts["fg_template"], context_examples, feature)
                derived_seed = bb.derive_seed(f"{run_id}|{ARM_FEATURE_GUIDED}", call_number)
                call_id = bb.make_call_id(run_id, call_number, CALL_ID_PHASE_FEATURE_GUIDED)
                if call_number == 1:
                    bb.print_raw_first_call(prompts["fg_system"], user_prompt, model_params, derived_seed)
                context_desc = ", ".join(
                    f"{c['type']}:{c.get('seed_example_id', c.get('accepted_id'))}" for c in context_examples
                )
                print(
                    f"  [fg wave {wave_idx} slot {slot_index}] call {call_number} ({call_id}) feature={feature_id} "
                    f"({feature['label']}) attempt={slot_info.feature_attempt} context=[{context_desc}]"
                )
                future = executor.submit(
                    bb.fetch_one, ctx, call_id, call_number, wave_idx, slot_index, user_prompt, context_examples, derived_seed
                )
                futures[future] = slot_index

            # SAE-verify each call as soon as it completes; also the wave's barrier.
            results_by_slot = {}
            sae_by_slot: dict[int, list[dict]] = {}
            for future in as_completed(futures):
                slot_index = futures[future]
                result = future.result()
                results_by_slot[slot_index] = result
                feature_id = slot_infos[slot_index].feature["feature_id"]
                sae_by_slot[slot_index] = [
                    fg.compute_target_activation(text, feature_id, sae_ctx) for text in (result.candidates if result.ok else [])
                ]

            outcomes = []
            pool_size_before = len(pool)
            for slot_index in range(slots_this_wave):
                outcome = fg.process_call_result(
                    results_by_slot[slot_index],
                    slot_infos[slot_index],
                    sae_by_slot[slot_index],
                    run_id=run_id,
                    args=fg_args,
                    seed_tokens=seed_tokens,
                    accepted_pool=pool,
                    pool_tokens=pool_tokens,
                    rejected_entries=fg_rejected,
                    failed_entries=fg_failed,
                    call_records=fg_state["call_records"],
                    counters=counters,
                    feature_stats=fg_state["feature_stats"],
                    endpoint_catalog=endpoint_catalog,
                )
                outcomes.append(outcome)
                fg.update_schedule(schedule, slot_infos[slot_index].feature["feature_id"], outcome, args.attempts_per_feature)
            tag_new_entries(pool, pool_size_before, PHASE_FEATURE_GUIDED)
            fg.advance_pass_if_done(schedule, features)
            fg_state["wave_idx"] += 1

            save_outputs()
            save_checkpoint()
            print(f"[checkpoint] Saved progress after fg wave {wave_idx} ({counters['n_accepted_this_run']}/{args.n_feature_guided} accepted so far).")

            if all(outcome == "failed" for outcome in outcomes):
                # Failed calls do not use up attempts, so a persistent API outage would otherwise loop forever.
                stop_reason = f"every API call of fg wave {wave_idx} failed"
                break
    finally:
        executor.shutdown(wait=True)

    save_outputs()
    print(
        f"Wrote {n_seeds} seed + {len(pool)} synthetic entrie(s) to {paths['accepted']} "
        f"(split by phase into {paths['bb_accepted'].name} / {paths['fg_accepted'].name})"
    )
    if counters["n_accepted_this_run"] < args.n_feature_guided:
        save_checkpoint()
        print(
            f"[warn] Feature-guided phase stopped without reaching its target ({stop_reason}; "
            f"{counters['n_accepted_this_run']}/{args.n_feature_guided} accepted, {counters['n_calls']} call(s) made); "
            f"checkpoint retained at {paths['checkpoint']}. Run again with --resume (e.g. with a higher "
            "--max-calls-feature-guided, if non-exhausted features remain) to continue.",
            file=sys.stderr,
        )
        return

    finished_at = datetime.now(timezone.utc).isoformat()
    append_run_log(
        paths["fg_log"], f"{args.prefix}_fg", domain_dir,
        {
            "run_id": run_id,
            "arm": ARM_FEATURE_GUIDED,
            "phase": PHASE_FEATURE_GUIDED,
            "model": args.model,
            "model_id": MODEL_ID,
            "path": str(domain_dir),
            "seed_group": args.seed_group,
            "seed_file": str(seed_file),
            "n_seeds": n_seeds,
            "n_blackbox_examples": bb_state["counters"]["n_accepted_this_run"],
            "feature_scores": str(feature_scores),
            "feature_labels": list(args.feature_labels),
            "attempts_per_feature": args.attempts_per_feature,
            "n_relevant_features": len(features),
            "n_covered": len(schedule["covered"]),
            "n_uncovered": len(schedule["uncovered"]),
            "n_seed_covered": len(schedule["seed_covered"]),
            "n_blackbox_covered": len(schedule["blackbox_covered"]),
            "n_blackbox_only_covered": len(schedule["blackbox_only_covered"]),
            "covered": schedule["covered"],
            "uncovered": schedule["uncovered"],
            "seed_covered": schedule["seed_covered"],
            "blackbox_covered": schedule["blackbox_covered"],
            "blackbox_only_covered": schedule["blackbox_only_covered"],
            "seed_peak_activations": schedule["seed_peak_activations"],
            "blackbox_peak_activations": schedule["blackbox_peak_activations"],
            "blackbox_entry_active_features": schedule["blackbox_entry_active_features"],
            "n_passes_started": schedule["pass_idx"] + 1,
            "n_exhausted": len(schedule["exhausted"]),
            "exhausted": schedule["exhausted"],
            "exhausted_in_pass": schedule["exhausted_in_pass"],
            "n_features_used": len(fg_state["feature_stats"]),
            "threshold": args.threshold,
            "sae_model": os.path.abspath(args.model_name),
            "sae_ckpt_path": os.path.abspath(args.sae_ckpt_path),
            "sae_layer": args.layer,
            "n_requested": args.n_feature_guided,
            "rouge_threshold": args.rouge_threshold,
            "n_calls": counters["n_calls"],
            "n_failed_calls": counters["n_failed_calls"],
            "n_accepted": counters["n_accepted_this_run"],
            "n_rejected": counters["n_rejected_this_run"],
            "n_feature_inactive": counters["n_feature_inactive"],
            "n_rouge_duplicate": counters["n_rouge_duplicate"],
            "prompt_tokens": counters["total_prompt_tokens"],
            "completion_tokens": counters["total_completion_tokens"],
            "total_tokens": counters["total_prompt_tokens"] + counters["total_completion_tokens"],
            "model_params": model_params,
            "sampling_verification_summary": bb.summarize_sampling_verification(fg_state["call_records"]),
            "endpoint_catalog": endpoint_catalog.snapshot(),
            "endpoint_catalog_fetched_at": endpoint_catalog.fetched_at,
            "accepted_file": str(paths["fg_accepted"]),
            "rejected_file": str(paths["fg_rejected"]),
            "failed_file": str(paths["fg_failed"]),
            "started_at": fg_state["started_at"],
            "finished_at": finished_at,
            "wall_clock_time": bb.format_wall_clock_slurm(fg_state["started_at"], finished_at),
            "feature_stats": fg_state["feature_stats"],
            "calls": fg_state["call_records"],
        },
    )

    bb_counters = bb_state["counters"]
    prompt_tokens = bb_counters["total_prompt_tokens"] + counters["total_prompt_tokens"]
    completion_tokens = bb_counters["total_completion_tokens"] + counters["total_completion_tokens"]
    append_run_log(
        paths["log"], args.prefix, domain_dir,
        {
            "run_id": run_id,
            "arm": ARM,
            "model": args.model,
            "model_id": MODEL_ID,
            "path": str(domain_dir),
            "seed_group": args.seed_group,
            "seed_file": str(seed_file),
            "n_seeds": n_seeds,
            "n_requested_blackbox": args.n_blackbox,
            "n_requested_feature_guided": args.n_feature_guided,
            "n_accepted_blackbox": bb_counters["n_accepted_this_run"],
            "n_accepted_feature_guided": counters["n_accepted_this_run"],
            "n_accepted": bb_counters["n_accepted_this_run"] + counters["n_accepted_this_run"],
            "n_calls_blackbox": bb_counters["n_calls"],
            "n_calls_feature_guided": counters["n_calls"],
            "n_relevant_features": len(features),
            "n_covered_before_feature_guided": len(schedule["covered"]),
            "threshold": args.threshold,
            "rouge_threshold": args.rouge_threshold,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model_params": model_params,
            "sampling_verification_summary": bb.summarize_sampling_verification(
                bb_state["call_records"] + fg_state["call_records"]
            ),
            "accepted_file": str(paths["accepted"]),
            "bb_accepted_file": str(paths["bb_accepted"]),
            "fg_accepted_file": str(paths["fg_accepted"]),
            "bb_log_file": str(paths["bb_log"]),
            "fg_log_file": str(paths["fg_log"]),
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_clock_time": bb.format_wall_clock_slurm(started_at, finished_at),
        },
    )

    if paths["checkpoint"].exists():
        paths["checkpoint"].unlink()
        print(f"Target reached; removed checkpoint {paths['checkpoint']}")

    print(
        f"Done: {bb_counters['n_accepted_this_run']} blackbox + {counters['n_accepted_this_run']} feature-guided accepted "
        f"({counters['n_feature_inactive']} feature inactive, {counters['n_rouge_duplicate']} ROUGE duplicate in phase 2) "
        f"over {bb_counters['n_calls']} + {counters['n_calls']} call(s); "
        f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"
    )


if __name__ == "__main__":
    main()
