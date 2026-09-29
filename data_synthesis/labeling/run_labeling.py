"""Label synthetic examples for one domain (e.g. toxicity_detection) via the OpenRouter
API, produced by blackbox/run_generation.py, feature_guided/run_generation.py, or
hybrid/run_generation.py.

--source selects the generation method (blackbox, feature_guided, or hybrid); together
with --path (the domain) this gives the source domain dir <source>/<path>. Reads
<source>/<path>/output/<prefix>_accepted.json (a JSON list of seed + synthetic entries,
same format for both methods) and asks an OpenRouter model to label every non-seed
("type" != "seed") entry's text, using the system prompt from
labeling/<domain>/prompt/prompt_step_2.py (SYSTEM_PROMPT / LABEL_FRAGMENTS) - shared by
both sources, so the same labeling prompt is used regardless of generation method. Seed entries
are not re-labeled - their label already exists in the seed_group TSV the original
generation run drew from (found via the "seed_file" recorded in
<source>/<path>/log/<prefix>_log.json), already stored in the exact "<prefix>: <label>"
format the model is asked to produce, so seed and model-produced labels end up formatted
identically in the output.

Writes one TSV to labeling/<domain>/<prefix>.tsv next to this script (prefix = the input
filename with its trailing "_accepted" stripped, domain = the source domain dir's name) -
one prompt/label pair per line, tab-separated, seed examples first (in seed order)
followed by the labeled synthetic examples (in accepted-pool order, regardless of
completion order - see fetch_one/main).

Requests run concurrently via a thread pool, reusing run_generation.py's RateLimiter to
respect the API's tps limit. Unlike run_generation.py's generation calls, labeling calls
don't build on each other (no shared mutable pool, no dedup against prior results), so
there is no need for its wave/barrier scheme: every synthetic entry's call is simply
submitted up front and results are collected as they complete, then written out in the
original entry order.

Resumable like run_generation.py: if <prefix>.tsv already exists, it is loaded first and
its non-empty labels (keyed by text) are reused as-is - only texts that are new or still
unlabeled (empty label, e.g. from a prior failed/unparsed call) are sent to the API this
run. The full seed + synthetic set is then rewritten to the same file, so a run never loses
labels obtained by an earlier one and can simply be re-invoked to fill in the rest after a
partial failure.

Once every sample has had an initial attempt, whatever is still unresolved (a response that
didn't parse, or a call that failed outright) is retried up to MAX_UNPARSED_RETRIES more
times as its own follow-up round(s). Anything still unresolved after that is assigned the
majority label among every other already-labeled data point (seed + synthetic), so the
output TSV never ends up with blank labels.

Every failed/unparsed attempt (initial and each retry round) is printed - including the raw,
unparseable model output, for the parse-failure case - and appended to
labeling/<domain>/<prefix>_failed.json, so nothing has to be reconstructed from terminal
scrollback after the fact. Like the TSV, this file is loaded and extended across runs rather
than overwritten.

Model, provider, and sampling params are independent of run_generation.py's: --model picks
one of the MODEL_PRESETS keywords hardcoded at the top of this file ("gpt" or "deepseek"),
each mapping to an OpenRouter model id + provider slug. --provider/--temperature/--max-tokens/
--extra-params override the preset's provider/LABELING_MODEL_PARAMS the same way
run_generation.py's equivalent flags override its own BASE_MODEL_PARAMS.

See blackbox/run_generation.py for the script this reuses HTTP/env/rate-limit plumbing from.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_SYNTHESIS_DIR = BASE_DIR.parent
SOURCE_DIRS = {
    "blackbox": DATA_SYNTHESIS_DIR / "blackbox",
    "feature_guided": DATA_SYNTHESIS_DIR / "feature_guided",
    "hybrid": DATA_SYNTHESIS_DIR / "hybrid",
}

# The shared HTTP/env/rate-limit plumbing lives in blackbox/run_generation.py.
sys.path.insert(0, str(SOURCE_DIRS["blackbox"]))

from run_generation import (  # noqa: E402
    DEFAULT_ENV_FILE,
    FP8_DATA_DENY_MODEL_ID_FRAGMENTS,
    MAX_RETRY_BACKOFF_SECONDS,
    RateLimiter,
    call_openrouter_chat,
    load_dotenv,
    load_json_list,
    save_json,
)

# Hardcoded analogously to run_generation.py's MODEL_ID/PROVIDER - independent of that
# script's choice. --model selects one of these by keyword: keyword -> (model id, provider).
MODEL_PRESETS = {
    "gpt": ("openai/gpt-4o-mini-2024-07-18", "openai"),
    "deepseek": ("deepseek/deepseek-v4-flash-0731", "baseten/fp8"),
}
DEFAULT_MODEL = "gpt"

# If True, every request is pinned to --provider (via provider.only/allow_fallbacks=False), as
# before. If False, requests are left free to be routed by OpenRouter across any provider that
# satisfies provider.require_parameters (i.e. load-balanced across all eligible providers).
PIN_PROVIDER = False

# The deepseek-v4-flash family is the only model this "reasoning": {"enabled": False} override
# is known to be needed/supported for; other models must not get it set.
DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT = "deepseek-v4-flash"

MAX_UNPARSED_RETRIES = 3

LABELING_MODEL_PARAMS = {
    "temperature": 0.0,
    # 32 was enough for a short "<prefix>: <label>" line (e.g. toxicity_detection's
    # "Answer: safe"); a full CVSS vector line is longer and got truncated mid-field
    # at 32 (e.g. ".../A:" with the final letter cut off), so it needs more headroom.
    "max_tokens": 64,
    "usage": {"include": True},
}


def resolve_domain_dir(source: str, path_arg: str) -> Path:
    """Resolves --path to a domain dir under <source>/ (e.g. blackbox/toxicity_detection),
    or takes it as-is if it is already a full/relative path to an existing directory.
    """
    candidate = Path(path_arg)
    under_source = SOURCE_DIRS[source] / candidate
    if under_source.is_dir():
        return under_source
    if candidate.is_dir():
        return candidate.resolve()
    raise SystemExit(
        f"--path '{path_arg}' does not resolve to a directory (tried '{under_source}' and '{candidate}')."
    )


def load_prompt_step_2_module(domain_name: str):
    prompt_path = BASE_DIR / domain_name / "prompt" / "prompt_step_2.py"
    if not prompt_path.exists():
        raise SystemExit(f"Expected prompt module at {prompt_path}, but it does not exist.")
    spec = importlib.util.spec_from_file_location(f"prompt_step_2_{domain_name}", prompt_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for attr in ("SYSTEM_PROMPT", "LABEL_FRAGMENTS"):
        if not hasattr(module, attr):
            raise SystemExit(f"{prompt_path} is missing required attribute '{attr}'.")
    # Optional: prepended to the text in the user message (e.g. cti_vsp's "CVE Description: ").
    # Domains that don't set it (e.g. toxicity_detection) get the bare text, as before.
    user_prompt_prefix = getattr(module, "USER_PROMPT_PREFIX", "")
    return module.SYSTEM_PROMPT, module.LABEL_FRAGMENTS, user_prompt_prefix


def resolve_input_json(domain_dir: Path, path_arg: Path) -> Path:
    candidates = [path_arg, domain_dir / path_arg, domain_dir / "output" / path_arg.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"--input-json '{path_arg}' does not resolve to an existing file (tried {candidates}).")


def output_prefix_from_input(input_json: Path) -> str:
    stem = input_json.stem
    suffix = "_accepted"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def find_log_file(domain_dir: Path, prefix: str) -> Path:
    log_path = domain_dir / "log" / f"{prefix}_log.json"
    if not log_path.exists():
        raise SystemExit(f"Expected log file at {log_path}, but it does not exist.")
    return log_path


def load_seed_labels(domain_dir: Path, prefix: str) -> dict[int, str]:
    """Maps seed_example_id -> already-labeled answer line (e.g. "Answer: safe"), read
    from the seed_group TSV used by the run that produced <prefix>_accepted.json (its
    path is recorded in the first run entry of <prefix>_log.json).
    """
    log_path = find_log_file(domain_dir, prefix)
    log_data = json.loads(log_path.read_text(encoding="utf-8"))
    runs = log_data.get("runs") or []
    if not runs:
        raise SystemExit(f"{log_path} has no recorded runs; cannot determine the seed_group TSV used.")
    seed_file = Path(runs[0]["seed_file"])
    if not seed_file.exists():
        raise SystemExit(f"Seed file recorded in {log_path} no longer exists: {seed_file}")

    labels: dict[int, str] = {}
    with open(seed_file, "r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.reader(f, delimiter="\t")):
            if len(row) < 2 or not row[0].strip():
                continue
            labels[i] = row[1].strip()
    return labels


def load_accepted_entries(accepted_path: Path) -> tuple[list[dict], list[dict]]:
    """Splits <prefix>_accepted.json into (seed entries, synthetic entries), preserving
    the order they appear in the file.
    """
    entries = load_json_list(accepted_path)
    if not entries:
        raise SystemExit(f"No entries found in {accepted_path}.")
    seed_entries = [e for e in entries if e.get("type") == "seed"]
    synthetic_entries = [e for e in entries if e.get("type") != "seed"]
    return seed_entries, synthetic_entries


def load_existing_labels(out_path: Path, fragments: dict, answer_regex: re.Pattern) -> dict[str, str]:
    """Maps text -> label for every non-empty label already written to a prior run's
    output TSV, so a resumed run can skip re-labeling (and skip an API call for) anything
    already labeled. A blank label (a prior failed request or unparsed response) is treated
    as not-yet-labeled and retried.

    Each label is re-normalized through normalize_label_line() against the current
    fragments/regex before being reused, so a formatting change to prompt_step_2.py's
    template is also applied to labels written by an earlier run, not just
    newly-labeled ones.
    """
    if not out_path.exists():
        return {}
    labels: dict[str, str] = {}
    with open(out_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2 or not row[0]:
                continue
            text, label = row[0], row[1].strip()
            if not label:
                continue
            normalized = normalize_label_line(label, fragments, answer_regex)
            if normalized is None:
                print(f"  [warn] Could not re-normalize existing label {label!r} for {text[:80]!r}; will re-label.", file=sys.stderr)
                continue
            labels[text] = normalized
    return labels


def build_vector_regex(fragments: dict) -> re.Pattern:
    """Builds a regex for a multi-field template like the CVSS vector format
    ("CVSS:3.1/AV:{AV}/AC:{AC}/..."), where each {FIELD} placeholder in
    fragments["template"] must match one of fragments["fields"][FIELD].
    """
    pattern = re.escape(fragments["template"])
    for field in fragments["field_order"]:
        placeholder = re.escape("{" + field + "}")
        alternatives = "|".join(re.escape(value) for value in fragments["fields"][field])
        pattern = pattern.replace(placeholder, f"(?P<{field}>{alternatives})", 1)
    return re.compile(rf"^\s*{pattern}\s*$", re.IGNORECASE)


def build_answer_regex(fragments: dict) -> re.Pattern:
    if fragments.get("kind") == "vector":
        return build_vector_regex(fragments)
    prefix = re.escape(fragments["prefix"])
    labels = "|".join(re.escape(label) for label in fragments["labels"])
    return re.compile(rf"^\s*{prefix}\s*:\s*({labels})\s*$", re.IGNORECASE)


def normalize_label_line(raw_text: str, fragments: dict, answer_regex: re.Pattern) -> str | None:
    """Extracts the label from a raw model completion and re-renders it in the exact
    format the seed TSVs use, regardless of incidental whitespace/casing/extra-text
    differences in what the model actually returned.

    Two shapes are supported, selected by fragments.get("kind"):
    - default ("enum"): a single "<prefix>: <label>" line, <label> drawn from a fixed
      list (e.g. toxicity_detection's "Answer: safe" / "Answer: toxic").
    - "vector": a multi-field template (e.g. cti_vsp's CVSS v3.1 Base vector), where
      each field is independently drawn from its own fixed set of letters and the
      whole line is re-rendered from fragments["template"] with the matched values.
    """
    for line in raw_text.strip().splitlines():
        match = answer_regex.match(line.strip())
        if not match:
            continue
        if fragments.get("kind") == "vector":
            values = {field: match.group(field).upper() for field in fragments["field_order"]}
            return fragments["template"].format(**values)
        return f"{fragments['prefix']}: {match.group(1).lower()}"
    return None


@dataclass
class WorkerContext:
    """Read-only bundle shared across worker threads; nothing here is mutated after construction."""

    api_key: str
    model: str
    model_params: dict
    system_prompt: str
    user_prompt_prefix: str
    fragments: dict
    answer_regex: re.Pattern
    max_retry_seconds: float
    request_timeout: float
    rate_limiter: RateLimiter


@dataclass
class CallResult:
    """Plain result of one worker's API call for one synthetic entry."""

    index: int
    text: str
    ok: bool
    label_line: str = ""
    error: str | None = None
    unparsed: bool = False
    raw_response: str = ""


def fetch_one(ctx: WorkerContext, index: int, text: str) -> CallResult:
    """Runs on a worker thread: labels one text via the API. Independent of every other
    call - no shared mutable state is read or written here.
    """
    messages = [
        {"role": "system", "content": ctx.system_prompt},
        {"role": "user", "content": f"{ctx.user_prompt_prefix}{text}"},
    ]
    try:
        response = call_openrouter_chat(
            ctx.api_key, ctx.model, messages, ctx.model_params, ctx.max_retry_seconds, ctx.request_timeout, ctx.rate_limiter
        )
    except RuntimeError as exc:
        return CallResult(index=index, text=text, ok=False, error=str(exc))

    raw_text = response["choices"][0]["message"].get("content") or ""
    label_line = normalize_label_line(raw_text, ctx.fragments, ctx.answer_regex)
    if label_line is None:
        return CallResult(index=index, text=text, ok=True, label_line="", unparsed=True, raw_response=raw_text)
    return CallResult(index=index, text=text, ok=True, label_line=label_line, raw_response=raw_text)


def run_batch(ctx: WorkerContext, indices_and_texts: list[tuple[int, str]], max_workers: int) -> list[CallResult]:
    """Dispatches one round of independent labeling calls across a thread pool and waits
    for all of them (success, failure, or unparsed all count as "done") before returning.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_one, ctx, idx, text) for idx, text in indices_and_texts]
        wait(futures, return_when=ALL_COMPLETED)
        return [future.result() for future in futures]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label synthetic blackbox/feature_guided examples for one domain via the OpenRouter API, "
        "with the domain's labeling/<domain>/prompt/prompt_step_2.py system prompt. Model/provider/params are "
        "independent of run_generation.py's - --model picks one of this script's MODEL_PRESETS "
        f"({', '.join(f'{k}={v[0]}' for k, v in MODEL_PRESETS.items())})."
    )
    parser.add_argument(
        "--source", type=str, required=True, choices=sorted(SOURCE_DIRS),
        help="Generation method the input JSON comes from: 'blackbox', 'feature_guided', or 'hybrid'.",
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Domain subfolder (e.g. 'toxicity_detection'), either a name under "
        "<source>/ or a full/relative path to it. Output goes to labeling/<domain name>/.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, choices=sorted(MODEL_PRESETS),
        help="Labeling model keyword, mapped to an OpenRouter model id + provider via MODEL_PRESETS "
        f"({', '.join(f'{k}={v[0]}' for k, v in MODEL_PRESETS.items())}; default: %(default)s).",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="OpenRouter provider slug to pin via provider.only when PIN_PROVIDER is set "
        "(default: the --model preset's provider).",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override LABELING_MODEL_PARAMS['temperature'].")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override LABELING_MODEL_PARAMS['max_tokens'].")
    parser.add_argument(
        "--extra-params", type=str, default=None,
        help="Additional OpenRouter request body parameters as a JSON object string.",
    )
    parser.add_argument(
        "--input-json", type=Path, required=True,
        help="Accepted-pool JSON to label, as produced by <source>/run_generation.py (e.g. "
        "'test_run_01_accepted.json' or 'output/test_run_01_accepted.json'). Only "
        "non-seed entries are sent to the API; seed entries are pulled pre-labeled from "
        "the seed_group TSV recorded in the matching log file.",
    )
    parser.add_argument(
        "--n", type=int, default=0,
        help="Only label the first n non-seed examples (0 = all).",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE, help="Path to a .env file providing OPENROUTER_API_KEY.")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-retry-seconds", type=float, default=600.0,
        help="Total wall-clock budget (seconds) to keep retrying a single failed HTTP request, with "
        "exponentially increasing backoff between attempts (capped at %ds per wait), before giving up, "
        "logging the call as an error, and moving on (default: %%(default)s = 10 minutes)." % MAX_RETRY_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--requests-per-second", type=float, default=None,
        help="Optional cap on request start rate, to respect the API's tps limit (default: unlimited).",
    )
    parser.add_argument(
        "--max-concurrent-requests", type=int, default=1,
        help="Number of worker threads (default: 1). Labeling calls are independent of "
        "each other, so - unlike run_generation.py's generation calls - all of them are "
        "simply submitted to the pool up front; results are written out in the original "
        "entry order regardless of completion order.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    load_dotenv(args.env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(f"OPENROUTER_API_KEY is not set. Set it in the environment or in {args.env_file}.")

    domain_dir = resolve_domain_dir(args.source, args.path)
    system_prompt, fragments, user_prompt_prefix = load_prompt_step_2_module(domain_dir.name)
    answer_regex = build_answer_regex(fragments)

    input_json = resolve_input_json(domain_dir, args.input_json)
    prefix = output_prefix_from_input(input_json)

    seed_entries, synthetic_entries = load_accepted_entries(input_json)
    seed_labels = load_seed_labels(domain_dir, prefix)

    seed_rows: list[tuple[str, str]] = []
    for entry in seed_entries:
        seed_id = entry.get("seed_example_id")
        label = seed_labels.get(seed_id)
        if label is None:
            raise SystemExit(f"No pre-labeled answer found for seed_example_id={seed_id} (text: {entry['text']!r}).")
        seed_rows.append((entry["text"], label))

    if args.n > 0:
        synthetic_entries = synthetic_entries[: args.n]

    output_dir = BASE_DIR / domain_dir.name
    print(f"Source: {args.source} | domain: {domain_dir} | input: {input_json} | output dir: {output_dir}")
    print(f"{len(seed_rows)} pre-labeled seed example(s), {len(synthetic_entries)} synthetic example(s) to label")
    model_id, provider = MODEL_PRESETS[args.model]
    if args.provider is not None:
        provider = args.provider
    print(f"Model: {args.model} -> {model_id} | provider: {provider}")

    model_params = dict(LABELING_MODEL_PARAMS)
    if DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT in model_id:
        model_params["reasoning"] = {"enabled": False}
    if args.temperature is not None:
        model_params["temperature"] = args.temperature
    if args.max_tokens is not None:
        model_params["max_tokens"] = args.max_tokens
    if args.extra_params:
        model_params.update(json.loads(args.extra_params))
    if PIN_PROVIDER:
        model_params["provider"] = {
            **model_params.get("provider", {}),
            "only": [provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    else:
        model_params["provider"] = {
            **model_params.get("provider", {}),
            "require_parameters": True,
        }
    if any(fragment in model_id for fragment in FP8_DATA_DENY_MODEL_ID_FRAGMENTS):
        model_params["provider"] = {
            **model_params["provider"],
            "quantizations": ["fp8"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        }

    ctx = WorkerContext(
        api_key=api_key,
        model=model_id,
        model_params=model_params,
        system_prompt=system_prompt,
        user_prompt_prefix=user_prompt_prefix,
        fragments=fragments,
        answer_regex=answer_regex,
        max_retry_seconds=args.max_retry_seconds,
        request_timeout=args.request_timeout,
        rate_limiter=RateLimiter(args.requests_per_second),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{prefix}.tsv"
    failed_path = output_dir / f"{prefix}_failed.json"
    failed_entries = load_json_list(failed_path)

    def record_failure(round_label: str, result: CallResult) -> None:
        failed_entries.append(
            {
                "round": round_label,
                "index": result.index,
                "text": result.text,
                "reason": "unparsed" if result.unparsed else "api_error",
                "raw_response": result.raw_response if result.unparsed else None,
                "error": result.error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    existing_labels = load_existing_labels(out_path, fragments, answer_regex)
    results_by_index: dict[int, str] = {}
    to_label: list[tuple[int, dict]] = []
    for idx, entry in enumerate(synthetic_entries):
        reused_label = existing_labels.get(entry["text"])
        if reused_label is not None:
            results_by_index[idx] = reused_label
        else:
            to_label.append((idx, entry))
    if existing_labels:
        print(f"Resuming from {out_path}: {len(results_by_index)} already labeled, {len(to_label)} left to label")

    if to_label:
        print("=== Full request messages (first labeling call) ===")
        print(f"--- system ---\n{system_prompt}")
        print(f"--- user ---\n{user_prompt_prefix}{to_label[0][1]['text']}")
        print("=== End full request messages ===")

    n_failed = 0
    n_unparsed = 0
    unresolved_indices: list[int] = []

    results = run_batch(ctx, [(idx, entry["text"]) for idx, entry in to_label], args.max_concurrent_requests)
    for n_done, result in enumerate(results, start=1):
        if not result.ok:
            print(f"  [error] [{n_done}/{len(results)}] {result.error}", file=sys.stderr)
            record_failure("initial", result)
            n_failed += 1
            unresolved_indices.append(result.index)
        elif result.unparsed:
            print(f"  [warn] [{n_done}/{len(results)}] Could not parse a label for: {result.text[:80]!r}", file=sys.stderr)
            print(f"    raw model output: {result.raw_response!r}", file=sys.stderr)
            record_failure("initial", result)
            n_unparsed += 1
            unresolved_indices.append(result.index)
        else:
            print(f"  [ok] [{n_done}/{len(results)}] {result.label_line}: {result.text[:80]!r}")
            results_by_index[result.index] = result.label_line

    for attempt in range(1, MAX_UNPARSED_RETRIES + 1):
        if not unresolved_indices:
            break
        print(f"Retrying {len(unresolved_indices)} unresolved sample(s) (attempt {attempt}/{MAX_UNPARSED_RETRIES})...")
        retry_results = run_batch(
            ctx, [(idx, synthetic_entries[idx]["text"]) for idx in unresolved_indices], args.max_concurrent_requests
        )
        still_unresolved: list[int] = []
        for n_done, result in enumerate(retry_results, start=1):
            if not result.ok:
                print(f"  [error] [retry {attempt}, {n_done}/{len(retry_results)}] {result.error}", file=sys.stderr)
                record_failure(f"retry_{attempt}", result)
                still_unresolved.append(result.index)
            elif result.unparsed:
                print(f"  [warn] [retry {attempt}, {n_done}/{len(retry_results)}] Could not parse a label for: {result.text[:80]!r}", file=sys.stderr)
                print(f"    raw model output: {result.raw_response!r}", file=sys.stderr)
                record_failure(f"retry_{attempt}", result)
                still_unresolved.append(result.index)
            else:
                print(f"  [ok] [retry {attempt}, {n_done}/{len(retry_results)}] {result.label_line}: {result.text[:80]!r}")
                results_by_index[result.index] = result.label_line
        unresolved_indices = still_unresolved

    n_majority_fallback = len(unresolved_indices)
    if unresolved_indices:
        label_counts = Counter(results_by_index.values()) + Counter(label for _, label in seed_rows)
        label_counts.pop("", None)
        if not label_counts:
            raise SystemExit(
                "Cannot assign a fallback label to the remaining unresolved sample(s): "
                "no successfully labeled data point exists to take a majority label from."
            )
        majority_label, _ = label_counts.most_common(1)[0]
        print(
            f"[warn] {n_majority_fallback} sample(s) still unresolved after {MAX_UNPARSED_RETRIES} retries; "
            f"assigning majority label {majority_label!r} to them.",
            file=sys.stderr,
        )
        for idx in unresolved_indices:
            results_by_index[idx] = majority_label

    synthetic_rows = [(entry["text"], results_by_index[idx]) for idx, entry in enumerate(synthetic_entries)]

    with out_path.open("w", encoding="utf-8", newline="") as out_fh:
        writer = csv.writer(out_fh, delimiter="\t", lineterminator="\n")
        for text, label in seed_rows + synthetic_rows:
            writer.writerow([text, label])

    save_json(failed_path, failed_entries)
    print(f"Wrote {len(seed_rows)} seed + {len(synthetic_rows)} synthetic labeled example(s) to {out_path}")
    if failed_entries:
        print(f"Wrote {len(failed_entries)} failed/unparsed attempt record(s) to {failed_path}")
    if n_failed:
        print(f"  failed_requests (first attempt): {n_failed}", file=sys.stderr)
    if n_unparsed:
        print(f"  unparsed_responses (first attempt): {n_unparsed}", file=sys.stderr)
    if n_majority_fallback:
        print(f"  majority_label_fallback (after {MAX_UNPARSED_RETRIES} retries): {n_majority_fallback}", file=sys.stderr)


if __name__ == "__main__":
    main()
