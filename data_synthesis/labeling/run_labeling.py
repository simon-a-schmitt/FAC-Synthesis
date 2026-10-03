"""Label synthetic examples for one domain (e.g. toxicity_detection) via the OpenRouter
API, produced by blackbox/run_generation.py, feature_guided/run_generation.py, or
hybrid/run_generation.py.

--source selects the generation method (blackbox, feature_guided, or hybrid); together
with --domain this gives the source domain dir <source>/<domain>. Reads
<source>/<domain>/output/<prefix>_accepted.json (a JSON list of seed + synthetic entries,
same format for both methods) and asks an OpenRouter model to label every non-seed
("type" != "seed") entry's text, using the system prompt from
prompts/<domain>/labeling.py (SYSTEM_PROMPT / LABEL_FRAGMENTS / USER_PROMPT_PREFIX) - shared by
all sources, so the same labeling prompt is used regardless of generation method. Seed entries
are not re-labeled - their label already exists in the seed_group TSV the original
generation run drew from (benchmarks/<domain>/<seed set>/, found via the seed_set/seed_group
recorded in <source>/<domain>/log/<prefix>_log.json), already stored in the exact "<prefix>: <label>"
format the model is asked to produce, so seed and model-produced labels end up formatted
identically in the output.

Writes one TSV to labeling/<domain>/<prefix>.tsv next to this script (prefix = the input
filename with its trailing "_accepted" stripped, domain = the source domain dir's name) -
one prompt/label pair per line, tab-separated, seed examples first (in seed order)
followed by the labeled synthetic examples (in accepted-pool order, regardless of
completion order - see fetch_one/main).

Requests run concurrently via a thread pool (shared/openrouter.py's RateLimiter respects the
API's tps limit). Unlike the generation calls, labeling calls
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

Every API call (initial and retry rounds, successful or not) is also recorded in a per-prefix
run log at labeling/<domain>/log/<prefix>_log.json, analogous to run_generation.py's: one
entry per invocation of this script under "runs" (run_id, model/params, counts, token usage,
the endpoint catalog snapshot, and one record per call with its OpenRouter response metadata
incl. usage and its sampling_verification), plus cumulative token totals across all runs.
summarize_labeling_run.py turns this log into an overview.

Model, provider, and sampling params are independent of the generator's: --model picks one of
the MODEL_PRESETS at the top of this file ("gpt" or "deepseek"). --provider/--temperature/
--max-tokens/--extra-params override the preset's provider/LABELING_MODEL_PARAMS, built by the
same shared.openrouter.build_model_params as the generation request parameters.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import uuid
from collections import Counter
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

BASE_DIR = Path(__file__).resolve().parent
DATA_SYNTHESIS_DIR = BASE_DIR.parent
sys.path.insert(0, str(DATA_SYNTHESIS_DIR))

from prompts import load_labeling_prompt  # noqa: E402
from shared.benchmarks import (  # noqa: E402
    DEFAULT_SEED_SET,
    DOMAINS,
    find_seed_file,
    find_seed_file_by_name,
    load_seed_labels,
)
from shared.openrouter import (  # noqa: E402
    EndpointCatalog,
    RateLimiter,
    add_api_args,
    build_model_params,
    call_openrouter_chat,
    require_api_key,
    response_meta,
    summarize_sampling_verification,
    verify_call,
)
from shared.run_io import (  # noqa: E402
    append_run_log,
    format_wall_clock_slurm,
    load_json_dict,
    load_json_list,
    save_json,
    utc_now,
)

SOURCES = ("blackbox", "feature_guided", "hybrid")

# Labeling models, independent of the generator: --model keyword -> (model id, provider).
MODEL_PRESETS = {
    "gpt": ("openai/gpt-4o-mini-2024-07-18", "openai"),
    "deepseek": ("deepseek/deepseek-v4-flash-0731", "baseten/fp8"),
}

MAX_UNPARSED_RETRIES = 3

LABELING_MODEL_PARAMS = {
    "temperature": 0.0,
    # 32 was enough for a short "<prefix>: <label>" line (e.g. toxicity_detection's
    # "Answer: safe"); a full CVSS vector line is longer and got truncated mid-field
    # at 32 (e.g. ".../A:" with the final letter cut off), so it needs more headroom.
    "max_tokens": 64,
    "usage": {"include": True},
}


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


def resolve_seed_file(domain: str, domain_dir: Path, prefix: str) -> Path:
    """The seed_group TSV the run that produced <prefix>_accepted.json drew from, resolved in
    benchmarks/ from the seed_set/seed_group of the first run in <prefix>_log.json (older logs
    without seed_set used the k5 seed groups; logs without seed_group are matched by file name)."""
    log_path = domain_dir / "log" / f"{prefix}_log.json"
    runs = (load_json_dict(log_path) or {}).get("runs") or []
    if not runs:
        raise SystemExit(f"{log_path} is missing or has no recorded runs; cannot determine the seed group used.")
    run = runs[0]
    if run.get("seed_group"):
        return find_seed_file(domain, run.get("seed_set", DEFAULT_SEED_SET), run["seed_group"])
    name = PureWindowsPath(run["seed_file"]).name  # also splits the Windows paths of old logs
    seed_file = find_seed_file_by_name(domain, name)
    if seed_file is None:
        raise SystemExit(f"Seed file {name!r} recorded in {log_path} not found under benchmarks/{domain}/.")
    return seed_file


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
    fragments/regex before being reused, so a formatting change to prompts/<domain>/labeling.py's
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
    openrouter_meta: dict = field(default_factory=dict)


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
    openrouter_meta = response_meta(response)
    label_line = normalize_label_line(raw_text, ctx.fragments, ctx.answer_regex)
    if label_line is None:
        return CallResult(
            index=index, text=text, ok=True, label_line="", unparsed=True, raw_response=raw_text,
            openrouter_meta=openrouter_meta,
        )
    return CallResult(
        index=index, text=text, ok=True, label_line=label_line, raw_response=raw_text, openrouter_meta=openrouter_meta
    )


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
        description="Label the synthetic examples of one generation run via the OpenRouter API, with the "
        "domain's prompts/<domain>/labeling.py prompt."
    )
    parser.add_argument("--source", type=str, required=True, choices=SOURCES,
                        help="Generation arm the input JSON comes from.")
    parser.add_argument("--domain", type=str, required=True, choices=DOMAINS)
    parser.add_argument("--model", type=str, required=True, choices=sorted(MODEL_PRESETS),
                        help="Labeling model: " + ", ".join(f"{k}={v[0]}" for k, v in MODEL_PRESETS.items()))
    parser.add_argument("--provider", type=str, default=None,
                        help="Provider slug to pin when PIN_PROVIDER is set (default: the preset's provider).")
    parser.add_argument(
        "--input-json", type=Path, required=True,
        help="Accepted-pool JSON to label (e.g. 'cti_vsp_bb_deepseek_01_accepted.json', looked up in "
        "<source>/<domain>/output/). Only non-seed entries are sent to the API.",
    )
    parser.add_argument("--n", type=int, default=0, help="Only label the first n non-seed examples (0 = all).")
    add_api_args(parser)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    started_at = utc_now()
    run_id = uuid.uuid4().hex
    api_key = require_api_key(args.env_file)

    domain_dir = DATA_SYNTHESIS_DIR / args.source / args.domain
    prompt = load_labeling_prompt(args.domain)
    system_prompt, fragments, user_prompt_prefix = prompt.system, prompt.fragments, prompt.user_prefix
    answer_regex = build_answer_regex(fragments)

    input_json = resolve_input_json(domain_dir, args.input_json)
    prefix = output_prefix_from_input(input_json)

    seed_entries, synthetic_entries = load_accepted_entries(input_json)
    seed_file = resolve_seed_file(args.domain, domain_dir, prefix)
    seed_labels = load_seed_labels(seed_file)

    seed_rows: list[tuple[str, str]] = []
    for entry in seed_entries:
        seed_id = entry.get("seed_example_id")
        if seed_id is None or not 0 <= seed_id < len(seed_labels):
            raise SystemExit(f"No label in {seed_file} for seed_example_id={seed_id} (text: {entry['text']!r}).")
        seed_rows.append((entry["text"], seed_labels[seed_id]))

    if args.n > 0:
        synthetic_entries = synthetic_entries[: args.n]

    output_dir = BASE_DIR / args.domain
    print(f"Source: {args.source} | domain: {domain_dir} | input: {input_json} | seeds: {seed_file} | output dir: {output_dir}")
    print(f"{len(seed_rows)} pre-labeled seed example(s), {len(synthetic_entries)} synthetic example(s) to label")
    model_id, provider = MODEL_PRESETS[args.model]
    if args.provider is not None:
        provider = args.provider
    print(f"Model: {args.model} -> {model_id} | provider: {provider}")
    model_params = build_model_params(model_id, provider, LABELING_MODEL_PARAMS, args)

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

    endpoint_catalog = EndpointCatalog(model_id, args.request_timeout)
    endpoint_catalog.refresh()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{prefix}.tsv"
    failed_path = output_dir / f"{prefix}_failed.json"
    log_path = output_dir / "log" / f"{prefix}_log.json"
    failed_entries = load_json_list(failed_path)

    call_records: list[dict] = []
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0}

    def record_call(round_label: str, result: CallResult) -> None:
        """Appends one log record per API call (main thread only - EndpointCatalog isn't thread-safe)."""
        call_id = f"{run_id}_{len(call_records) + 1:06d}"
        if not result.ok:
            status = "api_error"
        elif result.unparsed:
            status = "unparsed"
        else:
            status = "labeled"
        record = {
            "call_id": call_id,
            "call_number": len(call_records) + 1,
            "round": round_label,
            "index": result.index,
            "text": result.text,
            "status": status,
            "label": result.label_line or None,
            "raw_response": result.raw_response if result.ok else None,
            "error": result.error,
        }
        if result.ok:
            usage = result.openrouter_meta.get("usage") or {}
            token_totals["prompt_tokens"] += usage.get("prompt_tokens") or 0
            token_totals["completion_tokens"] += usage.get("completion_tokens") or 0
            record["openrouter_response"] = result.openrouter_meta
            record["sampling_verification"] = verify_call(
                call_id, model_params, result.openrouter_meta.get("provider"), endpoint_catalog
            )
        call_records.append(record)

    def record_failure(round_label: str, result: CallResult) -> None:
        failed_entries.append(
            {
                "round": round_label,
                "index": result.index,
                "text": result.text,
                "reason": "unparsed" if result.unparsed else "api_error",
                "raw_response": result.raw_response if result.unparsed else None,
                "error": result.error,
                "timestamp": utc_now(),
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
        record_call("initial", result)
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
            record_call(f"retry_{attempt}", result)
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
    majority_label = None
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

    finished_at = utc_now()
    successful_calls = [c for c in call_records if c["status"] != "api_error"]
    prompt_tokens = token_totals["prompt_tokens"]
    completion_tokens = token_totals["completion_tokens"]
    run_entry = {
        "run_id": run_id,
        "source": args.source,
        "domain": args.domain,
        "seed_file": str(seed_file),
        "model": args.model,
        "model_id": model_id,
        "provider": provider,
        "path": str(domain_dir),
        "input_json": str(input_json),
        "n_seed": len(seed_rows),
        "n_synthetic": len(synthetic_entries),
        "n_reused": len(synthetic_entries) - len(to_label),
        "n_to_label": len(to_label),
        "n_calls": len(call_records),
        "n_failed_calls": len(call_records) - len(successful_calls),
        "n_unparsed_calls": sum(1 for c in call_records if c["status"] == "unparsed"),
        "n_failed_first_attempt": n_failed,
        "n_unparsed_first_attempt": n_unparsed,
        "n_majority_fallback": n_majority_fallback,
        "majority_label": majority_label,
        "max_unparsed_retries": MAX_UNPARSED_RETRIES,
        "synthetic_label_counts": dict(Counter(label for _, label in synthetic_rows)),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "model_params": model_params,
        "sampling_verification_summary": summarize_sampling_verification(successful_calls),
        "endpoint_catalog": endpoint_catalog.snapshot(),
        "endpoint_catalog_fetched_at": endpoint_catalog.fetched_at,
        "output_file": str(out_path),
        "failed_file": str(failed_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_clock_time": format_wall_clock_slurm(started_at, finished_at),
        "calls": call_records,
    }
    append_run_log(log_path, prefix, str(domain_dir), run_entry)
    print(f"Tokens: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")


if __name__ == "__main__":
    main()
