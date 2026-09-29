"""Generate synthetic blackbox examples for one domain (cti_vsp / claudette_tos /
toxicity_detection) by prompting the generator model selected via --model (see MODEL_PRESETS)
via the OpenRouter API.

Loads the system/user prompt template from <path>/prompts/prompt_step_1.py and the seed
examples from data/seed_groups/<domain>/*_seed_group_<seed-group>.tsv. Seed examples and the accepted
pool are two distinct collections: each generation call is given 3 context examples - 2 drawn
from the seed examples plus 1 drawn from the accepted pool (or 3 seed examples, as long as the
accepted pool is still empty). Each freshly generated candidate is compared via ROUGE-L
F-measure against seed examples + the accepted pool (everything accepted so far, including in
earlier runs) and rejected if it is too similar to something already there. Accepted/rejected
samples are written to <path>/output (accepted.json contains both the seed examples and every
accepted synthetic sample, so it is a complete record of "all prompts"), token usage is logged
to <path>/log. Output and log files are named after --prefix.

Requests run concurrently in waves to work around a low per-request tps limit on the API:
each wave builds a batch of B = 2 * --max-concurrent-requests prompts up front (all drawing
their 3 context examples from the SAME frozen snapshot of the accepted pool taken at the start
of the wave, via a per-slot RNG seeded from SHA-256(run_id, seed_group, arm, wave_idx, slot)),
dispatches them concurrently, waits for the whole wave to finish (a barrier), then applies dedup
sequentially in slot order - never completion order - against seed examples + the accepted pool
+ whatever this same wave has already accepted. The pool is only ever mutated between waves
(i.e. once the wave's barrier has passed), never while a wave is in flight. ARM identifies this
script's generation branch (as opposed to sibling branches such as feature-guided generation)
so that its slot RNG never collides with theirs even under a shared run_id/seed_group.

See run_synthetic_generation.py (ma_synthesis_api root) for the sibling script this is
modeled on.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import json
import os
import random
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rouge_score import tokenizers
from rouge_score.rouge_scorer import _score_lcs

BASE_DIR = Path(__file__).parent
DATA_SYNTHESIS_DIR = BASE_DIR.parent
DEFAULT_ENV_FILE = DATA_SYNTHESIS_DIR / ".env"
# Shared with feature_guided/ and hybrid/: data/seed_groups/<domain>/*_seed_group_<id>.tsv.
SEED_GROUPS_DIR = DATA_SYNTHESIS_DIR / "data" / "seed_groups"

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_ENDPOINTS_URL_TEMPLATE = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"

# --model selects one of these by keyword: keyword -> (model id, provider), as in
# hybrid/run_generation.py and labeling/run_labeling.py.
MODEL_PRESETS = {
    "llama": ("meta-llama/llama-3.1-8b-instruct", "deepinfra/fp8"),
    "deepseek": ("deepseek/deepseek-v4-flash-0731", "baseten/fp8"),
}
DEFAULT_MODEL = "llama"

# Module-level defaults (DEFAULT_MODEL); main() overrides them via configure_model() from --model.
# feature_guided/ and hybrid/ import this module and overwrite these globals themselves.
MODEL_ID, PROVIDER = MODEL_PRESETS[DEFAULT_MODEL]

# If True, every request is pinned to PROVIDER (via provider.only/allow_fallbacks=False), as
# before. If False, PROVIDER is only used as EXPECTED_PROVIDER_NAME's default and requests are
# left free to be routed by OpenRouter across any provider that satisfies
# provider.require_parameters (i.e. load-balanced across all eligible providers).
PIN_PROVIDER = False

# The deepseek-v4-flash family is the only model this "reasoning": {"enabled": False} override
# is known to be needed/supported for; other models (e.g. llama) should not get it set.
DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT = "deepseek-v4-flash"
LLAMA_3_1_8B_INSTRUCT_MODEL_ID_FRAGMENT = "llama-3.1-8b-instruct"

# For these two models, every request is additionally pinned to an fp8 quantization and opts
# out of OpenRouter's prompt data collection, regardless of PIN_PROVIDER above.
FP8_DATA_DENY_MODEL_ID_FRAGMENTS = (
    DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT,
    LLAMA_3_1_8B_INSTRUCT_MODEL_ID_FRAGMENT,
)

# OpenRouter's response echoes back just the provider's display name (e.g. "DeepInfra"),
# not the "<provider>/<quantization>" slug used to pin it in the request.
EXPECTED_PROVIDER_NAME = PROVIDER.split("/")[0]

BASE_MODEL_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 2048,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "usage": {"include": True},
}

if DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT in MODEL_ID:
    BASE_MODEL_PARAMS["reasoning"] = {"enabled": False}


def configure_model(model_keyword: str) -> None:
    """Sets the generator model from --model. Only called from this script's main(), so importing
    this module (feature_guided/, hybrid/) is unaffected."""
    global MODEL_ID, PROVIDER, EXPECTED_PROVIDER_NAME
    MODEL_ID, PROVIDER = MODEL_PRESETS[model_keyword]
    EXPECTED_PROVIDER_NAME = PROVIDER.split("/")[0]
    BASE_MODEL_PARAMS.pop("reasoning", None)
    if DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT in MODEL_ID:
        BASE_MODEL_PARAMS["reasoning"] = {"enabled": False}

VALID_SEED_GROUPS = {"01", "02", "03", "04", "05"}

# Identifies this script's generation branch (as opposed to sibling branches such as a
# feature-guided variant), so its per-slot RNG never collides with theirs even if they
# happen to share a run_id/seed_group namespace.
ARM = "blackbox"

# The context-drawing scheme needs 2 distinct seeds normally, or 3 while the accepted pool
# is still empty - so at least 3 distinct seed examples are required.
MIN_SEED_EXAMPLES = 3


def load_dotenv(env_file: Path) -> None:
    if not env_file.exists():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def resolve_domain_dir(path_arg: str) -> Path:
    candidate = Path(path_arg)
    if candidate.is_absolute() and candidate.is_dir():
        return candidate
    under_base = BASE_DIR / candidate
    if under_base.is_dir():
        return under_base
    if candidate.is_dir():
        return candidate
    raise SystemExit(
        f"--path '{path_arg}' does not resolve to a directory (tried '{under_base}' and '{candidate}')."
    )


def load_prompt_module(domain_dir: Path):
    prompt_path = domain_dir / "prompts" / "prompt_step_1.py"
    if not prompt_path.exists():
        raise SystemExit(f"Expected prompt module at {prompt_path}, but it does not exist.")
    spec = importlib.util.spec_from_file_location(f"prompt_step_1_{domain_dir.name}", prompt_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    for attr in ("SYSTEM_PROMPT", "STEP_1_PROMPT_TEMPLATE"):
        if not hasattr(module, attr):
            raise SystemExit(f"{prompt_path} is missing required attribute '{attr}'.")
    return module.SYSTEM_PROMPT, module.STEP_1_PROMPT_TEMPLATE


def find_seed_file(domain_name: str, seed_group: str) -> Path:
    seed_dir = SEED_GROUPS_DIR / domain_name
    matches = sorted(seed_dir.glob(f"*_seed_group_{seed_group}.tsv"))
    if not matches:
        raise SystemExit(f"No seed file matching '*_seed_group_{seed_group}.tsv' found in {seed_dir}.")
    if len(matches) > 1:
        raise SystemExit(f"Multiple seed files match '*_seed_group_{seed_group}.tsv' in {seed_dir}: {matches}")
    return matches[0]


def load_seed_examples(seed_file: Path) -> list[str]:
    examples = []
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            first_col = line.rstrip("\r\n").split("\t")[0].strip()
            if first_col:
                examples.append(first_col)
    if not examples:
        raise ValueError(f"No seed examples found in {seed_file}")
    return examples


def parse_single_candidate(text: str) -> list[str]:
    """Extracts the one candidate a step-1 prompt call is expected to produce.

    Every domain's STEP_1_PROMPT_TEMPLATE asks for exactly one output ("Write one
    new ... Output the ... only. No numbering, no headers, no blank lines, no
    commentary."), unlike run_synthetic_generation.py's numbered-list multi-example
    prompts. So only the first non-empty line is treated as the candidate (with an
    optional leading "1." / "1)" stripped); any further non-empty lines are almost
    always the model violating that instruction - a stray self-comment, apology, or
    preamble - and are logged as discarded here rather than being fed through dedup
    as if they were additional, independently valid candidates.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return []
    match = re.match(r"^\d+[.)]\s*(.+)$", lines[0])
    candidate = match.group(1).strip() if match else lines[0]
    if len(lines) > 1:
        print(
            f"  [warn] Model returned {len(lines) - 1} extra line(s) beyond the first; "
            f"discarding as commentary (not treated as additional candidates): {lines[1:]!r}",
            file=sys.stderr,
        )
    return [candidate]


def _http_post_json(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class RateLimiter:
    """Thread-safe minimum-interval gate shared across worker threads."""

    def __init__(self, requests_per_second: float | None):
        self._interval = 1.0 / requests_per_second if requests_per_second else 0.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        if not self._interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + self._interval
            wait_for = start - now
        if wait_for > 0:
            time.sleep(wait_for)


MAX_RETRY_BACKOFF_SECONDS = 60.0


def make_call_id(run_id: str, call_number: int, phase: str | None = None) -> str:
    """Join key shared by a call's log record and every accepted/rejected/failed entry it produced.

    call_number alone only counts within one run, so it is namespaced by run_id to stay unique
    across all runs appended to the same output/log files - and by `phase` where one run has
    several call counters (hybrid/: blackbox + feature-guided phase, merged in one accepted file).
    """
    return f"{run_id}_{phase}_{call_number:06d}" if phase else f"{run_id}_{call_number:06d}"


class EndpointCatalog:
    """Cached view of OpenRouter's public endpoint list for MODEL_ID (provider -> quantization etc.).

    OpenRouter's chat completion response (and its /generation stats) only name the provider a
    call was routed to - never the quantization or the sampling params actually applied. The
    quantization is therefore resolved here from the endpoint catalog, by matching the response's
    provider name against endpoints that satisfy the request's provider.quantizations filter.
    Main-thread-only (used from process_call_result); re-fetched once on an unknown provider.
    """

    def __init__(self, model_id: str, timeout: float):
        self._url = OPENROUTER_ENDPOINTS_URL_TEMPLATE.format(model_id=model_id)
        self._timeout = timeout
        self._endpoints: list[dict] | None = None
        self.fetched_at: str | None = None

    def refresh(self) -> None:
        try:
            with urllib.request.urlopen(self._url, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            self._endpoints = data["data"]["endpoints"]
            self.fetched_at = datetime.now(timezone.utc).isoformat()
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, http.client.HTTPException,
                KeyError, json.JSONDecodeError) as exc:
            print(f"  [warn] Could not fetch OpenRouter endpoint catalog from {self._url}: {exc}", file=sys.stderr)

    def snapshot(self) -> list[dict]:
        if self._endpoints is None:
            self.refresh()
        return [
            {
                "provider_name": e.get("provider_name"),
                "tag": e.get("tag"),
                "quantization": e.get("quantization"),
                "supported_parameters": e.get("supported_parameters", []),
            }
            for e in (self._endpoints or [])
        ]

    def lookup(self, provider_name: str | None, allowed_quantizations: list[str] | None) -> list[dict]:
        def matches() -> list[dict]:
            return [
                e for e in self.snapshot()
                if provider_name is not None
                and (e["provider_name"] or "").lower() == provider_name.lower()
                and (not allowed_quantizations or e["quantization"] in allowed_quantizations)
            ]

        found = matches()
        if not found:
            self.refresh()
            found = matches()
        return found


def build_sampling_verification(
    requested_params: dict, response_provider: str | None, catalog: EndpointCatalog
) -> dict:
    """Per-call audit record: what was sent vs. which endpoint (and quantization) served it."""
    provider_prefs = requested_params.get("provider", {})
    allowed_quantizations = provider_prefs.get("quantizations")
    endpoints = catalog.lookup(response_provider, allowed_quantizations)
    quantizations = sorted({e["quantization"] for e in endpoints})
    temperature_supported = bool(endpoints) and all("temperature" in e["supported_parameters"] for e in endpoints)
    top_p_supported = bool(endpoints) and all("top_p" in e["supported_parameters"] for e in endpoints)
    return {
        "requested_temperature": requested_params.get("temperature"),
        "requested_top_p": requested_params.get("top_p"),
        "requested_quantizations": allowed_quantizations,
        "require_parameters": provider_prefs.get("require_parameters"),
        "allow_fallbacks": provider_prefs.get("allow_fallbacks"),
        "routed_provider": response_provider,
        "endpoint_tags": [e["tag"] for e in endpoints],
        "quantization": quantizations[0] if len(quantizations) == 1 else None,
        "endpoint_supports_temperature": temperature_supported,
        "endpoint_supports_top_p": top_p_supported,
        "verified": len(quantizations) == 1 and temperature_supported and top_p_supported,
        "catalog_fetched_at": catalog.fetched_at,
    }


def call_openrouter_chat(
    api_key: str,
    model: str,
    messages: list[dict],
    model_params: dict,
    max_retry_seconds: float,
    timeout: float,
    rate_limiter: RateLimiter | None = None,
) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, **model_params}

    # Besides HTTPError/URLError, a read/connect timeout or a dropped connection can surface
    # as a bare socket.timeout/TimeoutError, ConnectionError, or http.client.HTTPException
    # instead - none of which are URLError subclasses. All of these must be retried too,
    # rather than crashing the worker thread.
    retryable_errors = (
        urllib.error.HTTPError,
        urllib.error.URLError,
        socket.timeout,  # same class as TimeoutError on Python 3.10+, a distinct OSError subclass before that
        TimeoutError,
        ConnectionError,
        http.client.HTTPException,
    )

    # Retries for a single call span up to max_retry_seconds wall-clock time (default 10min),
    # not a fixed attempt count: providers like DeepInfra return transient 429s under shared-pool
    # overload (see engine_overloaded upstream errors) that can take several minutes to clear.
    # Backoff still grows exponentially per attempt, but is capped at MAX_RETRY_BACKOFF_SECONDS so
    # a single sleep can't eat the whole budget and starve later retries.
    deadline = time.monotonic() + max_retry_seconds
    last_error = None
    attempt = 0
    while True:
        attempt += 1
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            return _http_post_json(OPENROUTER_CHAT_URL, headers, payload, timeout)
        except retryable_errors as exc:
            detail = exc.read().decode("utf-8", "replace") if isinstance(exc, urllib.error.HTTPError) else str(exc)
            last_error = f"{exc} - {detail}"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = min(2 ** attempt, MAX_RETRY_BACKOFF_SECONDS, remaining)
            print(
                f"  [warn] OpenRouter chat request failed (attempt {attempt}, "
                f"{remaining:.0f}s left in retry budget): {last_error}. Retrying in {wait:.0f}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"OpenRouter chat request failed after {attempt} attempts over {max_retry_seconds:.0f}s: {last_error}"
    )


def print_raw_first_call(system_prompt: str, user_prompt: str, model_params: dict, derived_seed: int) -> None:
    """Prints the exact request body for this run's first call, for manual inspection.

    Shows the system/user message content twice: once via repr() (so hidden whitespace,
    newlines, and any stray characters are visible as escape sequences, not swallowed by the
    terminal) and once as the literal JSON body that goes out over the wire. Note this is the
    request WE send - OpenRouter/the backend's own chat template still wraps the messages with
    whatever model-specific special tokens (e.g. <|im_start|>, <|begin_of_text|>) it uses when
    turning them into a raw token sequence server-side; that step is not visible from here.
    """
    call_model_params = dict(model_params)
    # "seed" disabled: the baseten/fp8 endpoint for MODEL_ID doesn't support it, and with
    # provider.require_parameters=True that makes OpenRouter reject the call (404, no endpoints
    # left after "Filter by Parameters"). derived_seed is still computed/logged for traceability.
    # call_model_params["seed"] = derived_seed
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    payload = {"model": MODEL_ID, "messages": messages, **call_model_params}

    print("\n" + "=" * 100)
    print("[debug] RAW FIRST CALL OF THIS RUN - what actually goes to the OpenRouter API")
    print("=" * 100)
    print("--- system message (repr) ---")
    print(repr(system_prompt))
    print("--- user message (repr) ---")
    print(repr(user_prompt))
    print("--- full JSON request body (model/messages/params, incl. this call's seed) ---")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("=" * 100 + "\n")


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8").strip()
    return json.loads(content) if content else []


def load_json_dict(path: Path) -> dict | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    return json.loads(content) if content else None


def format_wall_clock_slurm(started_at: str, finished_at: str) -> str:
    """The run's wall-clock time (finished_at - started_at, both ISO timestamps) as HH:MM:SS, matching
    SLURM's own "Job Wall-clock time" format. Note this measures only from inside main() (after arg
    parsing) to just before the run log is written, so it undercounts SLURM's job time, which also
    includes job prologue/epilogue and script startup/shutdown outside our control."""
    elapsed_seconds = round((datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds())
    hours, remainder = divmod(max(elapsed_seconds, 0), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


_ROUGE_TOKENIZER = tokenizers.DefaultTokenizer(use_stemmer=False)


def rouge_tokenize(text: str) -> list[str]:
    return _ROUGE_TOKENIZER.tokenize(text)


def rouge_l_fmeasure(tokens_a: list[str], tokens_b: list[str]) -> float:
    return _score_lcs(tokens_a, tokens_b).fmeasure


def derive_seed(run_id: str, call_number: int) -> int:
    key = f"{run_id}|{call_number}|{MODEL_ID}"
    # Masked to fit a signed int32 (max 2147483647), see run_synthetic_generation.py.
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def derive_slot_rng(run_id: str, seed_group: str, arm: str, wave_idx: int, slot_index: int) -> random.Random:
    """Deterministic RNG for drawing a slot's context examples.

    Seeded independently of derive_seed (which controls the model's own sampling seed) so a
    slot's prompt content depends only on (run, seed group, arm, wave, slot) - not on which
    worker thread happens to execute it, and not on completion timing.
    """
    key = f"{run_id}|{seed_group}|{arm}|{wave_idx}|{slot_index}"
    seed_int = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return random.Random(seed_int)


def pick_context_examples(rng: random.Random, seed_examples: list[str], pool_snapshot: list[dict]) -> list[dict]:
    """Draws the 3 context examples for one slot's prompt.

    Normally 2 distinct seed examples + 1 accepted-pool entry. While the accepted pool
    snapshot is still empty (e.g. the very first wave of the very first run), draws a 3rd
    distinct seed example instead, since there is nothing yet to draw from the pool.
    """
    n_seeds = len(seed_examples)
    if not pool_snapshot:
        seed_indices = rng.sample(range(n_seeds), 3)
        return [{"type": "seed", "seed_example_id": i, "text": seed_examples[i]} for i in seed_indices]

    seed_indices = rng.sample(range(n_seeds), 2)
    pool_entry = pool_snapshot[rng.randrange(len(pool_snapshot))]
    context = [{"type": "seed", "seed_example_id": i, "text": seed_examples[i]} for i in seed_indices]
    context.append({"type": "accepted", "accepted_id": pool_entry["id"], "text": pool_entry["text"]})
    return context


def render_seed_examples_block(context_examples: list[dict]) -> str:
    return "\n".join(f"{i + 1}. {entry['text']}" for i, entry in enumerate(context_examples))


@dataclass
class WorkerContext:
    """Read-only bundle shared across worker threads; nothing here is mutated after construction.

    Prompts are already fully built by the main thread before dispatch (see main()'s wave loop),
    so a worker only needs enough to make the HTTP call - not the seed examples or pool.
    """

    api_key: str
    model_params: dict
    system_prompt: str
    max_retry_seconds: float
    request_timeout: float
    rate_limiter: RateLimiter


@dataclass
class CallResult:
    """Plain result of one worker's API call; carries no reference to the shared pool."""

    call_id: str
    call_number: int
    wave_idx: int
    slot_index: int
    derived_seed: int
    context_examples: list[dict]
    requested_params: dict
    ok: bool
    error: str | None = None
    candidates: list[str] = field(default_factory=list)
    openrouter_meta: dict = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def fetch_one(
    ctx: WorkerContext,
    call_id: str,
    call_number: int,
    wave_idx: int,
    slot_index: int,
    user_prompt: str,
    context_examples: list[dict],
    derived_seed: int,
) -> CallResult:
    """Runs on a worker thread: calls the API with an already-built prompt, parses candidates.

    Never touches the accepted pool - only the main thread does that, after the wave's barrier.
    """
    call_model_params = dict(ctx.model_params)
    # "seed" disabled: see print_raw_first_call() for why.
    # call_model_params["seed"] = derived_seed
    messages = [
        {"role": "system", "content": ctx.system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def failure(error: str) -> CallResult:
        return CallResult(
            call_id=call_id,
            call_number=call_number,
            wave_idx=wave_idx,
            slot_index=slot_index,
            derived_seed=derived_seed,
            context_examples=context_examples,
            requested_params=call_model_params,
            ok=False,
            error=error,
        )

    try:
        response = call_openrouter_chat(
            ctx.api_key, MODEL_ID, messages, call_model_params, ctx.max_retry_seconds, ctx.request_timeout, ctx.rate_limiter
        )
    except RuntimeError as exc:
        return failure(str(exc))

    response_provider = response.get("provider")
    if PIN_PROVIDER:
        assert response_provider is not None and response_provider.lower() == EXPECTED_PROVIDER_NAME.lower(), (
            f"Expected provider '{EXPECTED_PROVIDER_NAME}' (pinned via provider.only=['{PROVIDER}']), "
            f"but OpenRouter routed call {call_number} to '{response_provider}'. Aborting."
        )

    choice = response["choices"][0]
    raw_text = choice["message"].get("content")
    finish_reason = choice.get("finish_reason")
    if raw_text is None:
        print(f"  [warn] Empty message content (finish_reason={finish_reason}); treating as no candidates.", file=sys.stderr)
        raw_text = ""
    usage = response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    candidates = parse_single_candidate(raw_text)
    openrouter_meta = {
        "id": response.get("id"),
        "model": response.get("model"),
        "provider": response.get("provider"),
        "usage": usage,
        "finish_reason": finish_reason,
    }
    return CallResult(
        call_id=call_id,
        call_number=call_number,
        wave_idx=wave_idx,
        slot_index=slot_index,
        derived_seed=derived_seed,
        context_examples=context_examples,
        requested_params=call_model_params,
        ok=True,
        candidates=candidates,
        openrouter_meta=openrouter_meta,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def process_call_result(
    result: CallResult,
    *,
    run_id: str,
    args: argparse.Namespace,
    seed_tokens: list[list[str]],
    accepted_pool: list[dict],
    pool_tokens: list[list[str]],
    rejected_entries: list[dict],
    failed_entries: list[dict],
    call_records: list[dict],
    counters: dict,
    endpoint_catalog: EndpointCatalog,
) -> None:
    """Applies one call's result to the shared pool. Main-thread-only, called strictly in slot
    order for a wave - so a candidate is deduped against the pool snapshot plus whatever this
    same wave has already accepted (earlier slots), never against later, not-yet-applied slots.
    """
    context_texts = [c["text"] for c in result.context_examples]

    if not result.ok:
        print(f"  [error] {result.error}", file=sys.stderr)
        counters["n_failed_calls"] += 1
        failed_entries.append(
            {
                "run_id": run_id,
                "call_id": result.call_id,
                "call_number": result.call_number,
                "wave_idx": result.wave_idx,
                "slot_index": result.slot_index,
                "seed_sent": result.derived_seed,
                "context_examples": context_texts,
                "error": result.error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return

    counters["total_prompt_tokens"] += result.prompt_tokens
    counters["total_completion_tokens"] += result.completion_tokens

    sampling_verification = build_sampling_verification(
        result.requested_params, result.openrouter_meta.get("provider"), endpoint_catalog
    )
    if not sampling_verification["verified"]:
        print(
            f"  [warn] Call {result.call_id} could not be verified against the endpoint catalog "
            f"(provider={sampling_verification['routed_provider']!r}, "
            f"endpoint_tags={sampling_verification['endpoint_tags']})",
            file=sys.stderr,
        )
    # Filled in below with one entry per parsed candidate, so the log record points straight at
    # the accepted/rejected entries (same call_id) it produced.
    outcomes: list[dict] = []
    call_records.append(
        {
            "call_id": result.call_id,
            "call_number": result.call_number,
            "wave_idx": result.wave_idx,
            "slot_index": result.slot_index,
            "seed_sent": result.derived_seed,
            "context_examples": result.context_examples,
            "n_parsed_candidates": len(result.candidates),
            "outcomes": outcomes,
            "openrouter_response": result.openrouter_meta,
            "sampling_verification": sampling_verification,
        }
    )
    generation_id = result.openrouter_meta.get("id")

    for candidate_text in result.candidates:
        if counters["n_accepted_this_run"] >= args.n:
            rejected_entries.append(
                {
                    "run_id": run_id,
                    "call_id": result.call_id,
                    "generation_id": generation_id,
                    "call_number": result.call_number,
                    "wave_idx": result.wave_idx,
                    "slot_index": result.slot_index,
                    "rejected_text": candidate_text,
                    "rejected_reason": "target_reached",
                    "context_examples": context_texts,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            counters["n_rejected_this_run"] += 1
            outcomes.append({"status": "rejected", "rejected_reason": "target_reached"})
            continue

        candidate_tokens = rouge_tokenize(candidate_text)
        best_score = -1.0
        best_match: dict | None = None
        for i, entry_tokens in enumerate(seed_tokens):
            score = rouge_l_fmeasure(candidate_tokens, entry_tokens)
            if score > best_score:
                best_score = score
                best_match = {"type": "seed", "id": i}
        for i, entry_tokens in enumerate(pool_tokens):
            score = rouge_l_fmeasure(candidate_tokens, entry_tokens)
            if score > best_score:
                best_score = score
                best_match = {"type": "synthetic", "id": accepted_pool[i]["id"], "text": accepted_pool[i]["text"]}

        if best_score > args.rouge_threshold:
            rejected_entries.append(
                {
                    "run_id": run_id,
                    "call_id": result.call_id,
                    "generation_id": generation_id,
                    "call_number": result.call_number,
                    "wave_idx": result.wave_idx,
                    "slot_index": result.slot_index,
                    "rejected_text": candidate_text,
                    "rejected_reason": "rouge_duplicate",
                    "rouge_l_score": best_score,
                    "matched_pool_id": best_match["id"],
                    "matched_pool_type": best_match["type"],
                    "context_examples": context_texts,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            counters["n_rejected_this_run"] += 1
            outcomes.append({"status": "rejected", "rejected_reason": "rouge_duplicate"})
            print(f"  [reject] score={best_score:.3f} vs {best_match['type']} id={best_match['id']}: {candidate_text[:80]!r}")
        else:
            accepted_pool.append(
                {
                    "id": counters["next_id"],
                    "type": "synthetic",
                    "text": candidate_text,
                    "context_examples": context_texts,
                    "run_id": run_id,
                    "call_id": result.call_id,
                    "generation_id": generation_id,
                    "call_number": result.call_number,
                    "wave_idx": result.wave_idx,
                    "slot_index": result.slot_index,
                    "added_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            pool_tokens.append(candidate_tokens)
            outcomes.append({"status": "accepted", "accepted_id": counters["next_id"]})
            counters["next_id"] += 1
            counters["n_accepted_this_run"] += 1
            print(f"  [accept] ({counters['n_accepted_this_run']}/{args.n}) best_score={best_score:.3f}: {candidate_text[:80]!r}")


def summarize_sampling_verification(call_records: list[dict]) -> dict:
    """Run-level roll-up of every call's sampling_verification (calls from before this field existed
    - e.g. checkpointed by an older version of this script - are counted as missing)."""
    verifications = [c.get("sampling_verification") for c in call_records]
    present = [v for v in verifications if v is not None]
    def count(key: str) -> dict:
        counts: dict = {}
        for v in present:
            value = v.get(key)
            value = value if isinstance(value, str) else json.dumps(value)
            counts[value] = counts.get(value, 0) + 1
        return counts
    return {
        "n_calls": len(call_records),
        "n_missing": len(verifications) - len(present),
        "n_unverified": sum(1 for v in present if not v["verified"]),
        "all_verified": len(present) == len(call_records) and all(v["verified"] for v in present),
        "routed_provider_counts": count("routed_provider"),
        "quantization_counts": count("quantization"),
        "requested_temperature_counts": count("requested_temperature"),
        "requested_top_p_counts": count("requested_top_p"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic blackbox examples for one domain via the OpenRouter API, "
        "with ROUGE-L dedup against an accepted-sample pool."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, choices=sorted(MODEL_PRESETS),
        help="Generator model keyword, mapped to an OpenRouter model id + provider via "
        f"MODEL_PRESETS ({', '.join(f'{k}={v[0]}' for k, v in MODEL_PRESETS.items())}; default: %(default)s).",
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Domain subfolder (e.g. 'toxicity_detection', 'cti_vsp', 'claudette_tos'), "
        "either a name under blackbox_generation/ or a full/relative path to it.",
    )
    parser.add_argument(
        "--seed-group", type=str, required=True, choices=sorted(VALID_SEED_GROUPS),
        help="Seed group id (01-05); selects data/seed_groups/<domain>/*_seed_group_<id>.tsv.",
    )
    parser.add_argument(
        "--n", type=int, required=True,
        help="Number of ACCEPTED samples to generate before the script stops.",
    )
    parser.add_argument(
        "--prefix", type=str, required=True,
        help="Filename prefix for output ('<prefix>_accepted.json' / '<prefix>_rejected.json') "
        "and log ('<prefix>_log.json') files.",
    )
    parser.add_argument(
        "--rouge-threshold", type=float, default=0.7,
        help="Reject a candidate if its highest ROUGE-L F-measure against the accepted pool "
        "exceeds this value (default: %(default)s).",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override BASE_MODEL_PARAMS['temperature'].")
    parser.add_argument("--top-p", type=float, default=None, help="Override BASE_MODEL_PARAMS['top_p'].")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override BASE_MODEL_PARAMS['max_tokens'].")
    parser.add_argument(
        "--extra-params", type=str, default=None,
        help="Additional OpenRouter request body parameters as a JSON object string.",
    )
    parser.add_argument(
        "--max-calls", type=int, default=None,
        help="Safety cap on total API calls (default: 50x --n) to avoid looping forever if "
        "candidates keep getting rejected.",
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
        "--max-concurrent-requests", type=int, default=1,
        help="Number of worker threads (default: 1). Requests run in waves of "
        "2 * --max-concurrent-requests prompts: a wave's prompts are all built from the same "
        "frozen pool snapshot and dispatched together, the run blocks on a barrier until the "
        "whole wave completes, then dedup is applied sequentially in slot order before the next "
        "wave's snapshot is taken. Dedup and prompt content are always reproducible for a given "
        "run_id/seed_group/wave/slot, independent of completion order or worker count.",
    )
    parser.add_argument(
        "--requests-per-second", type=float, default=None,
        help="Optional cap on request START rate shared across all concurrent workers, to respect the "
        "API's tps limit (default: unlimited).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run from its checkpoint ('<prefix>_checkpoint.json' in <path>/output) "
        "instead of starting a fresh run_id at wave 0. The checkpoint is written after every wave and "
        "deleted once the run's --n target is reached; if it still exists but --resume is not passed, "
        "the script refuses to start (to avoid silently orphaning the in-progress run) until you either "
        "pass --resume or delete the checkpoint file yourself.",
    )
    return parser


def checkpoint_resolved_args(domain_dir: Path, args: argparse.Namespace, max_calls: int) -> dict:
    """Snapshot of the args that materially affect a checkpointed run's semantics.

    Used to detect a --resume invocation that quietly drifted from the run it's resuming
    (e.g. a different --seed-group, which would corrupt the dedup/context universe).
    """
    return {
        "model": args.model,
        "path": str(domain_dir),
        "seed_group": args.seed_group,
        "n": args.n,
        "rouge_threshold": args.rouge_threshold,
        "max_calls": max_calls,
        "max_concurrent_requests": args.max_concurrent_requests,
        "requests_per_second": args.requests_per_second,
    }


def save_checkpoint(
    path: Path,
    *,
    run_id: str,
    started_at: str,
    wave_idx: int,
    counters: dict,
    call_records: list[dict],
    resolved_args: dict,
) -> None:
    save_json(
        path,
        {
            "run_id": run_id,
            "started_at": started_at,
            "wave_idx": wave_idx,
            "counters": counters,
            "call_records": call_records,
            "resolved_args": resolved_args,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_model(args.model)

    load_dotenv(args.env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(f"OPENROUTER_API_KEY is not set. Set it in the environment or in {args.env_file}.")

    domain_dir = resolve_domain_dir(args.path)
    domain_name = domain_dir.name
    system_prompt, prompt_template = load_prompt_module(domain_dir)
    seed_file = find_seed_file(domain_name, args.seed_group)
    seed_examples = load_seed_examples(seed_file)
    n_seeds = len(seed_examples)
    if n_seeds < MIN_SEED_EXAMPLES:
        raise SystemExit(
            f"Need at least {MIN_SEED_EXAMPLES} seed examples for the context-drawing scheme "
            f"(2 seeds + 1 accepted-pool item, or 3 seeds while the pool is still empty), "
            f"but {seed_file} only has {n_seeds}."
        )

    model_params = dict(BASE_MODEL_PARAMS)
    if args.temperature is not None:
        model_params["temperature"] = args.temperature
    if args.top_p is not None:
        model_params["top_p"] = args.top_p
    if args.max_tokens is not None:
        model_params["max_tokens"] = args.max_tokens
    if args.extra_params:
        model_params.update(json.loads(args.extra_params))
    if PIN_PROVIDER:
        model_params["provider"] = {
            **model_params.get("provider", {}),
            "only": [PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    else:
        model_params["provider"] = {
            **model_params.get("provider", {}),
            "require_parameters": True,
        }
    if any(fragment in MODEL_ID for fragment in FP8_DATA_DENY_MODEL_ID_FRAGMENTS):
        model_params["provider"] = {
            **model_params["provider"],
            "quantizations": ["fp8"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        }

    max_calls = args.max_calls if args.max_calls is not None else args.n * 50

    output_dir = domain_dir / "output"
    log_dir = domain_dir / "log"
    accepted_path = output_dir / f"{args.prefix}_accepted.json"
    rejected_path = output_dir / f"{args.prefix}_rejected.json"
    failed_path = output_dir / f"{args.prefix}_failed.json"
    checkpoint_path = output_dir / f"{args.prefix}_checkpoint.json"
    log_path = log_dir / f"{args.prefix}_log.json"

    resolved_args = checkpoint_resolved_args(domain_dir, args, max_calls)
    checkpoint = load_json_dict(checkpoint_path)

    if args.resume:
        if checkpoint is None:
            raise SystemExit(f"--resume was given but no checkpoint found at {checkpoint_path}; nothing to resume.")
        checkpoint_seed_group = checkpoint["resolved_args"]["seed_group"]
        if checkpoint_seed_group != args.seed_group:
            raise SystemExit(
                f"Checkpoint at {checkpoint_path} was created with --seed-group {checkpoint_seed_group!r}, "
                f"but this invocation passed --seed-group {args.seed_group!r}. Resuming across different "
                "seed groups would corrupt the context/dedup universe - use the original --seed-group, "
                "or delete the checkpoint file to discard it and start fresh."
            )
        # Checkpoints written before --model existed carry no "model" key; those are not checked.
        checkpoint_model = checkpoint["resolved_args"].get("model")
        if checkpoint_model is not None and checkpoint_model != args.model:
            raise SystemExit(
                f"Checkpoint at {checkpoint_path} was created with --model {checkpoint_model!r}, "
                f"but this invocation passed --model {args.model!r}. Resuming with a different generator "
                "model would mix models within one run - use the original --model, or delete the "
                "checkpoint file to discard it and start fresh."
            )
        for key in ("n", "rouge_threshold", "max_calls", "max_concurrent_requests", "requests_per_second"):
            old_value = checkpoint["resolved_args"].get(key)
            new_value = resolved_args[key]
            if old_value != new_value:
                print(f"[resume] Note: --{key.replace('_', '-')} changed since the checkpoint was saved ({old_value!r} -> {new_value!r}).")
        run_id = checkpoint["run_id"]
        started_at = checkpoint["started_at"]
        wave_idx = checkpoint["wave_idx"]
        counters = checkpoint["counters"]
        call_records = checkpoint["call_records"]
        print(f"[resume] Continuing run {run_id} from wave {wave_idx} (checkpoint saved at {checkpoint['saved_at']}).")
    else:
        if checkpoint is not None:
            raise SystemExit(
                f"Found an incomplete checkpoint at {checkpoint_path} (run_id={checkpoint['run_id']}, "
                f"next wave {checkpoint['wave_idx']}, {checkpoint['counters']['n_accepted_this_run']} accepted "
                f"so far). Pass --resume to continue it, or delete the file to discard it and start a fresh run."
            )
        run_id = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        wave_idx = 0
        counters = None  # filled in below, once next_id is known
        call_records = []

    print(f"Domain: {domain_dir} | seed group: {args.seed_group} | seed file: {seed_file} ({n_seeds} seeds)")
    print(f"Run id: {run_id} | model: {MODEL_ID} | target accepted: {args.n} | max calls: {max_calls}")

    # Seed examples and the accepted pool are two distinct dedup/context sources, but the
    # persisted accepted.json still merges both (type=="seed" / type=="synthetic") so it stays
    # a complete record of every prompt-eligible text. On load we split them back apart; if the
    # stored seed entries don't match the current seed file (e.g. first run ever), regenerate
    # them fresh from seed_examples so ids/text always agree with what's actually being drawn.
    loaded_pool = load_json_list(accepted_path)
    stored_seed_entries = [e for e in loaded_pool if e.get("type") == "seed"]
    accepted_pool = [e for e in loaded_pool if e.get("type") == "synthetic"]
    if len(stored_seed_entries) == n_seeds:
        seed_pool_entries = stored_seed_entries
    else:
        seed_pool_entries = [
            {"id": i, "type": "seed", "text": text, "seed_example_id": i, "run_id": None, "added_at": started_at}
            for i, text in enumerate(seed_examples)
        ]
    seed_tokens = [rouge_tokenize(e["text"]) for e in seed_pool_entries]
    pool_tokens = [rouge_tokenize(entry["text"]) for entry in accepted_pool]
    next_id = max((entry["id"] for entry in accepted_pool), default=len(seed_pool_entries) - 1) + 1
    print(
        f"Loaded {len(seed_pool_entries)} seed example(s) and {len(accepted_pool)} accepted "
        f"synthetic entrie(s) from {accepted_path} (ROUGE-L threshold: {args.rouge_threshold})"
    )

    rejected_entries = load_json_list(rejected_path)
    failed_entries = load_json_list(failed_path)

    if counters is None:  # fresh start (not resuming) - checkpoint branch above already set these
        counters = {
            "n_calls": 0,
            "n_failed_calls": 0,
            "n_accepted_this_run": 0,
            "n_rejected_this_run": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "next_id": next_id,
        }
    else:
        # Resuming: next_id must still track the true max id across the pool as loaded from
        # disk (accepted.json was written up to the last checkpoint too), not go stale.
        counters["next_id"] = max(counters["next_id"], next_id)

    ctx = WorkerContext(
        api_key=api_key,
        model_params=model_params,
        system_prompt=system_prompt,
        max_retry_seconds=args.max_retry_seconds,
        request_timeout=args.request_timeout,
        rate_limiter=RateLimiter(args.requests_per_second),
    )
    endpoint_catalog = EndpointCatalog(MODEL_ID, args.request_timeout)
    endpoint_catalog.refresh()

    n_workers = args.max_concurrent_requests
    wave_batch_size = 2 * n_workers
    # wave_idx is already set above: 0 for a fresh run, or the checkpoint's next wave on --resume.

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        while counters["n_accepted_this_run"] < args.n:
            remaining_budget = max_calls - counters["n_calls"]
            if remaining_budget <= 0:
                print(
                    f"[warn] Reached --max-calls ({max_calls}) with only "
                    f"{counters['n_accepted_this_run']}/{args.n} accepted; stopping.",
                    file=sys.stderr,
                )
                break

            slots_this_wave = min(wave_batch_size, remaining_budget)
            pool_snapshot = list(accepted_pool)  # frozen for this wave's prompt construction

            print(
                f"[wave {wave_idx}] dispatching {slots_this_wave} call(s) "
                f"(accepted {counters['n_accepted_this_run']}/{args.n}, pool snapshot size {len(pool_snapshot)})..."
            )

            futures: dict = {}
            for slot_index in range(slots_this_wave):
                counters["n_calls"] += 1
                call_number = counters["n_calls"]
                slot_rng = derive_slot_rng(run_id, args.seed_group, ARM, wave_idx, slot_index)
                context_examples = pick_context_examples(slot_rng, seed_examples, pool_snapshot)
                seed_examples_block = render_seed_examples_block(context_examples)
                user_prompt = prompt_template.replace("{{SEED_EXAMPLES}}", seed_examples_block)
                derived_seed = derive_seed(run_id, call_number)
                call_id = make_call_id(run_id, call_number)

                if call_number == 1:
                    print_raw_first_call(system_prompt, user_prompt, model_params, derived_seed)

                context_desc = ", ".join(
                    f"{c['type']}:{c.get('seed_example_id', c.get('accepted_id'))}" for c in context_examples
                )
                print(f"  [wave {wave_idx} slot {slot_index}] call {call_number} ({call_id}, seed_sent={derived_seed}) context=[{context_desc}]")

                future = executor.submit(
                    fetch_one, ctx, call_id, call_number, wave_idx, slot_index, user_prompt, context_examples, derived_seed
                )
                futures[future] = slot_index

            # Barrier: wait for the whole wave to finish (success, failure, or a raised
            # AssertionError all count as "done" here) before applying any of its results.
            wait(futures.keys(), return_when=ALL_COMPLETED)
            results_by_slot = {futures[future]: future.result() for future in futures}
            for slot_index in range(slots_this_wave):
                process_call_result(
                    results_by_slot[slot_index],
                    run_id=run_id,
                    args=args,
                    seed_tokens=seed_tokens,
                    accepted_pool=accepted_pool,
                    pool_tokens=pool_tokens,
                    rejected_entries=rejected_entries,
                    failed_entries=failed_entries,
                    call_records=call_records,
                    counters=counters,
                    endpoint_catalog=endpoint_catalog,
                )

            wave_idx += 1

            # Checkpoint after every wave: the pool is only ever in a consistent, fully-committed
            # state between waves (never mid-wave), so this is the natural interval to persist at.
            # A crash before the next wave's barrier then loses at most one wave of progress, and
            # --resume can pick up exactly here (same run_id, next wave_idx, same counters).
            save_json(accepted_path, seed_pool_entries + accepted_pool)
            save_json(rejected_path, rejected_entries)
            save_json(failed_path, failed_entries)
            save_checkpoint(
                checkpoint_path,
                run_id=run_id,
                started_at=started_at,
                wave_idx=wave_idx,
                counters=counters,
                call_records=call_records,
                resolved_args=resolved_args,
            )
            print(f"[checkpoint] Saved progress after wave {wave_idx - 1} ({counters['n_accepted_this_run']}/{args.n} accepted so far).")

    n_calls = counters["n_calls"]
    n_failed_calls = counters["n_failed_calls"]
    n_accepted_this_run = counters["n_accepted_this_run"]
    n_rejected_this_run = counters["n_rejected_this_run"]
    total_prompt_tokens = counters["total_prompt_tokens"]
    total_completion_tokens = counters["total_completion_tokens"]
    target_reached = n_accepted_this_run >= args.n

    save_json(accepted_path, seed_pool_entries + accepted_pool)
    print(f"Wrote {len(seed_pool_entries)} seed + {len(accepted_pool)} synthetic entrie(s) to {accepted_path}")

    save_json(rejected_path, rejected_entries)
    print(f"Wrote {len(rejected_entries)} rejected entrie(s) to {rejected_path}")

    save_json(failed_path, failed_entries)
    print(f"Wrote {len(failed_entries)} failed call entrie(s) to {failed_path}")

    if not target_reached:
        # Not done yet (stopped at --max-calls). counters/call_records already reflect the full
        # run_id's progress and are safely checkpointed (every wave writes counters["total_*"] as
        # the RUN's cumulative totals, not a per-invocation delta) - writing a log entry now would
        # double-count those totals into cumulative_prompt_tokens/cumulative_completion_tokens
        # a second time once this run eventually completes. So the log entry is deferred until
        # the run actually finishes; the checkpoint (already up to date from the last wave) stays
        # in place for a later --resume.
        print(
            f"[warn] Stopped without reaching target ({n_accepted_this_run}/{args.n} accepted, {n_calls} call(s) "
            f"made); checkpoint retained at {checkpoint_path}. Run again with --resume (optionally with a "
            "higher --max-calls) to continue.",
            file=sys.stderr,
        )
        return

    finished_at = datetime.now(timezone.utc).isoformat()
    run_entry = {
        "run_id": run_id,
        "model": args.model,
        "model_id": MODEL_ID,
        "path": str(domain_dir),
        "seed_group": args.seed_group,
        "seed_file": str(seed_file),
        "n_seeds": n_seeds,
        "n_requested": args.n,
        "rouge_threshold": args.rouge_threshold,
        "n_calls": n_calls,
        "n_failed_calls": n_failed_calls,
        "n_accepted": n_accepted_this_run,
        "n_rejected": n_rejected_this_run,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "model_params": model_params,
        "sampling_verification_summary": summarize_sampling_verification(call_records),
        "endpoint_catalog": endpoint_catalog.snapshot(),
        "endpoint_catalog_fetched_at": endpoint_catalog.fetched_at,
        "accepted_file": str(accepted_path),
        "rejected_file": str(rejected_path),
        "failed_file": str(failed_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "calls": call_records,
    }

    log_dir.mkdir(parents=True, exist_ok=True)
    existing_log = {}
    if log_path.exists():
        content = log_path.read_text(encoding="utf-8").strip()
        existing_log = json.loads(content) if content else {}

    cumulative_prompt_tokens = existing_log.get("cumulative_prompt_tokens", 0) + total_prompt_tokens
    cumulative_completion_tokens = existing_log.get("cumulative_completion_tokens", 0) + total_completion_tokens
    runs = existing_log.get("runs", [])
    runs.append(run_entry)

    log_entry = {
        "prefix": args.prefix,
        "path": str(domain_dir),
        "cumulative_prompt_tokens": cumulative_prompt_tokens,
        "cumulative_completion_tokens": cumulative_completion_tokens,
        "cumulative_total_tokens": cumulative_prompt_tokens + cumulative_completion_tokens,
        "runs": runs,
    }
    save_json(log_path, log_entry)
    print(f"Wrote run log to {log_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Target reached; removed checkpoint {checkpoint_path}")

    print(
        f"Done: {n_accepted_this_run} accepted / {n_rejected_this_run} rejected this run "
        f"over {n_calls} call(s) ({n_failed_calls} failed); "
        f"prompt_tokens={total_prompt_tokens}, completion_tokens={total_completion_tokens}"
    )


if __name__ == "__main__":
    main()
