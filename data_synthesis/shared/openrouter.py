"""OpenRouter plumbing shared by every generation arm and by labeling: model presets, request
parameters, the retrying chat call, rate limiting and the per-call sampling verification."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from shared.benchmarks import DATA_SYNTHESIS_DIR
from shared.run_io import utc_now

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_ENDPOINTS_URL_TEMPLATE = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"
DEFAULT_ENV_FILE = DATA_SYNTHESIS_DIR / ".env"

# Generator models of all three arms: --model keyword -> (OpenRouter model id, provider slug).
GENERATION_MODEL_PRESETS = {
    "llama": ("meta-llama/llama-3.1-8b-instruct", "deepinfra/fp8"),
    "deepseek": ("deepseek/deepseek-v4-flash-0731", "baseten/fp8"),
}

# If True, every request is pinned to the preset's provider (provider.only). If False, OpenRouter
# load-balances across every provider that satisfies provider.require_parameters.
PIN_PROVIDER = False

# The deepseek-v4-flash family is the only model the "reasoning": {"enabled": False} override is
# known to be needed/supported for.
DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT = "deepseek-v4-flash"
# For these models every request is additionally pinned to fp8 and opts out of prompt data collection.
FP8_DATA_DENY_MODEL_ID_FRAGMENTS = (DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT, "llama-3.1-8b-instruct")

MAX_RETRY_BACKOFF_SECONDS = 60.0


def load_dotenv(env_file: Path) -> None:
    if not env_file.exists():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_api_key(env_file: Path) -> str:
    load_dotenv(env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(f"OPENROUTER_API_KEY is not set. Set it in the environment or in {env_file}.")
    return api_key


def build_model_params(model_id: str, provider: str, base_params: dict, args: argparse.Namespace) -> dict:
    """Request body parameters: base_params, the model-specific settings, the CLI overrides
    (--temperature/--top-p/--max-tokens/--extra-params) and the provider routing preferences."""
    params = dict(base_params)
    if DEEPSEEK_V4_FLASH_MODEL_ID_FRAGMENT in model_id:
        params["reasoning"] = {"enabled": False}
    for key in ("temperature", "top_p", "max_tokens"):
        if getattr(args, key, None) is not None:
            params[key] = getattr(args, key)
    if args.extra_params:
        params.update(json.loads(args.extra_params))
    routing = {"require_parameters": True}
    if PIN_PROVIDER:
        routing.update(only=[provider], allow_fallbacks=True)
    if any(fragment in model_id for fragment in FP8_DATA_DENY_MODEL_ID_FRAGMENTS):
        routing.update(quantizations=["fp8"], allow_fallbacks=True, data_collection="deny")
    params["provider"] = {**params.get("provider", {}), **routing}
    return params


def expected_provider_name(provider: str) -> str | None:
    """The provider display name OpenRouter echoes back (e.g. "DeepInfra" for "deepinfra/fp8") -
    only checked when PIN_PROVIDER is set."""
    return provider.split("/")[0] if PIN_PROVIDER else None


def add_api_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("OpenRouter API")
    group.add_argument("--temperature", type=float, default=None, help="Override the base temperature.")
    group.add_argument("--top-p", type=float, default=None, help="Override the base top_p.")
    group.add_argument("--max-tokens", type=int, default=None, help="Override the base max_tokens.")
    group.add_argument("--extra-params", type=str, default=None,
                       help="Additional OpenRouter request body parameters as a JSON object string.")
    group.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE,
                       help="Path to a .env file providing OPENROUTER_API_KEY.")
    group.add_argument("--request-timeout", type=float, default=120.0)
    group.add_argument(
        "--max-retry-seconds", type=float, default=600.0,
        help="Wall-clock budget to keep retrying one failed HTTP request (exponential backoff, capped at "
        f"{MAX_RETRY_BACKOFF_SECONDS:.0f}s per wait) before logging the call as failed (default: %(default)s).",
    )
    group.add_argument("--max-concurrent-requests", type=int, default=1, help="Worker threads (default: 1).")
    group.add_argument("--requests-per-second", type=float, default=None,
                       help="Optional cap on the request START rate across all workers (default: unlimited).")


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


# Besides HTTPError/URLError, a timeout or dropped connection can surface as socket.timeout /
# TimeoutError, ConnectionError or http.client.HTTPException - none of them URLError subclasses.
_RETRYABLE_ERRORS = (
    urllib.error.HTTPError,
    urllib.error.URLError,
    socket.timeout,
    TimeoutError,
    ConnectionError,
    http.client.HTTPException,
)


def call_openrouter_chat(
    api_key: str,
    model: str,
    messages: list[dict],
    model_params: dict,
    max_retry_seconds: float,
    timeout: float,
    rate_limiter: RateLimiter | None = None,
) -> dict:
    """POSTs one chat completion. Retries span up to max_retry_seconds wall-clock time (providers
    return transient 429s under overload that can take minutes to clear), with exponential backoff
    capped at MAX_RETRY_BACKOFF_SECONDS. Raises RuntimeError once the budget is used up."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = json.dumps({"model": model, "messages": messages, **model_params}).encode("utf-8")
    deadline = time.monotonic() + max_retry_seconds
    last_error = None
    attempt = 0
    while True:
        attempt += 1
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            request = urllib.request.Request(OPENROUTER_CHAT_URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except _RETRYABLE_ERRORS as exc:
            detail = exc.read().decode("utf-8", "replace") if isinstance(exc, urllib.error.HTTPError) else str(exc)
            last_error = f"{exc} - {detail}"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait = min(2 ** attempt, MAX_RETRY_BACKOFF_SECONDS, remaining)
            print(
                f"  [warn] OpenRouter chat request failed (attempt {attempt}, {remaining:.0f}s left in retry "
                f"budget): {last_error}. Retrying in {wait:.0f}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"OpenRouter chat request failed after {attempt} attempts over {max_retry_seconds:.0f}s: {last_error}")


def response_meta(response: dict) -> dict:
    """The parts of an OpenRouter chat response every log record keeps."""
    choice = response["choices"][0]
    return {
        "id": response.get("id"),
        "model": response.get("model"),
        "provider": response.get("provider"),
        "usage": response.get("usage", {}),
        "finish_reason": choice.get("finish_reason"),
    }


class EndpointCatalog:
    """Cached view of OpenRouter's public endpoint list for a model (provider -> quantization etc.).

    A chat response only names the provider a call was routed to - never the quantization or the
    sampling params actually applied. The quantization is therefore resolved here by matching the
    routed provider against the endpoints that satisfy the request's provider.quantizations filter.
    Main-thread-only; re-fetched once on an unknown provider.
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
            self.fetched_at = utc_now()
        except (*_RETRYABLE_ERRORS, KeyError, json.JSONDecodeError) as exc:
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


def build_sampling_verification(requested_params: dict, response_provider: str | None, catalog: EndpointCatalog) -> dict:
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


def verify_call(call_id: str, requested_params: dict, response_provider: str | None, catalog: EndpointCatalog) -> dict:
    """build_sampling_verification, with a warning for calls that could not be verified."""
    verification = build_sampling_verification(requested_params, response_provider, catalog)
    if not verification["verified"]:
        print(
            f"  [warn] Call {call_id} could not be verified against the endpoint catalog "
            f"(provider={verification['routed_provider']!r}, endpoint_tags={verification['endpoint_tags']})",
            file=sys.stderr,
        )
    return verification


def summarize_sampling_verification(call_records: list[dict]) -> dict:
    """Run-level roll-up of every call's sampling_verification."""
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
