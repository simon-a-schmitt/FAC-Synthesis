"""OpenRouter prediction backend for the run_*_benchmark.py scripts (--mode api).

Reuses data_synthesis/shared/openrouter.py for everything request-related: the model presets, the
request parameters (build_model_params: deepseek-v4-flash reasoning off, fp8 + data_collection=deny
for llama/deepseek, provider.require_parameters), the retrying chat call and the rate limiter.

Sampling is fixed to temperature 0. Provider fallbacks are allowed (provider.allow_fallbacks=True):
OpenRouter may route every call to any provider that satisfies the request's parameters; the
provider that actually served a call is logged per example under "openrouter".

The API path free-generates the answer and parses it from the text. For the slot benchmarks
(claudette, toxicity) the continuous AUPRC score is read from the returned token logprobs instead
of a forward pass: logprobs/top_logprobs are requested (with require_parameters only providers that
support them are routed to), the generated token carrying a slot's value is located by character
offset, and the candidates' log-probs are taken from that token's top_logprobs (see slot_logprobs).
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_SYNTHESIS_DIR = ROOT_DIR / "data_synthesis"
sys.path.insert(0, str(DATA_SYNTHESIS_DIR))

from shared.openrouter import (  # noqa: E402
    GENERATION_MODEL_PRESETS,
    RateLimiter,
    build_model_params,
    call_openrouter_chat,
    require_api_key,
    response_meta,
)

# --api-model keyword -> (OpenRouter model id, provider). llama/deepseek are the generator presets
# of data_synthesis, gpt is the labeling model of data_synthesis/labeling/run_labeling.py.
API_MODEL_PRESETS = {
    **GENERATION_MODEL_PRESETS,
    "gpt": ("openai/gpt-4o-mini-2024-07-18", "openai"),
}

API_BASE_PARAMS = {
    "temperature": 0.0,
    "usage": {"include": True},
}

# OpenAI's maximum; the other providers accept at least as many.
API_TOP_LOGPROBS = 20

DEFAULT_ENV_FILE = ROOT_DIR / ".env"


def add_api_cli_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("OpenRouter API (--mode api)")
    group.add_argument("--api-model", choices=sorted(API_MODEL_PRESETS), default=None,
                       help="Model for --mode api: " + ", ".join(f"{k}={v[0]}" for k, v in API_MODEL_PRESETS.items()))
    group.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE,
                       help="Path to a .env file providing OPENROUTER_API_KEY (default: %(default)s).")
    group.add_argument("--request-timeout", type=float, default=120.0)
    group.add_argument("--max-retry-seconds", type=float, default=600.0,
                       help="Wall-clock budget to keep retrying one failed request before the run aborts "
                            "(rerun with --resume to continue).")
    group.add_argument("--max-concurrent-requests", type=int, default=8,
                       help="Parallel requests within one --batch-size chunk (default: %(default)s).")
    group.add_argument("--requests-per-second", type=float, default=None,
                       help="Optional cap on the request start rate (default: unlimited).")


def validate_api_args(args) -> None:
    """--mode api needs --api-model; the local-model flags must not silently do nothing."""
    if args.mode == "api":
        if not args.api_model:
            raise SystemExit("--api-model is required when --mode api")
        if args.max_input_tokens:
            raise SystemExit("--max-input-tokens is not supported with --mode api (prompts are never truncated)")
    elif not args.model_path:
        raise SystemExit(f"--model-path is required when --mode {args.mode}")


class ApiPredictor:
    def __init__(self, args, *, max_tokens: int, logprobs: bool):
        self.model_id, provider = API_MODEL_PRESETS[args.api_model]
        self.api_model = args.api_model
        base = dict(API_BASE_PARAMS, max_tokens=max_tokens)
        if logprobs:
            base.update(logprobs=True, top_logprobs=API_TOP_LOGPROBS)
        # No CLI overrides: temperature stays 0.
        overrides = argparse.Namespace(temperature=None, top_p=None, max_tokens=None, extra_params=None)
        self.model_params = build_model_params(self.model_id, provider, base, overrides)
        self.model_params["provider"]["allow_fallbacks"] = True
        self._api_key = require_api_key(args.env_file)
        self._timeout = args.request_timeout
        self._max_retry_seconds = args.max_retry_seconds
        self._workers = max(1, args.max_concurrent_requests)
        self._rate_limiter = RateLimiter(args.requests_per_second)
        print(f"API model: {self.api_model} -> {self.model_id}")
        print(f"API request params: {json.dumps(self.model_params)}")

    def describe(self) -> dict:
        """Run-level record for the summary JSON."""
        return {"api_model": self.api_model, "model_id": self.model_id, "model_params": self.model_params}

    def _predict_one(self, messages: list[dict]) -> dict:
        response = call_openrouter_chat(
            self._api_key, self.model_id, messages, self.model_params,
            self._max_retry_seconds, self._timeout, self._rate_limiter,
        )
        choice = response["choices"][0]
        return {
            "text": (choice.get("message") or {}).get("content") or "",
            "logprobs": (choice.get("logprobs") or {}).get("content"),
            "openrouter": response_meta(response),
        }

    def predict_many(self, batch_messages: list[list[dict]]) -> list[dict]:
        """One completion per message list, in input order. A call that still fails after the retry
        budget raises, aborting the run before this chunk is written (rerun with --resume)."""
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            return list(pool.map(self._predict_one, batch_messages))


def _value_token_index(tokens: list[str], text: str, char_pos: int, candidates: list[str], norm) -> int | None:
    """Index of the generated token holding the value that starts at text[char_pos].

    Primarily by character offset. Some providers return lossy token strings (e.g. "L" for the
    token "LAW"), so the tokens do not reassemble the text; then the value is matched by order:
    the k-th standalone candidate occurrence in the text <-> the k-th token that is a candidate,
    provided both counts agree.
    """
    joined = "".join(tokens)
    shift = joined.find(text)
    if shift >= 0:
        offset = 0
        for i, tok in enumerate(tokens):
            if offset <= char_pos + shift < offset + len(tok):
                return i
            offset += len(tok)
        return None
    alternation = "|".join(re.escape(c) for c in candidates)
    flags = re.IGNORECASE if norm("A") == "a" else 0
    starts = [m.start() for m in re.finditer(rf"(?<![A-Za-z])(?:{alternation})(?![A-Za-z])", text, flags)]
    wanted = {norm(c) for c in candidates}
    value_tokens = [i for i, tok in enumerate(tokens) if norm(tok) in wanted]
    if char_pos not in starts or len(starts) != len(value_tokens):
        return None
    return value_tokens[starts.index(char_pos)]


def slot_logprobs(logprobs_content: list[dict] | None, text: str, char_pos: int,
                  candidates: list[str], *, ignore_case: bool = False) -> dict[str, float] | None:
    """{candidate: log p} at the generated token holding the slot value that starts at text[char_pos].

    Returns None if there are no logprobs, the token cannot be located (see _value_token_index),
    or it is not exactly one candidate (modulo surrounding whitespace). A candidate missing from
    that token's top_logprobs gets the smallest listed log-prob (an upper bound on its true value).
    """
    if not logprobs_content:
        return None

    def norm(s: str) -> str:
        s = s.strip()
        return s.lower() if ignore_case else s

    tokens = [t.get("token") or "" for t in logprobs_content]
    i = _value_token_index(tokens, text, char_pos, candidates, norm)
    wanted = {norm(c): c for c in candidates}
    if i is None or norm(tokens[i]) not in wanted:
        return None
    entry = logprobs_content[i]
    alternatives = [(entry.get("token") or "", entry.get("logprob"))]
    alternatives += [(a.get("token") or "", a.get("logprob")) for a in entry.get("top_logprobs") or []]
    alternatives = [(t, lp) for t, lp in alternatives if lp is not None]
    if not alternatives:
        return None
    floor = min(lp for _, lp in alternatives)
    out: dict[str, float] = {}
    for t, lp in alternatives:
        cand = wanted.get(norm(t))
        if cand is not None and lp > out.get(cand, float("-inf")):
            out[cand] = float(lp)
    for c in candidates:
        out.setdefault(c, float(floor))
    return out


def first_messages_dump(messages: list[dict]) -> None:
    bar = "=" * 88
    print(bar)
    print("RAW API INPUT -- messages of the first example (printed once)")
    print(bar)
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print(bar, flush=True)


def check_resume_api_model(results: list[dict], api_model: str) -> None:
    """Refuse to resume an output JSONL written by another model (or by a local-model run)."""
    others = {r.get("api_model") for r in results} - {api_model}
    if others:
        raise SystemExit(
            f"--resume: existing output contains results from {sorted(map(str, others))}, "
            f"not from --api-model {api_model}. Use a different --output-jsonl."
        )
