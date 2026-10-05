"""Bookkeeping of prompt truncation, shared by the run_*_benchmark.py scripts. It never changes
what is fed to the model, but truncation must never happen silently: a run with a truncated
prompt fails at the end (require_no_truncation), after its summary has been written.

generate_batch() in every benchmark script left-truncates the encoded chat text to its last
max_input_tokens tokens (ids[-max_input_tokens:]) only if --max-input-tokens is given (kept for
manual experiments; the experiment config does not set it). A too-long prompt then loses its
beginning first: BOS, system prompt and - in --mode icl - the leading few-shot examples. The
CUDA/OOM fallbacks of the scripts never truncate, they only split batches.

Every per-example result records prompt_tokens (length of the rendered chat text, encoded the
same way generate_batch() encodes it), the limit in effect and whether it was truncated;
truncation_summary() aggregates that over all results, including resumed ones.
"""

from __future__ import annotations


def prompt_token_counts(tokenizer, texts: list[str]) -> list[int]:
    # Same encoding as generate_batch(): the chat template already contains the special tokens.
    return [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]


def truncation_fields(n_tokens: int, max_input_tokens: int | None) -> dict:
    return {
        "prompt_tokens": n_tokens,
        "max_input_tokens": max_input_tokens,
        "truncated": bool(max_input_tokens) and n_tokens > max_input_tokens,
    }


def truncation_summary(results: list[dict], max_input_tokens: int | None) -> dict:
    """n_truncated over all results; results from older runs without the fields (resumed output)
    are counted as n_truncation_unknown."""
    known = [r for r in results if "truncated" in r]
    return {
        "max_input_tokens": max_input_tokens,
        "n_truncated": sum(1 for r in known if r["truncated"]),
        "n_truncation_unknown": len(results) - len(known),
        "max_prompt_tokens": max((r["prompt_tokens"] for r in known), default=None),
    }


def require_no_truncation(t: dict) -> None:
    """Exit != 0 if any prompt was truncated - or may have been: results resumed from an output
    written without token counts (older script version, whose CUDA fallback could truncate)."""
    problems = []
    if t["n_truncated"]:
        problems.append(f"{t['n_truncated']} prompt(s) truncated to max_input_tokens={t['max_input_tokens']} "
                        f"(longest prompt: {t['max_prompt_tokens']} tokens)")
    if t["n_truncation_unknown"]:
        problems.append(f"{t['n_truncation_unknown']} resumed result(s) without token counts - "
                        "rerun without --resume (move the output JSONL away)")
    if problems:
        raise SystemExit("ERROR: prompt truncation: " + "; ".join(problems) + ".")


def format_truncation_summary(t: dict) -> str:
    line = (f"  truncated_prompts: {t['n_truncated']}  (max_input_tokens={t['max_input_tokens']}, "
            f"max_prompt_tokens={t['max_prompt_tokens']}")
    if t["n_truncation_unknown"]:
        line += f", unknown (resumed without token counts): {t['n_truncation_unknown']}"
    return line + ")"
