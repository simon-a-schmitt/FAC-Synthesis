#!/usr/bin/env python3
"""Benchmark-agnostic constrained slot scoring.

One primitive, three benchmarks
-------------------------------
An answer is modelled as an ordered sequence of SLOTS. Each slot is a fixed
literal text FRAGMENT followed by exactly one value drawn from a small
CANDIDATE alphabet:

  CLAUDETTE-ToS  8 slots, binary     "LTD: Y|TER: N|...|ARB: N"
  CTI-VSP        8 slots, k-way      "CVSS:3.1/AV: N/AC: L/.../A: H"
  Toxicity       1 slot,  binary     "Answer: safe"

The model never free-generates the value. Per slot we push the fragment tokens
through the model, stop at the position immediately after them, read the
full-vocabulary log-softmax once, and keep only the entries belonging to that
slot's candidates. The argmax gives the hard label (for F1); the differences
between candidate log-probs give the continuous score (for AUPRC). The chosen
token is fed back into the KV-cache, so slot k+1 is conditioned on the answer
realized at slot k.

Why the plan is built, not written
----------------------------------
Llama-3's BPE merges a leading punctuation character into the following word
where the vocabulary happens to contain the merged form. ":N" is a single
token; ":Y" is not. Writing scaffold fragments by hand and tokenizing them in
isolation therefore produces a token sequence that differs from the canonical
tokenization of the full answer string, and the model gets scored at a position
that never occurs in natural text. `build_slot_plan()` instead derives every
split point from `encode(prefix + fragment + candidate)` and refuses to build a
plan it cannot score exactly. Putting a space before each value ("LTD: Y", not
"LTD:Y") makes the split clean; the builder verifies this rather than assuming
it.

Caller responsibilities
-----------------------
Chat-template rendering, tokenization, left-padding and any OOM batch-halving
stay with the benchmark script. This module owns the answer format, the
tokenization contract and the scoring loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


# ==========================================================================
# Format construction -- single source of truth per benchmark
# ==========================================================================

def make_fragments(
    names: Sequence[str],
    *,
    sep: str = "|",
    assign: str = ": ",
    prefix: str = "",
) -> list[str]:
    """Build the per-slot literal fragments of an answer template.

    The trailing space in the default `assign` is what makes each candidate a
    clean single token (' Y' / ' N' rather than a ':N' merge). Change it only if
    `build_slot_plan()` still accepts the result.

      make_fragments(["LTD", "TER"])
        -> ["LTD: ", "|TER: "]                       # "LTD: Y|TER: N"
      make_fragments(CVSS, sep="/", prefix="CVSS:3.1/")
        -> ["CVSS:3.1/AV: ", "/AC: ", ...]           # "CVSS:3.1/AV: N/AC: L/..."
    """
    return [f"{prefix if i == 0 else sep}{name}{assign}" for i, name in enumerate(names)]


def format_example(fragments: Sequence[str], placeholder: str = "?") -> str:
    """Render the template with placeholders, for use in the system prompt.

    Always derive the prompt's format example from the same fragments the plan
    is built from, so prompt and scoring cannot drift apart.
    """
    return "".join(f"{frag}{placeholder}" for frag in fragments)


def answer_string(fragments: Sequence[str], values: Sequence[str]) -> str:
    """Render a concrete answer, e.g. for fine-tuning targets or assertions."""
    if len(fragments) != len(values):
        raise ValueError("fragments and values must have equal length")
    return "".join(f"{f}{v}" for f, v in zip(fragments, values))


# ==========================================================================
# Plan
# ==========================================================================

@dataclass(frozen=True)
class SlotSpec:
    name: str
    fragment: str
    candidates: tuple[str, ...]
    fragment_ids: tuple[int, ...]   # forced into the cache before the read
    candidate_ids: tuple[int, ...]  # in-context single token per candidate


@dataclass(frozen=True)
class SlotPlan:
    slots: tuple[SlotSpec, ...]
    tie_policy: str

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.slots]

    @property
    def fragments(self) -> list[str]:
        return [s.fragment for s in self.slots]

    def answer_string(self, values: Sequence[str]) -> str:
        return answer_string(self.fragments, values)

    def describe(self, tokenizer) -> str:
        lines = [f"slot plan ({len(self.slots)} slots, tie policy: {self.tie_policy})"]
        for s in self.slots:
            frag = tokenizer.convert_ids_to_tokens(list(s.fragment_ids))
            cand = tokenizer.convert_ids_to_tokens(list(s.candidate_ids))
            lines.append(f"  {s.name:<5} forced={frag} -> {dict(zip(s.candidates, cand))}")
        return "\n".join(lines)


class SlotPlanError(ValueError):
    """The answer template cannot be scored with one logit read per slot."""


def _lcp(seqs: list[list[int]]) -> list[int]:
    out: list[int] = []
    for parts in zip(*seqs):
        if len(set(parts)) != 1:
            break
        out.append(parts[0])
    return out


def build_slot_plan(
    tokenizer,
    names: Sequence[str],
    fragments: Sequence[str],
    candidates_per_slot: Sequence[Sequence[str]],
    *,
    leads: Sequence[str],
    tie_policy: str = "last",
) -> SlotPlan:
    """Resolve fragment and candidate token ids from canonical tokenization.

    `leads` are one or more rendered prompt strings up to the point where the
    answer begins (chat template including the generation prompt). Pass at least
    two, with a short and a long user message, so the builder can confirm the
    plan does not depend on what precedes the answer. In practice it cannot,
    because the assistant header ends in special tokens that no BPE merge
    crosses, but the check is free.

    `tie_policy` decides which candidate wins when two log-probs are exactly
    equal -- bf16 quantizes coarsely enough that this happens at the decision
    boundary. "last" picks the last entry of `candidates`; order each alphabet
    so the majority class sits last (["Y", "N"], ["toxic", "safe"]) and ties
    will not systematically inflate the minority class. "first" is plain argmax
    and exists only for comparison.

    Raises SlotPlanError if a slot cannot be scored with a single logit read.
    Do not work around that in the caller -- change the template (a space before
    each value normally resolves it) and mirror the change in the fine-tuning
    targets.
    """
    if not (len(names) == len(fragments) == len(candidates_per_slot)):
        raise ValueError("names, fragments and candidates_per_slot must align")
    if not leads:
        raise ValueError("at least one lead string is required")
    if tie_policy not in ("first", "last"):
        raise ValueError("tie_policy must be 'first' or 'last'")

    def enc(s: str) -> list[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    specs: list[SlotSpec] = []

    for k, (name, frag, cands) in enumerate(zip(names, fragments, candidates_per_slot)):
        if len(cands) < 2:
            raise ValueError(f"slot {name}: need at least two candidates")

        # Probe every lead x every possible value of the PREVIOUS slot. The
        # previous value is the only realistic source of a context-dependent
        # merge, and bounding the probe there keeps this linear in slot count.
        probes: list[str] = []
        for lead in leads:
            if k == 0:
                probes.append(lead)
            else:
                head = lead + "".join(
                    f + c[0]
                    for f, c in zip(fragments[: k - 1], candidates_per_slot[: k - 1])
                )
                probes.extend(head + fragments[k - 1] + v for v in candidates_per_slot[k - 1])

        resolved = None
        for prefix in probes:
            prefix_ids = enc(prefix)
            variants = [enc(prefix + frag + c) for c in cands]
            common = _lcp(variants)

            if common[: len(prefix_ids)] != prefix_ids:
                raise SlotPlanError(
                    f"slot {name} ({frag!r}): appending the fragment retokenizes "
                    f"already-committed context. Insert a separator before the fragment."
                )

            fragment_ids = tuple(common[len(prefix_ids):])
            tails = [v[len(common):] for v in variants]

            if any(len(t) != 1 for t in tails):
                detail = {c: tokenizer.convert_ids_to_tokens(t) for c, t in zip(cands, tails)}
                raise SlotPlanError(
                    f"slot {name} ({frag!r}): candidates are not single tokens in "
                    f"context: {detail}. Add a space before the value (e.g. "
                    f"{frag.rstrip() + ' ' + cands[0]!r}) and apply the same change "
                    f"to the fine-tuning targets."
                )

            candidate_ids = tuple(t[0] for t in tails)
            if len(set(candidate_ids)) != len(candidate_ids):
                raise SlotPlanError(
                    f"slot {name}: two candidates map to the same token id and would "
                    f"be indistinguishable at this position."
                )

            sig = (fragment_ids, candidate_ids)
            if resolved is None:
                resolved = sig
            elif resolved != sig:
                raise SlotPlanError(
                    f"slot {name} ({frag!r}): tokenization depends on the preceding "
                    f"context, so a static plan is not valid. Insert a separator "
                    f"between the previous value and this fragment."
                )

        specs.append(
            SlotSpec(
                name=name,
                fragment=frag,
                candidates=tuple(cands),
                fragment_ids=resolved[0],
                candidate_ids=resolved[1],
            )
        )

    plan = SlotPlan(slots=tuple(specs), tie_policy=tie_policy)

    # Replay: forcing the plan must reproduce the canonical tokenization of a
    # concrete answer. This is the assertion that would have caught the original
    # bug. Two value patterns, because a merge can affect only some values.
    for lead in leads:
        for pick in (0, -1):
            values = [s.candidates[pick] for s in plan.slots]
            replay = list(enc(lead))
            for s in plan.slots:
                replay += list(s.fragment_ids) + [s.candidate_ids[pick]]
            if replay != enc(lead + plan.answer_string(values)):
                raise SlotPlanError(
                    f"plan replay does not match the canonical tokenization of "
                    f"{plan.answer_string(values)!r}. Do not proceed."
                )

    return plan


def assert_target_matches_plan(plan: SlotPlan, target: str, values: Sequence[str]) -> None:
    """Fail loudly if a fine-tuning target does not match the scoring template.

    Call this once at startup with one real training example. Training and
    inference formats drifting apart is silent otherwise, and would invalidate
    every number the benchmark produces.
    """
    expected = plan.answer_string(values)
    if target.strip() != expected:
        raise SlotPlanError(
            f"fine-tuning target and scoring template differ:\n"
            f"  target   : {target.strip()!r}\n"
            f"  template : {expected!r}"
        )


# ==========================================================================
# Scoring
# ==========================================================================

@dataclass
class SlotResult:
    """Per-example scoring result. Keys are slot names throughout."""
    values: dict[str, str]                     # chosen candidate -> F1
    log_probs: dict[str, dict[str, float]]     # log p(candidate), full-vocab softmax
    probs: dict[str, dict[str, float]]         # renormalized over candidates only
    ties: dict[str, bool]                      # exact tie at the decision boundary

    def log_odds(self, slot: str) -> float:
        """log p(first candidate) - log p(second candidate). Binary slots only."""
        lp = self.log_probs[slot]
        if len(lp) != 2:
            raise ValueError(f"slot {slot} is not binary; use probs[{slot!r}] instead")
        a, b = lp.values()
        return a - b


@torch.no_grad()
def score_slots(
    hf_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    plan: SlotPlan,
    *,
    device: str,
) -> list[SlotResult]:
    """Constrained slot scoring over a left-padded, chat-templated prompt batch.

    Per slot: force the fragment tokens through the model (extending the
    KV-cache; the first call also carries the whole prompt), read
    logits[:, -1, :] -- the position fixed a priori by construction, never
    located post-hoc in decoded text -- take the full-vocabulary log-softmax,
    restrict to the slot's candidate ids, pick a winner under the plan's tie
    policy, and feed that token back so the next slot sees the realized answer.

    Every forward call after the first processes only a handful of new tokens,
    so peak logits memory is (batch, few, vocab) rather than
    (batch, full_seq, vocab).
    """
    batch_size = input_ids.shape[0]
    running_mask = attention_mask
    past_key_values = None
    prev_token = None

    acc = [{"values": {}, "log_probs": {}, "probs": {}, "ties": {}} for _ in range(batch_size)]

    for spec in plan.slots:
        frag = torch.tensor(spec.fragment_ids, device=device, dtype=torch.long)
        frag = frag.unsqueeze(0).expand(batch_size, -1)
        step = (torch.cat([input_ids, frag], dim=1) if prev_token is None
                else torch.cat([prev_token, frag], dim=1))

        # Append a mask entry only for tokens not already covered by running_mask:
        # on the first slot the prompt is already masked, so just the fragment;
        # on later slots the whole step (fed-back winner + fragment) is new.
        n_new = frag.shape[1] if prev_token is None else step.shape[1]
        running_mask = torch.cat(
            [running_mask,
             torch.ones(batch_size, n_new, dtype=running_mask.dtype, device=device)],
            dim=1,
        )
        past_len = 0 if past_key_values is None else past_key_values.get_seq_length()
        assert running_mask.shape[1] == past_len + step.shape[1], (
            f"mask {running_mask.shape[1]} != kv {past_len + step.shape[1]}"
        )
        # Left-padding-aware position ids, matching HF generate().
        pos = running_mask.long().cumsum(-1) - 1
        pos = pos.masked_fill(running_mask == 0, 0)[:, -step.shape[1]:]

        out = hf_model(
            input_ids=step,
            attention_mask=running_mask,
            position_ids=pos,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = out.past_key_values

        log_probs = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        cand = torch.tensor(spec.candidate_ids, device=device, dtype=torch.long)
        cl = log_probs[:, cand]                                    # (batch, n_cand)

        if plan.tie_policy == "last":
            flipped = torch.flip(cl, dims=[-1])
            best = cl.shape[-1] - 1 - flipped.argmax(dim=-1)
        else:
            best = cl.argmax(dim=-1)
        is_tie = (cl == cl.max(dim=-1, keepdim=True).values).sum(-1) > 1
        renorm = torch.softmax(cl, dim=-1)

        for b in range(batch_size):
            acc[b]["values"][spec.name] = spec.candidates[int(best[b])]
            acc[b]["log_probs"][spec.name] = {
                c: float(cl[b, i]) for i, c in enumerate(spec.candidates)
            }
            acc[b]["probs"][spec.name] = {
                c: float(renorm[b, i]) for i, c in enumerate(spec.candidates)
            }
            acc[b]["ties"][spec.name] = bool(is_tie[b])

        prev_token = cand[best].unsqueeze(1)

    return [SlotResult(**a) for a in acc]


# ==========================================================================
# Self-check:  python slot_scoring.py /path/to/Llama-3.1-8B-Instruct
# ==========================================================================

if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)

    def leads(system_prompt: str = "test") -> list[str]:
        return [
            tok.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True,
            )
            for u in ("short.",
                      "a considerably longer user message, with punctuation: commas, "
                      "colons and a trailing period.")
        ]

    CLAUDETTE = ["LTD", "TER", "CH", "CR", "USE", "LAW", "J", "ARB"]
    CVSS = [("AV", ["N", "A", "L", "P"]), ("AC", ["L", "H"]), ("PR", ["N", "L", "H"]),
            ("UI", ["N", "R"]), ("S", ["U", "C"]), ("C", ["H", "L", "N"]),
            ("I", ["H", "L", "N"]), ("A", ["H", "L", "N"])]

    configs = [
        ("CLAUDETTE", CLAUDETTE, make_fragments(CLAUDETTE), [["Y", "N"]] * 8),
        ("CTI-VSP", [m for m, _ in CVSS],
         make_fragments([m for m, _ in CVSS], sep="/", prefix="CVSS:3.1/"),
         [v for _, v in CVSS]),
        ("Toxicity", ["label"], ["Answer: "], [["toxic", "safe"]]),
    ]

    failed = False
    for label, names, frags, cands in configs:
        print("=" * 72)
        print(f"{label}   template: {format_example(frags)}")
        print("=" * 72)
        try:
            plan = build_slot_plan(tok, names, frags, cands, leads=leads())
            print(plan.describe(tok))
            print(f"  example answer: {plan.answer_string([c[0] for c in cands])!r}")
        except SlotPlanError as e:
            failed = True
            print(f"  REJECTED: {e}")
        print()
    raise SystemExit(1 if failed else 0)