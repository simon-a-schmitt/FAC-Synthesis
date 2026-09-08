#!/usr/bin/env python3
"""Benchmark-agnostic constrained slot-decoding scoring core.

`score_slots()` implements one reusable primitive: prompt + an ordered list of
fixed scaffold text fragments + a candidate answer alphabet per slot -> log
p(candidate) for every slot and every candidate, plus the greedily chosen
candidate per slot. Nothing here is CLAUDETTE-specific: an 8-slot Y/N vector
(CLAUDETTE-TOS), a single {safe, toxic} slot (ToxicChat), and an 8-slot k>2-way
vector (CTI-VSP) all fit the same "forced scaffold, free value token" shape and
can share this function. System-prompt construction and chat-template
rendering are the caller's responsibility and are not touched here.
"""
from __future__ import annotations

import torch


def score_slots(
    hf_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    scaffold_token_ids: list[list[int]],
    candidate_token_ids: list[list[int]],
    *,
    device: str,
) -> tuple[list[list[int]], list[list[list[float]]]]:
    """Score a fixed multi-slot scaffold via constrained decoding with a KV-cache.

    `input_ids`/`attention_mask` are an already left-padded, chat-templated
    prompt batch (batch, prompt_len) -- tokenization/padding stays the
    caller's job so it can also own the OOM batch-halving fallback around this
    call. `scaffold_token_ids` is the ordered, pre-tokenized list of the
    literal text fragments to force-feed before each slot's answer token
    (e.g. ["LTD:", "|TER:", ..., "|ARB:"] for CLAUDETTE-TOS, or [""] tokenized
    for a single-slot prompt whose scaffold is empty). `candidate_token_ids`
    gives, per slot, the token ids the model is allowed to answer with at that
    slot (e.g. [y_id, n_id] for a binary slot, or a k>2-way set for CTI-VSP);
    the same list is reused across slots when every slot shares an alphabet.

    For each slot in order this:
      1. forces the slot's scaffold fragment through the model (extending the
         KV-cache with a single forward call -- the first slot's call also
         carries the full prompt, since nothing is cached yet),
      2. reads logits[:, -1, :] -- the position fixed by construction right
         after the forced scaffold fragment, i.e. exactly where the model's
         value token was going to go -- a priori, never located post-hoc in
         decoded text,
      3. restricts those logits to `candidate_token_ids[slot]`, records
         log p(candidate) for each candidate (from the full-vocab
         log_softmax, so it is the model's actual probability, not one
         renormalized over only the candidate subset), and picks the
         argmax-scoring candidate as that slot's realized answer,
      4. feeds the chosen candidate token id back into the cache together with
         the next slot's scaffold fragment, preserving the autoregressive
         dependency between slots (the answer chosen at slot k is genuinely
         part of the context the model sees when answering slot k+1).

    The value token is never free-generated: it is always one of
    `candidate_token_ids[slot]`, so there is no failure mode where the model's
    answer can't be found or was merged into a neighboring token -- the
    logit difference between candidates measures exactly the decision that was
    forced and made.

    Because each forward call after the first only processes a handful of new
    tokens (one chosen token + one short scaffold fragment) rather than a full
    generated continuation, the peak logits tensor is (batch, few_tokens,
    vocab) instead of (batch, full_padded_seq_len, vocab) -- substantially
    lower peak vocab-logits memory than scoring by re-running the whole
    generated sequence through the model.

    Returns `(chosen_token_ids, log_probs)`:
      chosen_token_ids: batch-list of per-slot chosen token ids (one entry
        from `candidate_token_ids[slot]` per slot).
      log_probs: batch-list of per-slot list of log p(candidate), in the same
        order as `candidate_token_ids[slot]`.
    """
    batch_size = input_ids.shape[0]
    n_slots = len(scaffold_token_ids)
    if len(candidate_token_ids) != n_slots:
        raise ValueError(
            f"candidate_token_ids has {len(candidate_token_ids)} entries, expected "
            f"one per slot ({n_slots}, matching scaffold_token_ids)."
        )

    running_mask = attention_mask
    past_key_values = None
    chosen_tok = None  # (batch, 1) token id chosen at the previous slot, or None before slot 0

    chosen_ids_out: list[list[int]] = [[] for _ in range(batch_size)]
    log_probs_out: list[list[list[float]]] = [[] for _ in range(batch_size)]

    for slot in range(n_slots):
        scaffold = torch.tensor(scaffold_token_ids[slot], device=device, dtype=torch.long)
        scaffold = scaffold.unsqueeze(0).expand(batch_size, -1)

        if chosen_tok is None:
            # Nothing is cached yet: this single forward call carries the whole prompt
            # plus the first scaffold fragment.
            step_input = torch.cat([input_ids, scaffold], dim=1)
            new_token_count = scaffold.shape[1]
        else:
            step_input = torch.cat([chosen_tok, scaffold], dim=1)
            new_token_count = step_input.shape[1]

        running_mask = torch.cat(
            [running_mask, torch.ones(batch_size, new_token_count, dtype=running_mask.dtype, device=device)],
            dim=1,
        )

        # Standard left-padding-aware position ids: cumulative count of real
        # (non-pad) tokens up to each position, minus one. Slicing the last
        # step_input.shape[1] columns gives exactly the positions for the new
        # tokens in this forward call, matching how HF's generate() derives
        # position_ids for a left-padded batch with a KV-cache.
        full_position_ids = running_mask.long().cumsum(-1) - 1
        full_position_ids = full_position_ids.masked_fill(running_mask == 0, 0)
        position_ids = full_position_ids[:, -step_input.shape[1]:]

        outputs = hf_model(
            input_ids=step_input,
            attention_mask=running_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        logits = outputs.logits[:, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)

        cand_ids = torch.tensor(candidate_token_ids[slot], device=device, dtype=torch.long)
        cand_log_probs = log_probs[:, cand_ids]
        best_idx = cand_log_probs.argmax(dim=-1)
        chosen_tok_id = cand_ids[best_idx]

        for b in range(batch_size):
            chosen_ids_out[b].append(int(chosen_tok_id[b].item()))
            log_probs_out[b].append(cand_log_probs[b].tolist())

        chosen_tok = chosen_tok_id.unsqueeze(1)

    return chosen_ids_out, log_probs_out
