#!/usr/bin/env python3
"""Can the canonical CVSS format be scored with slot_scoring?

  python -m benchmark_play_ground.check_cvss_plan \
      --model-path /path/to/Llama-3.1-8B-Instruct \
      --data-tsv benchmarks/cti_vsp/cti_vsp_benchmark_test_500.tsv

Three stages, from cheap to authoritative:
  1. vocab   -- is ':X' (canonical) / ' X' (spaced) a single token in isolation?
  2. plan    -- does build_slot_plan() accept each template in chat context?
  3. replay  -- does forcing the plan reproduce the canonical tokenization of
                EVERY distinct GT vector in the TSV (not just the two value
                patterns build_slot_plan replays)?
Exit code 0 iff the canonical template passes stages 2 and 3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from transformers import AutoTokenizer

from benchmark_play_ground.data_loader import load_cti_vsp_tsv
from benchmark_play_ground.slot_scoring import (
    SlotPlanError, build_slot_plan, format_example, make_fragments,
)

CVSS = [("AV", ["N", "A", "L", "P"]), ("AC", ["L", "H"]), ("PR", ["N", "L", "H"]),
        ("UI", ["N", "R"]), ("S", ["U", "C"]), ("C", ["H", "L", "N"]),
        ("I", ["H", "L", "N"]), ("A", ["H", "L", "N"])]
NAMES = [m for m, _ in CVSS]
CANDS = [v for _, v in CVSS]
ALPHABET = sorted({c for v in CANDS for c in v})

VARIANTS = {
    "canonical": dict(assign=":"),    # CVSS:3.1/AV:N/AC:L/...
    "spaced":    dict(assign=": "),   # CVSS:3.1/AV: N/AC: L/...
}


def enc(tok, s: str) -> list[int]:
    return tok.encode(s, add_special_tokens=False)


def leads(tok) -> list[str]:
    # System prompt content is irrelevant for the split points: the assistant
    # header ends in special tokens that no BPE merge crosses.
    return [
        tok.apply_chat_template(
            [{"role": "system", "content": "test"}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True,
        )
        for u in ("CVE Description: short.",
                  "CVE Description: A considerably longer description, with "
                  "punctuation: commas, colons, paths like /api/v1/ and a period.")
    ]


def stage_vocab(tok) -> None:
    print("[1] vocab (isolated tokenization)")
    for x in ALPHABET:
        row = []
        for label, s in (("canonical", f":{x}"), ("spaced", f" {x}")):
            ids = enc(tok, s)
            pieces = tok.convert_ids_to_tokens(ids)
            flag = "ok " if len(ids) == 1 else "MULTI"
            row.append(f"{label} {s!r:>6} -> {flag} {pieces}")
        print("   " + "   |   ".join(row))


def parse_gt(gt: str) -> list[str] | None:
    parts = dict(seg.split(":", 1) for seg in gt.split("/")[1:] if ":" in seg)
    vals = [parts.get(m, "").strip() for m in NAMES]
    return vals if all(vals) else None


def stage_replay(tok, plan, lead: str, gt_values: list[list[str]]) -> list[str]:
    lead_ids = enc(tok, lead)
    idx = [{c: i for i, c in enumerate(s.candidates)} for s in plan.slots]
    bad = []
    for vals in gt_values:
        replay = list(lead_ids)
        for s, m, v in zip(plan.slots, idx, vals):
            replay += list(s.fragment_ids) + [s.candidate_ids[m[v]]]
        answer = plan.answer_string(vals)
        if replay != enc(tok, lead + answer):
            bad.append(answer)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-tsv", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    lds = leads(tok)

    gts = sorted({r["gt"] for r in load_cti_vsp_tsv(args.data_tsv)})
    gt_values = [v for v in map(parse_gt, gts) if v is not None]
    print(f"{len(gts)} distinct GT vectors, {len(gt_values)} parseable\n")

    stage_vocab(tok)

    passed = {}
    for label, kw in VARIANTS.items():
        frags = make_fragments(NAMES, sep="/", prefix="CVSS:3.1/", **kw)
        print(f"\n[2] plan -- {label}: {format_example(frags)}")
        try:
            plan = build_slot_plan(tok, NAMES, frags, CANDS, leads=lds)
        except SlotPlanError as e:
            print(f"   REJECTED: {e}")
            passed[label] = False
            continue
        print("   " + plan.describe(tok).replace("\n", "\n   "))

        bad = [a for lead in lds for a in stage_replay(tok, plan, lead, gt_values)]
        print(f"[3] replay -- {label}: {len(gt_values) * len(lds) - len(bad)}"
              f"/{len(gt_values) * len(lds)} ok")
        for a in bad[:5]:
            print(f"   MISMATCH: {a!r}")
        passed[label] = not bad

    print("\nresult:", {k: ("PASS" if v else "FAIL") for k, v in passed.items()})
    return 0 if passed.get("canonical") else 1


if __name__ == "__main__":
    raise SystemExit(main())