import re
import json
from typing import Dict, Any, List, Tuple

def _extract_json_block(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("Model output not string.")
    m = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text, flags=re.I)
    if not m:
        m = re.search(r"\{[\s\S]+?\}", text)
    if not m:
        raise ValueError("No JSON found in output.")
    candidate = m.group(1) if m.lastindex else m.group(0)
    candidate = candidate.strip().replace("“", '"').replace("”", '"').replace("’", "'")
    candidate = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode error: {e}")

_HUMAN_RE = re.compile(r"^\s*human\s*:\s*", re.I)
_ASSIST_RE = re.compile(r"^\s*assistant\s*:\s*", re.I)

def _parse_transcript(s: str) -> List[Dict[str, str]]:
    lines = [l.strip() for l in s.replace("\r", "").split("\n") if l.strip()]
    msgs = []
    for l in lines:
        if _HUMAN_RE.match(l):
            msgs.append({"role": "user", "content": _HUMAN_RE.sub("", l).strip()})
        elif _ASSIST_RE.match(l):
            msgs.append({"role": "assistant", "content": _ASSIST_RE.sub("", l).strip()})
    return msgs

def _check_alternating_roles(turns: List[Dict[str, str]]) -> bool:
    if len(turns) < 2 or len(turns) % 2 != 0:
        return False
    for i, m in enumerate(turns):
        want = "user" if i % 2 == 0 else "assistant"
        if m.get("role") != want or not m.get("content", "").strip():
            return False
    return True

def _validate_multi_turn_pair(chosen: str, rejected: str, max_exchanges: int = 3) -> Tuple[bool, str]:
    c = _parse_transcript(chosen)
    r = _parse_transcript(rejected)
    if len(c) < 4 or len(c) > 2 * max_exchanges or len(c) % 2 != 0:
        return False, "invalid_turn_count"
    if len(c) != len(r):
        return False, "turn_count_mismatch"
    if not _check_alternating_roles(c) or not _check_alternating_roles(r):
        return False, "bad_role_pattern"
    for i in range(len(c) - 1):
        c_i = c[i]["content"].strip().rstrip(".!?")
        r_i = r[i]["content"].strip().rstrip(".!?")
        if c_i != r_i:
            return False, f"content_mismatch_at_{i}"
    if c[-1]["content"].strip() == r[-1]["content"].strip():
        return False, "last_assistant_identical"
    return True, "ok"

def _validate_single_turn_pair(chosen: str, rejected: str) -> Tuple[bool, str]:
    c = _parse_transcript(chosen)
    r = _parse_transcript(rejected)
    if len(c) != 2 or len(r) != 2:
        return False, "not_single_turn"
    if not _check_alternating_roles(c) or not _check_alternating_roles(r):
        return False, "bad_role_pattern"
    if c[0]["content"].strip() != r[0]["content"].strip():
        return False, "human_not_identical"
    if c[1]["content"].strip() == r[1]["content"].strip():
        return False, "assistant_identical"
    return True, "ok"

def parse_instruction_input_pairs(text: str) -> List[Tuple[str, str]]:
    pairs = []
    blocks = re.split(r"\n\s*\d+\.\s*", "\n" + text)
    for block in blocks:
        b = block.strip()
        if not b:
            continue
        m = re.search(r'[\s\-–—\*\u2022]*Input\s*:\s*(.*)$', b, flags=re.I | re.S)
        if m:
            ins = b[:m.start()].strip()
            inp = m.group(1).strip()
        else:
            ins, inp = b, "<noinput>"
        pairs.append((ins, inp))
    return pairs

def _clean_ins_inp(ins: str, inp: str) -> Tuple[str, str]:
    ins = re.sub(r'^\s*(Instruction|Instructions)\s*:\s*', '', ins.strip(), flags=re.I)
    ins = re.sub(r'\[\s*FID\s*=\s*\d+\s*\]\s*', '', ins)
    inp = re.sub(r'^\s*(Input|Inputs)\s*:\s*', '', inp.strip(), flags=re.I)
    if inp.lower() in {"<noinput>", "noinput", "<no input>"} or not inp:
        inp = "<noinput>"
    return ins.strip(), inp.strip()

def _prepend_input_to_human(transcript: str, ins: str, inp: str) -> str:
    turns = _parse_transcript(transcript)
    if not turns or turns[0]["role"] != "user":
        return transcript

    expected_first = ins.strip() if inp == "<noinput>" else f"{ins.strip()}\n\nInput: {inp.strip()}"
    turns[0]["content"] = expected_first

    out_lines = []
    for t in turns:
        prefix = "Human:" if t["role"] == "user" else "Assistant:"
        out_lines.append(f"{prefix} {t['content'].strip()}")
        out_lines.append("")
    return "\n".join(out_lines).strip()

