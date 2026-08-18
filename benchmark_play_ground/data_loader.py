import json
import pandas as pd
from typing import List, Dict, Tuple, Optional


DEFAULT_CASEHOLD_INSTRUCTION = (
    "Identify the single correct legal holding statement from options A-E to fill the <HOLDING> placeholder in the citation and output only the corresponding letter."
)


def map_casehold_label(label) -> str:
    label_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", "0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "A": "A", "B": "B", "C": "C", "D": "D", "E": "E"}
    return label_map.get(label, label_map.get(str(label), str(label)))


def strip_casehold_instruction(text: str, instruction: str = DEFAULT_CASEHOLD_INSTRUCTION) -> str:
    text = str(text).strip()
    instruction = instruction.strip()
    if text.startswith(instruction):
        return text[len(instruction):].lstrip("\n\r \t").strip()
    return text


def build_casehold_prompt_from_row(row: pd.Series, instruction_prompt: str) -> str:
    prompt_parts = [instruction_prompt, strip_casehold_instruction(row.get("Citing Text", row.get("prompt", "")))]
    # Try to collect five options
    for i, letter in enumerate("ABCDE"):
        key = f"Holding Statement {i}"
        if key in row:
            prompt_parts.append(f"{letter}. {row[key]}")
    # If explicit options not found, try columns 0..4
    if len(prompt_parts) == 1 + 1:
        # fallback: search numeric columns like 0..4
        for i, letter in enumerate("ABCDE"):
            if str(i) in row.index:
                prompt_parts.append(f"{letter}. {row[str(i)]}")
    return " \n".join(prompt_parts)


def load_casehold_tsv(path: str, instruction_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Load a CaseHold-style TSV and return list of records with keys: prompt, gt_label

    The loader is tolerant to several possible column layouts used in this repo.
    """
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")

    records = []
    # If file already exported by notebook (prompt + gt holding label)
    if "prompt" in df.columns and any(c.lower().startswith("gt") for c in df.columns):
        gt_col = [c for c in df.columns if c.lower().startswith("gt")][0]
        for _, r in df.iterrows():
            rec = {"prompt": strip_casehold_instruction(r["prompt"]), "gt": map_casehold_label(r[gt_col])}
            records.append(rec)
        return records

    # If it contains Citing Text and Holding Statement columns
    if "Citing Text" in df.columns:
        gt_col = None
        for c in ["GT Holding Label", "GT Holding", "gt holding label", "gt"]:
            if c in df.columns:
                gt_col = c
                break
        for _, r in df.iterrows():
            prompt = build_casehold_prompt_from_row(r, instruction_prompt or "Identify the single correct legal holding statement from options A-E to fill the <HOLDING> placeholder in the citation and output only the corresponding letter.")
            gt = map_casehold_label(r.get(gt_col, "")) if gt_col else ""
            records.append({"prompt": prompt, "gt": gt})
        return records

    # Fallback: if there are only two columns 'prompt' and 'gt'
    cols = list(df.columns)
    if len(cols) >= 2:
        # try to heuristically pick columns
        prompt_col = cols[0]
        gt_col = cols[1]
        for _, r in df.iterrows():
            records.append({"prompt": str(r[prompt_col]), "gt": map_casehold_label(r[gt_col])})
        return records

    raise ValueError(f"Unrecognized CaseHold TSV format for file: {path}")


PUBMEDQA_LABELS = ("yes", "no", "maybe")


def map_pubmedqa_label(label) -> str:
    label = str(label).strip().lower()
    return label if label in PUBMEDQA_LABELS else label


def load_pubmedqa_tsv(path: str) -> List[Dict[str, str]]:
    """Load a PubMedQA-style TSV and return list of records with keys: prompt, gt

    Produced by benchmarks/pubmedqa/transform_pubmedqa_to_tsv.py: each row holds a
    fully-formed prompt (context + question + answer instruction) and a gt label
    of "yes", "no", or "maybe". The file has no header row, so column names are
    assigned positionally.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, header=None, names=["prompt", "gt"], quoting=3).fillna("")

    # If the first row is actually a header (e.g. "prompt"/"gt label"), drop it.
    if df.iloc[0]["prompt"].strip().lower() == "prompt":
        df = df.iloc[1:].reset_index(drop=True)

    records = []
    for _, r in df.iterrows():
        prompt = str(r["prompt"]).replace("\\n", "\n")
        records.append({"prompt": prompt, "gt": map_pubmedqa_label(r["gt"])})
    return records


def load_cti_vsp_tsv(path: str) -> List[Dict[str, str]]:
    """Load a CTI-VSP-style TSV and return list of records with keys: prompt, gt

    Produced by benchmarks/cti_vsp/build_benchmark.py: each row holds a fully-formed
    prompt (CVSS instructions + CVE description) and a gt CVSS v3.1 vector string
    such as "CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:N". The file has no header
    row, so column names are assigned positionally.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, header=None, names=["prompt", "gt"], quoting=3).fillna("")

    # If the first row is actually a header (e.g. "Prompt"/"GT"), drop it.
    if df.iloc[0]["prompt"].strip().lower() == "prompt":
        df = df.iloc[1:].reset_index(drop=True)

    records = []
    for _, r in df.iterrows():
        prompt = str(r["prompt"]).replace("\\n", "\n")
        records.append({"prompt": prompt, "gt": str(r["gt"]).strip()})
    return records


def load_claudette_tsv(path: str) -> List[Dict[str, str]]:
    """Load a CLAUDETTE-TOS-style TSV and return list of records with keys: prompt, gt

    Each row holds a single ToS sentence and a gt unfairness-type vector string
    such as "LTD:N|TER:N|CH:N|CR:N|USE:N|LAW:N|J:N|ARB:Y|". The file has no header
    row, so column names are assigned positionally.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, header=None, names=["prompt", "gt"], quoting=3).fillna("")

    # If the first row is actually a header (e.g. "sentence"/"labels"), drop it.
    if df.iloc[0]["prompt"].strip().lower() == "prompt":
        df = df.iloc[1:].reset_index(drop=True)

    records = []
    for _, r in df.iterrows():
        prompt = str(r["prompt"]).replace("\\n", "\n")
        records.append({"prompt": prompt, "gt": str(r["gt"]).strip()})
    return records


def load_cti_vsp_metric_classes(path: str) -> Dict[str, List[str]]:
    """Load the per-metric class list from cti_vsp_gt_metric_distribution.json.

    Returns e.g. {"AV": ["N", "L", "A", "P"], "AC": ["L", "H"], ...} — the set of
    classes observed in the ground truth for each CVSS base metric, used as the
    `labels` argument to macro-F1 so scores don't depend on classes a model
    happens to predict but that never occur in the ground truth.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {metric: list(classes.keys()) for metric, classes in data["metric_distribution"].items()}
