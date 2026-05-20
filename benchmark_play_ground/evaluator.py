import re
import json
from typing import List, Dict


def extract_label_from_text(text: str) -> str:
    # Normalize and search for A-E
    if not text:
        return ""
    text = text.strip()
    # Common patterns: 'A', 'A.', 'Answer: A', 'The correct answer is A'
    m = re.search(r"\b([A-E])\b", text.upper())
    if m:
        return m.group(1)
    # fallback: first char if it's A-E
    if text and text[0].upper() in "ABCDE":
        return text[0].upper()
    return ""


def evaluate_predictions(records: List[Dict], output_path: str = None) -> Dict:
    total = 0
    correct = 0
    for r in records:
        total += 1
        if r.get("predicted") == r.get("gt") and r.get("predicted") != "":
            correct += 1

    acc = correct / total if total else 0.0
    summary = {"total": total, "correct": correct, "accuracy": acc}
    if output_path:
        with open(output_path, "w") as fh:
            json.dump(summary, fh, indent=2)
    return summary
