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


PUBMEDQA_LABELS = ("yes", "no", "maybe")


def extract_pubmedqa_label_from_text(text: str) -> str:
    """Extract a yes/no/maybe answer from raw model output."""
    if not text:
        return ""
    text = text.strip().lower()
    m = re.search(r"\b(yes|no|maybe)\b", text)
    return m.group(1) if m else ""


def evaluate_pubmedqa_predictions(records: List[Dict], output_path: str = None) -> Dict:
    """Compute accuracy, macro F1, per-class recall and a confusion matrix.

    `predicted`/`gt` values that fall outside PUBMEDQA_LABELS (e.g. empty strings
    from unparseable model output) are counted towards totals/accuracy but are
    treated as a distinct "unparsed" bucket so they don't silently collapse into
    one of the three real classes in the confusion matrix.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        recall_score,
    )

    total = len(records)
    y_true = [r.get("gt", "") for r in records]
    y_pred = [r.get("predicted", "") for r in records]
    correct = sum(1 for t, p in zip(y_true, y_pred) if p == t and p != "")

    labels = list(PUBMEDQA_LABELS)
    if any(p not in PUBMEDQA_LABELS for p in y_pred):
        labels = labels + ["unparsed"]
    y_pred_bucketed = [p if p in PUBMEDQA_LABELS else "unparsed" for p in y_pred]

    accuracy = accuracy_score(y_true, y_pred_bucketed) if total else 0.0
    macro_f1 = f1_score(y_true, y_pred_bucketed, labels=list(PUBMEDQA_LABELS), average="macro", zero_division=0) if total else 0.0
    per_class_recall_values = recall_score(y_true, y_pred_bucketed, labels=list(PUBMEDQA_LABELS), average=None, zero_division=0) if total else [0.0] * len(PUBMEDQA_LABELS)
    per_class_recall = {label: float(r) for label, r in zip(PUBMEDQA_LABELS, per_class_recall_values)}
    cm = confusion_matrix(y_true, y_pred_bucketed, labels=labels) if total else []

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_recall": per_class_recall,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist() if total else [],
        },
    }
    if output_path:
        with open(output_path, "w") as fh:
            json.dump(summary, fh, indent=2)
    return summary


def format_confusion_matrix(confusion_matrix_summary: Dict) -> str:
    """Render the confusion_matrix block from evaluate_pubmedqa_predictions as a text table."""
    labels = confusion_matrix_summary["labels"]
    matrix = confusion_matrix_summary["matrix"]
    col_width = max(len(l) for l in labels + ["true\\pred"]) + 2

    header = "true\\pred".ljust(col_width) + "".join(l.rjust(col_width) for l in labels)
    lines = [header]
    for label, row in zip(labels, matrix):
        lines.append(label.ljust(col_width) + "".join(str(v).rjust(col_width) for v in row))
    return "\n".join(lines)
