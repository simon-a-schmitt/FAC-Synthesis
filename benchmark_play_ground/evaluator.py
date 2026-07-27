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


CVSS_METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]

# Matches an individual "KEY:VALUE" pair, e.g. "AV:N" or "S:U". The \b before the
# key relies on CVSS metric keys only ever being glued to another letter (as in
# "AC", "UI"), never to a metric on the other side of a "/" or start-of-string,
# so e.g. the "C" in "AC:L" is never mistaken for the standalone "C" (Confidentiality) metric.
_CVSS_TOKEN_RE = re.compile(r"\b(AV|AC|PR|UI|S|C|I|A):([A-Z])\b")

# Matches a full (or near-full) CVSS v3 vector string: an optional "CVSS:3.x/"
# header followed by 6-8 "KEY:VALUE" pairs. Used to isolate the vector from
# surrounding prose in raw model output before parsing individual metrics.
_CVSS_VECTOR_RE = re.compile(
    r"(?:CVSS:3\.[01]/)?(?:(?:AV|AC|PR|UI|S|C|I|A):[A-Z]/){5,7}(?:AV|AC|PR|UI|S|C|I|A):[A-Z]"
)


def extract_cvss_vector_from_text(text: str) -> str:
    """Return the last CVSS vector-shaped substring found in text, or "" if none."""
    if not text:
        return ""
    matches = list(_CVSS_VECTOR_RE.finditer(text))
    return matches[-1].group(0) if matches else ""


def extract_cvss_metrics_from_text(text: str) -> Dict[str, str]:
    """Extract per-metric values (AV, AC, PR, UI, S, C, I, A) from a CVSS vector
    string or from free-form text containing one (e.g. raw model output).

    Prefers parsing a full vector substring if one can be found (so prose that
    happens to mention e.g. "AV:N" earlier in the reasoning doesn't override the
    final answer); falls back to scanning the whole text otherwise. Metrics that
    can't be found are returned as "".
    """
    metrics = {m: "" for m in CVSS_METRICS}
    if not text:
        return metrics

    vector = extract_cvss_vector_from_text(text)
    scan_text = vector if vector else text

    for m in _CVSS_TOKEN_RE.finditer(scan_text):
        metrics[m.group(1)] = m.group(2)
    return metrics


def evaluate_cti_vsp_predictions(
    records: List[Dict], metric_classes: Dict[str, List[str]], output_path: str = None
) -> Dict:
    """Compute macro-F1 per CVSS base metric and the overall macro-F1 (the mean
    of the per-metric macro-F1 scores) over a list of prediction records.

    Each record must carry `predicted_metrics` and `gt_metrics` dicts (as produced
    by `extract_cvss_metrics_from_text`). `metric_classes` restricts the F1
    computation for each metric to the classes observed in the ground truth
    (from cti_vsp_gt_metric_distribution.json), so an unparsed/unseen prediction
    value simply counts as a miss rather than inventing a new class.
    """
    from sklearn.metrics import f1_score

    total = len(records)
    per_metric_macro_f1: Dict[str, float] = {}
    for metric in CVSS_METRICS:
        labels = metric_classes.get(metric, [])
        y_true = [r.get("gt_metrics", {}).get(metric, "") for r in records]
        y_pred = [r.get("predicted_metrics", {}).get(metric, "") for r in records]
        f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) if total else 0.0
        per_metric_macro_f1[metric] = float(f1)

    overall_macro_f1 = sum(per_metric_macro_f1.values()) / len(per_metric_macro_f1) if per_metric_macro_f1 else 0.0
    exact_match = sum(
        1 for r in records if r.get("predicted_vector") and r.get("predicted_vector") == r.get("gt")
    )

    summary = {
        "total": total,
        "exact_match": exact_match,
        "exact_match_accuracy": exact_match / total if total else 0.0,
        "overall_macro_f1": overall_macro_f1,
        "per_metric_macro_f1": per_metric_macro_f1,
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
