"""Metrics of the labeling benchmark - the same for both parsers of a run here and for the
experiments outputs (lb_experiments_eval.py).

Input per example: gold slot values and predicted slot values, a predicted value None = invalid.
An invalid slot counts as a false negative of its gold class and never as a false positive of
another class (it is a prediction that matches no class).

  toxicity_detection  f1_toxic        F1 of the class "toxic"
  claudette_tos       weighted_f1     per clause type (8 classes, "Y" positive) the F1, averaged
                                      with the support (number of gold "Y") as weight
  cti_vsp             macro_f1        per metric the mean F1 over its gold classes (classes observed
                                      in the test set with support > 0), averaged over the 8 metrics
Validity (every benchmark): valid_example_rate (all slots of an example valid), valid_slot_rate.
"""

from __future__ import annotations

from lb_parsers import CLAUDETTE_METRICS, CVSS_METRICS, SLOTS, TOXICITY_SLOT

PRIMARY_METRIC = {
    "toxicity_detection": "f1_toxic",
    "claudette_tos": "weighted_f1",
    "cti_vsp": "macro_f1",
}


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def class_stats(pairs: list[tuple[str, str | None]], cls: str) -> dict:
    """pairs = (gold, predicted); predicted None = invalid -> FN for its gold class, never FP."""
    tp = sum(1 for g, p in pairs if p == cls and g == cls)
    fp = sum(1 for g, p in pairs if p == cls and g != cls)
    fn = sum(1 for g, p in pairs if p != cls and g == cls)
    stats = prf(tp, fp, fn)
    stats["support"] = sum(1 for g, _ in pairs if g == cls)
    return stats


def validity(gold: list[dict], pred: list[dict], slots: list[str]) -> dict:
    n = len(gold)
    invalid_by_slot = {s: sum(1 for p in pred if p[s] is None) for s in slots}
    n_valid_examples = sum(1 for p in pred if all(p[s] is not None for s in slots))
    n_slots = n * len(slots)
    n_valid_slots = n_slots - sum(invalid_by_slot.values())
    return {
        "n_examples": n,
        "n_valid_examples": n_valid_examples,
        "valid_example_rate": n_valid_examples / n if n else 0.0,
        "n_slots": n_slots,
        "n_valid_slots": n_valid_slots,
        "valid_slot_rate": n_valid_slots / n_slots if n_slots else 0.0,
        "n_invalid_by_slot": invalid_by_slot,
    }


def metrics_toxicity(gold: list[dict], pred: list[dict]) -> dict:
    pairs = [(g[TOXICITY_SLOT], p[TOXICITY_SLOT]) for g, p in zip(gold, pred)]
    toxic = class_stats(pairs, "toxic")
    return {"f1_toxic": toxic["f1"], "toxic": toxic}


def metrics_claudette(gold: list[dict], pred: list[dict]) -> dict:
    per_class = {}
    for m in CLAUDETTE_METRICS:
        pairs = [("Y" if g[m] == "Y" else "N", p[m]) for g, p in zip(gold, pred)]
        per_class[m] = class_stats(pairs, "Y")
    total_support = sum(s["support"] for s in per_class.values())
    weighted = sum(s["support"] * s["f1"] for s in per_class.values()) / total_support if total_support else 0.0
    return {"weighted_f1": weighted, "per_class": per_class}


def metrics_cti_vsp(gold: list[dict], pred: list[dict]) -> dict:
    per_metric = {}
    for m in CVSS_METRICS:
        pairs = [(g[m], p[m]) for g, p in zip(gold, pred)]
        classes = {cls: class_stats(pairs, cls) for cls in sorted({g for g, _ in pairs})}
        f1s = [s["f1"] for s in classes.values()]  # every class here is a gold class, support > 0
        per_metric[m] = {"macro_f1": sum(f1s) / len(f1s) if f1s else 0.0, "classes": classes}
    return {
        "macro_f1": sum(per_metric[m]["macro_f1"] for m in CVSS_METRICS) / len(CVSS_METRICS),
        "per_metric": per_metric,
    }


METRIC_FUNCTIONS = {
    "toxicity_detection": metrics_toxicity,
    "claudette_tos": metrics_claudette,
    "cti_vsp": metrics_cti_vsp,
}


def evaluate(domain: str, gold: list[dict], pred: list[dict]) -> dict:
    """{"metric": name, "value": primary metric, "validity": ..., "details": ...}."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold vs {len(pred)} predicted examples.")
    details = METRIC_FUNCTIONS[domain](gold, pred)
    metric = PRIMARY_METRIC[domain]
    return {"metric": metric, "value": details[metric], "validity": validity(gold, pred, SLOTS[domain]),
            "details": details}
