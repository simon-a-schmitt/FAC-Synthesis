"""The two parsers every raw answer goes through. Both return {slot: value or None}; None = the slot
is invalid (counts as a false negative of its gold class, never as a false positive; lb_metrics.py).

  exact    the labeling parser (data_synthesis/labeling/run_labeling.py build_answer_regex +
           normalize_label_line): one line of the answer must match the full answer template of
           prompts/<domain>/labeling.py - all slots valid or none.
  lenient  the parsers of the experiments benchmark scripts, copied verbatim (see the source names
           below), applied to the whole answer:
             toxicity_detection  first "Answer: toxic|safe" anywhere (case-insensitive)
             claudette_tos       per slot the first "<KEY>: Y|N"; a slot without a match is invalid
                                 (in run_claudette_benchmark.py --mode api it counted as "N")
             cti_vsp             cut at the first stop string, then per metric the first
                                 "<KEY>: <valid letter>"; a metric without a match is invalid
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "data_synthesis"))
sys.path.insert(0, str(PROJECT_DIR / "data_synthesis" / "labeling"))

from prompts import LabelingPrompt, load_labeling_prompt  # noqa: E402
from run_labeling import build_answer_regex, normalize_label_line  # noqa: E402

TOXICITY_SLOT = "label"
TOXICITY_CLASSES = ["toxic", "safe"]
CLAUDETTE_METRICS = ["LTD", "TER", "CH", "CR", "USE", "LAW", "J", "ARB"]
CVSS_METRICS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
SLOTS = {
    "toxicity_detection": [TOXICITY_SLOT],
    "claudette_tos": CLAUDETTE_METRICS,
    "cti_vsp": CVSS_METRICS,
}
PARSERS = ("exact", "lenient")


# ---------------------------------------------------------------------------
# exact (run_labeling.py)
# ---------------------------------------------------------------------------

class ExactParser:
    """normalize_label_line of run_labeling.py, with the slot values of the normalized line."""

    def __init__(self, domain: str, prompt: LabelingPrompt | None = None):
        self.fragments = (prompt or load_labeling_prompt(domain)).fragments
        self.regex = build_answer_regex(self.fragments)
        self.slots = SLOTS[domain]
        if self.fragments.get("kind") == "vector" and list(self.fragments["field_order"]) != self.slots:
            raise SystemExit(f"{domain}: labeling fields {self.fragments['field_order']} != benchmark slots {self.slots}")
        if self.fragments.get("kind") != "vector" and sorted(self.fragments["labels"]) != sorted(TOXICITY_CLASSES):
            raise SystemExit(f"{domain}: labeling labels {self.fragments['labels']} != {TOXICITY_CLASSES}")

    def parse(self, text: str) -> dict:
        line = normalize_label_line(text or "", self.fragments, self.regex)
        if line is None:
            return {s: None for s in self.slots}
        match = self.regex.match(line)
        if self.fragments.get("kind") == "vector":
            return {f: match.group(f).upper() for f in self.slots}
        return {self.slots[0]: match.group(1).lower()}


# ---------------------------------------------------------------------------
# lenient - copied from the experiments benchmark scripts
# ---------------------------------------------------------------------------

# benchmark_play_ground/run_toxicity_benchmark.py
TOXICITY_ANSWER_RE = re.compile(r"Answer:\s*(toxic|safe)", re.IGNORECASE)


def extract_toxicity_label_from_text(text: str) -> str | None:
    """Pull "toxic" / "safe" out of an "Answer: ..." string (ground truth only)."""
    match = TOXICITY_ANSWER_RE.search(text or "")
    return match.group(1).lower() if match else None


# benchmark_play_ground/run_claudette_benchmark.py
# --mode api only: the free-generated answer is parsed slot by slot, so one
# malformed slot does not discard the other seven (\b keeps e.g. "J:" from
# matching inside another key). A slot that cannot be parsed counts as "N".
# (Here: a slot that cannot be parsed is invalid, see LenientParser.)
CLAUDETTE_SLOT_RE = {m: re.compile(rf"\b{m}:\s*([YN])\b") for m in CLAUDETTE_METRICS}


# benchmark_play_ground/run_cti_vsp_benchmark.py
# Valid letters per CVSS base metric.
CVSS_METRIC_VALUE_CHARS = {
    "AV": "NALP",
    "AC": "LH",
    "PR": "NLH",
    "UI": "NR",
    "S": "UC",
    "C": "NLH",
    "I": "NLH",
    "A": "NLH",
}

# Per-metric "KEY:\s*VALUE" regex, mirroring CLAUDETTE_VECTOR_RE's tolerance for
# whitespace after the colon, but restricted to that metric's own valid letters
# so a garbled or out-of-range value is never accepted -- it becomes None
# instead, same as a metric missing from the text entirely (\b before the key
# keeps e.g. "\bA:" from matching inside "AV:"/"AC:", since there is no word
# boundary between "A" and the following "V"/"C").
CVSS_METRIC_RE = {
    m: re.compile(rf"\b{m}:\s*([{chars}])") for m, chars in CVSS_METRIC_VALUE_CHARS.items()
}

CTI_VSP_STOP_STRINGS = ["\nCVE Description:", "\n\n"]


def extract_cvss_metrics_from_text(text: str) -> dict:
    """Extract each CVSS base metric's value independently via CVSS_METRIC_RE.

    Used both for the ground-truth vector loaded verbatim from the TSV / the
    few-shot labels, and for the model's raw free-generated output. A metric
    whose value cannot be found, or isn't one of that metric's valid letters,
    is left as None -- a parse failure for that slot; see
    evaluate_cti_vsp_predictions.
    """
    if not text:
        return {m: None for m in CVSS_METRICS}
    metrics = {}
    for m in CVSS_METRICS:
        match = CVSS_METRIC_RE[m].search(text)
        metrics[m] = match.group(1) if match else None
    return metrics


def cut_at_stop_strings(text: str) -> str:
    """The returned text up to the first CTI_VSP_STOP_STRINGS match (leading whitespace
    stripped first, so a leading blank line does not cut away the whole answer)."""
    text = text.lstrip()
    cut = min((i for i in (text.find(stop) for stop in CTI_VSP_STOP_STRINGS) if i >= 0), default=len(text))
    return text[:cut]


class LenientParser:
    def __init__(self, domain: str):
        if domain not in SLOTS:
            raise SystemExit(f"Unknown benchmark {domain!r}.")
        self.domain = domain

    def parse(self, text: str) -> dict:
        text = text or ""
        if self.domain == "toxicity_detection":
            return {TOXICITY_SLOT: extract_toxicity_label_from_text(text)}
        if self.domain == "claudette_tos":
            out = {}
            for m in CLAUDETTE_METRICS:
                match = CLAUDETTE_SLOT_RE[m].search(text)
                out[m] = match.group(1) if match else None
            return out
        return extract_cvss_metrics_from_text(cut_at_stop_strings(text))


def make_parsers(domain: str) -> dict:
    return {"exact": ExactParser(domain), "lenient": LenientParser(domain)}


def gold_values(domain: str, records: list[dict], source: str) -> list[dict]:
    """Slot values of every record's gt via the exact parser; aborts on a gt outside the labeling format."""
    exact = ExactParser(domain)
    out = []
    for i, rec in enumerate(records):
        values = exact.parse(rec["gt"])
        if any(v is None for v in values.values()):
            raise SystemExit(f"{source}: gt of record {i} is not in the labeling format: {rec['gt']!r}")
        out.append(values)
    return out
