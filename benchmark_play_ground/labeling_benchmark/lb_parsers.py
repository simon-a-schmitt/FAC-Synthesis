"""The parser every raw answer goes through: the labeling parser of
data_synthesis/labeling/run_labeling.py (build_answer_regex + normalize_label_line). One line of the
answer must match the full answer template of prompts/<domain>/labeling.py; otherwise the whole
answer is invalid - all slots None, never a partial parse. parse() returns {slot: value or None};
None = invalid (counts as a false negative of its gold class, never as a false positive; lb_metrics.py).
"""

from __future__ import annotations

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


class LabelingParser:
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


def gold_values(domain: str, records: list[dict], source: str) -> list[dict]:
    """Slot values of every record's gt via the labeling parser; aborts on a gt outside the labeling format."""
    parser = LabelingParser(domain)
    out = []
    for i, rec in enumerate(records):
        values = parser.parse(rec["gt"])
        if any(v is None for v in values.values()):
            raise SystemExit(f"{source}: gt of record {i} is not in the labeling format: {rec['gt']!r}")
        out.append(values)
    return out
