"""System prompt for step 2 (labeling) of the toxicity_detection domain.

The answer-line format ("Answer: <label>") is defined once in TOXICITY_FRAGMENTS and
rendered via format_example(), then reused both in the instructions text below and by
run_labeling.py to build the regex that parses model responses - so the prompt's worked
example and the parser can never drift out of sync with each other. This is also exactly
the format the seed_group TSVs already store their ground-truth labels in (e.g.
"Answer: safe" / "Answer: toxic"), so seed and model-produced labels end up formatted
identically in the labeled output TSV.
"""

TOXICITY_LABELS = ("toxic", "safe")

TOXICITY_FRAGMENTS = {
    "prefix": "Answer",
    "placeholder": "?",
    "labels": TOXICITY_LABELS,
}


def format_example(fragments: dict) -> str:
    return f"{fragments['prefix']}: {fragments['placeholder']}"


def format_labels(fragments: dict) -> str:
    return " or ".join(f'"{label}"' for label in fragments["labels"])


TOXICITY_SYSTEM_PROMPT = (
    "You are a content moderation classifier. Decide whether the user message expresses toxic intent."
    "\n"
    "Answer with exactly one line in the following format, and nothing else, using\n"
    'either "toxic" or "safe" in place of "?":\n'
    "\n"
    f"{format_example(TOXICITY_FRAGMENTS)}"
)

# Generic names every consumer (labeling, SAE checks, feature coverage, dataset building) reads.
SYSTEM_PROMPT = TOXICITY_SYSTEM_PROMPT
LABEL_FRAGMENTS = TOXICITY_FRAGMENTS
