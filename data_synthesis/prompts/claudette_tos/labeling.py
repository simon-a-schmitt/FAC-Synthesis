"""System prompt for step 2 (labeling) of the claudette_tos domain.

Like cti_vsp's CVSS vector, a claudette_tos label is a multi-field template with
independent per-field values, so it reuses labeling/run_labeling.py's "vector" kind rather than
toxicity_detection's single "<prefix>: <label>" line. Unlike CVSS, every field here shares
the same Y/N alphabet - each of the eight CLAUDETTE unfairness metrics (LTD, TER, CH, CR,
USE, LAW, J, ARB; see Lippi et al., "CLAUDETTE: an automated detector of potentially
unfair clauses in online terms of service") is answered independently for the sentence.
CLAUDETTE_FRAGMENTS declares the exact output template and, for each metric, the letters
the model is allowed to answer with - so the prompt's worked example and
labeling/run_labeling.py's parser can never drift out of sync with each other. Seed_group TSVs
must store their ground-truth label in this exact rendered format (e.g.
"LTD: N|TER: N|CH: N|CR: N|USE: N|LAW: N|J: N|ARB: N"), so seed and model-produced labels
end up formatted identically in the labeled output TSV.
"""

CLAUDETTE_METRICS = {
    "LTD": ("Y", "N"),
    "TER": ("Y", "N"),
    "CH": ("Y", "N"),
    "CR": ("Y", "N"),
    "USE": ("Y", "N"),
    "LAW": ("Y", "N"),
    "J": ("Y", "N"),
    "ARB": ("Y", "N"),
}

CLAUDETTE_FIELD_ORDER = ("LTD", "TER", "CH", "CR", "USE", "LAW", "J", "ARB")

CLAUDETTE_METRIC_NAMES = {
    "LTD": "limitation of liability",
    "TER": "unilateral termination",
    "CH": "unilateral change",
    "CR": "content removal",
    "USE": "contract by using",
    "LAW": "choice of law",
    "J": "jurisdiction",
    "ARB": "arbitration",
}

# Spaced format ("LTD: N", not "LTD:N") - what the seed_group TSVs already store their
# ground-truth labels in (unlike cti_vsp's unspaced "AV:N" CVSS notation).
CLAUDETTE_TEMPLATE = "LTD: {LTD}|TER: {TER}|CH: {CH}|CR: {CR}|USE: {USE}|LAW: {LAW}|J: {J}|ARB: {ARB}"

CLAUDETTE_FRAGMENTS = {
    "kind": "vector",
    "template": CLAUDETTE_TEMPLATE,
    "fields": CLAUDETTE_METRICS,
    "field_order": CLAUDETTE_FIELD_ORDER,
}


def format_example(fragments: dict, placeholder: str = "?") -> str:
    return fragments["template"].format(**{field: placeholder for field in fragments["field_order"]})


def format_metric_options(fragments: dict) -> str:
    return "\n".join(
        f"  {field:<4} {CLAUDETTE_METRIC_NAMES[field]}" for field in fragments["field_order"]
    )


CLAUDETTE_SYSTEM_PROMPT = (
    "You are analyzing a single sentence from the Terms of Service of an\n"
    "online platform under EU consumer law (Directive 93/13/EEC).\n"
    "\n"
    "Decide, for each of the following eight clause types, whether the\n"
    "sentence contains a potentially unfair clause of that type:\n"
    "\n"
    f"{format_metric_options(CLAUDETTE_FRAGMENTS)}\n"
    "\n"
    "Most sentences contain no unfair clause of any type.\n"
    "\n"
    "Answer with exactly one line in the following format, using Y or N for\n"
    "each type, and nothing else:\n"
    "\n"
    # Derived from the same fragments labeling/run_labeling.py's vector parser is built from, so
    # the prompt and the parser can never drift apart (see build_vector_regex).
    f"{format_example(CLAUDETTE_FRAGMENTS)}"
)

# Generic names every consumer (labeling, SAE checks, feature coverage, dataset building) reads.
SYSTEM_PROMPT = CLAUDETTE_SYSTEM_PROMPT
LABEL_FRAGMENTS = CLAUDETTE_FRAGMENTS
