"""System prompt for step 2 (labeling) of the cti_vsp domain.

Unlike toxicity_detection's single "<prefix>: <label>" line, a cti_vsp label is a full
CVSS v3.1 Base vector with 8 independent metric fields. CVSS_FRAGMENTS (kind="vector")
declares the exact output template and, for each metric, the letters the model is
allowed to answer with - so the prompt's worked example and run_labeling.py's parser
can never drift out of sync with each other. Seed_group TSVs must store their
ground-truth label in this exact rendered format (e.g.
"CVSS:3.1/AV: N/AC: L/PR: L/UI: N/S: U/C: H/I: H/A: H"), so seed and model-produced
labels end up formatted identically in the labeled output TSV.
"""

CVSS_METRICS = {
    "AV": ("N", "A", "L", "P"),
    "AC": ("L", "H"),
    "PR": ("N", "L", "H"),
    "UI": ("N", "R"),
    "S": ("U", "C"),
    "C": ("N", "L", "H"),
    "I": ("N", "L", "H"),
    "A": ("N", "L", "H"),
}

CVSS_FIELD_ORDER = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")

CVSS_METRIC_NAMES = {
    "AV": "Attack Vector",
    "AC": "Attack Complexity",
    "PR": "Privileges Required",
    "UI": "User Interaction",
    "S": "Scope",
    "C": "Confidentiality",
    "I": "Integrity",
    "A": "Availability",
}

# No space after each metric's colon (e.g. "AV:N", not "AV: N") - the plain CVSS spec
# notation, and what the seed_group TSVs already store their ground-truth labels in.
CVSS_TEMPLATE = "CVSS:3.1/AV:{AV}/AC:{AC}/PR:{PR}/UI:{UI}/S:{S}/C:{C}/I:{I}/A:{A}"

CVSS_FRAGMENTS = {
    "kind": "vector",
    "template": CVSS_TEMPLATE,
    "fields": CVSS_METRICS,
    "field_order": CVSS_FIELD_ORDER,
}


def format_example(fragments: dict) -> str:
    return fragments["template"].format(**{field: "_" for field in fragments["field_order"]})


def format_metric_options(fragments: dict) -> str:
    return "\n".join(
        f"- {CVSS_METRIC_NAMES[field]} ({field}): {', '.join(fragments['fields'][field])}"
        for field in fragments["field_order"]
    )


CVSS_SYSTEM_PROMPT = (
    "Analyze the CVE description and output the CVSS v3.1 Base vector string. "
    "Do not explain your reasoning. Output only the vector string and nothing else.\n"
    "Valid options for each metric:\n"
    "- Attack Vector (AV): N, A, L, P\n"
    "- Attack Complexity (AC): L, H\n"
    "- Privileges Required (PR): N, L, H\n"
    "- User Interaction (UI): N, R\n"
    "- Scope (S): U, C\n"
    "- Confidentiality (C): N, L, H\n"
    "- Integrity (I): N, L, H\n"
    "- Availability (A): N, L, H\n"
    "Output format (exactly this, no other text): "
    "CVSS:3.1/AV:_/AC:_/PR:_/UI:_/S:_/C:_/I:_/A:_"
)



# Prepended to the CVE text in the user message, so the model sees "CVE Description: <text>"
# rather than the bare text - matches the format the labeling prompt was designed around.
CVSS_USER_PROMPT_PREFIX = "CVE Description: "

# Generic aliases so run_labeling.py can load any domain's prompt_step_2.py the same way
# run_generation.py loads prompt_step_1.py's SYSTEM_PROMPT / STEP_1_PROMPT_TEMPLATE.
SYSTEM_PROMPT = CVSS_SYSTEM_PROMPT
LABEL_FRAGMENTS = CVSS_FRAGMENTS
USER_PROMPT_PREFIX = CVSS_USER_PROMPT_PREFIX
