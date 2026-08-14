"""
Shared CVSS-classification prompt-construction logic for the CVSS-related
pipelines under fac_test_pipeline/:

    feature_coverage/run_feature_coverage.py
    feature_coverage_filter/run_feature_coverage_filter.py
    run_synthetic_feature_activation_check.py

All of them run the same fixed CVSS v3.1 classification instruction as a
SYSTEM turn, followed by a CVE description as a USER turn
("CVE Description: <description>"), through Llama-3.1-8B-Instruct + the SAE
hook - matching the genuine two-turn chat structure used by the actual
CVSS-classification benchmark/fine-tuning pipelines
(benchmark_play_ground/run_cti_vsp_benchmark.py,
benchmarks/cti_vsp/transform_llamafactory_format.py), so all three pipelines
share identical prompt structure AND wording.

CVSS_SYSTEM_PROMPT below is a LITERAL, BYTE-FOR-BYTE COPY of
CTI_VSP_SYSTEM_PROMPT defined in
benchmark_play_ground/run_cti_vsp_benchmark.py (lines 31-45). It is
duplicated rather than imported so fac_test_pipeline/ stays self-contained.
If that constant changes, this one must be updated to match.

Before this two-turn conversion, the CVSS pipelines under fac_test_pipeline/
built a single flat instruction+description string (and did so
inconsistently with each other - see git history). normalize_cvss_prompt()
below still handles reconstructing a description out of that kind of legacy
flattened single-string prompt, but now returns it paired with
CVSS_SYSTEM_PROMPT as a (system, user_content) pair instead of splicing the
instruction back onto the front of one string.
"""

from typing import Optional, Tuple

# NOTE: keep byte-for-byte in sync with CTI_VSP_SYSTEM_PROMPT in
# benchmark_play_ground/run_cti_vsp_benchmark.py:31-45.
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

CVSS_OPENING_PHRASE = "CVE Description:"


def build_cvss_user_content(description: str) -> str:
    """Build the user-turn content for a known-clean CVE description string:
    CVSS_OPENING_PHRASE + " " + description.strip(), byte-identical to the
    user-turn content built by
    benchmarks/cti_vsp/transform_llamafactory_format.py
    ("CVE Description: " + description) for the same description text.
    """
    return f"{CVSS_OPENING_PHRASE} {description.strip()}"


def normalize_cvss_prompt(prompt: str) -> Tuple[Optional[str], str]:
    """Rebuild a legacy flattened single-string CVSS prompt into the
    canonical (system, user_content) pair:

        (CVSS_SYSTEM_PROMPT, "CVE Description: " + description)

    Discards everything up to and including the first occurrence of
    CVSS_OPENING_PHRASE in `prompt` and treats what follows (stripped) as
    the description.

    Falls back to (None, prompt) - i.e. `prompt` returned completely
    unchanged, with NO system turn - if CVSS_OPENING_PHRASE is missing or
    nothing follows it, printing a warning. This is a true no-op: fed to
    compute_feature_stats_for_prompt(user_content, ..., system=None) it
    reproduces exactly the single-flat-user-turn behaviour every caller had
    before this two-turn conversion - the safe fallback for non-CVSS input.
    """
    idx = prompt.find(CVSS_OPENING_PHRASE)
    if idx < 0:
        print(
            f"Warning: opening phrase {CVSS_OPENING_PHRASE!r} not found in prompt; "
            "leaving prompt unnormalized (no system turn)."
        )
        return None, prompt

    description = prompt[idx + len(CVSS_OPENING_PHRASE):].strip()
    if not description:
        print(
            "Warning: empty CVE description after normalizing prompt "
            f"(phrase {CVSS_OPENING_PHRASE!r} found, but nothing followed it); "
            "leaving prompt unnormalized (no system turn)."
        )
        return None, prompt

    return CVSS_SYSTEM_PROMPT, build_cvss_user_content(description)
