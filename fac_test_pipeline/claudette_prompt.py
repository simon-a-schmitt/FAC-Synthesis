"""
Shared CLAUDETTE-TOS-classification prompt-construction logic for the
CLAUDETTE-related pipelines under fac_test_pipeline/:

    feature_coverage_filter/claudette_tos/run_feature_coverage_filter.py

All of them run the same fixed CLAUDETTE-TOS unfair-clause-classification
instruction as a SYSTEM turn, followed by a single Terms-of-Service sentence
as a USER turn, through Llama-3.1-8B-Instruct + the SAE hook - matching the
genuine two-turn chat structure used by the actual CLAUDETTE-TOS benchmark
pipeline (benchmark_play_ground/run_claudette_benchmark.py, "plain" mode), so
prompt structure AND wording stay consistent between the two.

CLAUDETTE_SYSTEM_PROMPT below is a LITERAL, BYTE-FOR-BYTE COPY of
CLAUDETTE_SYSTEM_PROMPT defined in
benchmark_play_ground/run_claudette_benchmark.py (lines 28-50). It is
duplicated rather than imported so fac_test_pipeline/ stays self-contained.
If that constant changes, this one must be updated to match.

Unlike the CVSS pipelines (see cvss_prompt.py), the CLAUDETTE user turn has
no fixed opening phrase preceding the content: the user turn *is* the ToS
sentence verbatim, with nothing prepended. So there is no instruction
preamble to skip past when restricting feature-activation stats to content
tokens - the entire user turn is already content, and callers should pass
opening_phrase=None (the default) to
run_fac_test_pipeline_feature_stats.compute_feature_stats_for_prompt().
"""

# NOTE: keep byte-for-byte in sync with CLAUDETTE_SYSTEM_PROMPT in
# benchmark_play_ground/run_claudette_benchmark.py:28-50.
CLAUDETTE_SYSTEM_PROMPT = (
    "You are analyzing a single sentence from the Terms of Service of an\n"
    "online platform under EU consumer law (Directive 93/13/EEC).\n"
    "\n"
    "Decide, for each of the following eight clause types, whether the\n"
    "sentence contains a potentially unfair clause of that type:\n"
    "\n"
    "  LTD  limitation of liability\n"
    "  TER  unilateral termination\n"
    "  CH   unilateral change\n"
    "  CR   content removal\n"
    "  USE  contract by using\n"
    "  LAW  choice of law\n"
    "  J    jurisdiction\n"
    "  ARB  arbitration\n"
    "\n"
    "Most sentences contain no unfair clause of any type.\n"
    "\n"
    "Answer with exactly one line in the following format, using Y or N for\n"
    "each type, and nothing else:\n"
    "\n"
    "LTD:?|TER:?|CH:?|CR:?|USE:?|LAW:?|J:?|ARB:?"
)


def build_claudette_user_content(sentence: str) -> str:
    """Build the user-turn content for a ToS sentence: the sentence verbatim,
    with the same literal "\\n" -> newline escape-sequence unescaping applied
    by benchmark_play_ground/data_loader.py:load_claudette_tsv(), byte-identical
    to the query_prompt used by benchmark_play_ground/run_claudette_benchmark.py
    ("plain" mode) for the same sentence text.
    """
    return sentence.replace("\\n", "\n")
