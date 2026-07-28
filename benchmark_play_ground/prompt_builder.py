from typing import List

INSTRUCTION_PROMPT = (
    "Your task is to identify the single correct legal holding statement from options A-E to fill the <HOLDING> placeholder in the given context. "
    "Strict Constraint: Output ONLY the corresponding letter (A, B, C, D, or E) of the correct answer."
)


def build_plain_prompt(citing_text: str, options: List[str], instruction: str = INSTRUCTION_PROMPT) -> str:
    parts = [instruction, citing_text]
    for letter, opt in zip("ABCDE", options):
        parts.append(f"{letter}. {opt}")
    return "\n".join(parts)


def _strip_instruction(prompt: str, instruction: str) -> str:
    prompt = prompt.strip()
    instruction = instruction.strip()
    if prompt.startswith(instruction):
        remainder = prompt[len(instruction):].lstrip("\n\r \t")
        return remainder.strip()
    return prompt


def build_icl_prompt(examples: List[dict], query_prompt: str, instruction: str = INSTRUCTION_PROMPT) -> str:
    sections = [
        "# Instruction",
        instruction,
        "",
    ]

    for idx, ex in enumerate(examples, start=1):
        context = _strip_instruction(ex.get("prompt", ""), instruction)
        label = ex.get("label", ex.get("gt", ""))
        sections.extend(
            [
                f"## Example {idx}",
                f"Context: {context}",
                f"Answer: {label}",
                "",
                "---",
                "",
            ]
        )

    sections.extend(
        [
            "## Target Task",
            f"Context: {_strip_instruction(query_prompt, instruction)}",
            "Answer:",
        ]
    )
    return "\n".join(sections)


PUBMEDQA_INSTRUCTION_PROMPT = (
    "Answer each biomedical research question with exactly one word: yes, no, or maybe."
)


def build_pubmedqa_icl_prompt(examples: List[dict], query_prompt: str, instruction: str = PUBMEDQA_INSTRUCTION_PROMPT) -> str:
    """Build a few-shot PubMedQA prompt.

    Each PubMedQA record's `prompt` already ends in "...\\nAnswer:\\n", so examples
    are rendered by simply appending their gt label after that trailing "Answer:".
    """
    sections = [
        "# Instruction",
        instruction,
        "",
    ]

    for idx, ex in enumerate(examples, start=1):
        context = ex.get("prompt", "").rstrip()
        label = ex.get("label", ex.get("gt", ""))
        sections.extend(
            [
                f"## Example {idx}",
                f"{context} {label}",
                "",
                "---",
                "",
            ]
        )

    sections.extend(
        [
            "## Target Task",
            query_prompt.rstrip(),
        ]
    )
    return "\n".join(sections)


CTI_VSP_INSTRUCTION_PROMPT = (
    "Analyze each CVE description and output the CVSS v3.1 Base vector string. Do not explain your reasoning. Output only the vector string and nothing else.\n"
    "\n"
    "Valid options for each metric:\n"
    "- Attack Vector (AV): N, A, L, P\n"
    "- Attack Complexity (AC): L, H\n"
    "- Privileges Required (PR): N, L, H\n"
    "- User Interaction (UI): N, R\n"
    "- Scope (S): U, C\n"
    "- Confidentiality (C): N, L, H\n"
    "- Integrity (I): N, L, H\n"
    "- Availability (A): N, L, H\n"
    "\n"
    "Output format (exactly this, no other text):\n"
    "CVSS:3.1/AV:_/AC:_/PR:_/UI:_/S:_/C:_/I:_/A:_"
)

_CVE_DESCRIPTION_MARKER = "CVE Description:"


def _extract_cve_description_block(prompt: str) -> str:
    """Strip any baked-in instruction preamble, keeping from "CVE Description:" onward.

    Records in this benchmark's TSVs store `prompt` as the full instruction text
    concatenated with the CVE description, so this trims each example (and the
    query) down to just its "CVE Description: ..." block before it is placed
    under a single shared instruction.
    """
    idx = prompt.find(_CVE_DESCRIPTION_MARKER)
    if idx == -1:
        return prompt.strip()
    return prompt[idx:].strip()


def build_cti_vsp_icl_prompt(
    examples: List[dict], query_prompt: str, instruction: str = CTI_VSP_INSTRUCTION_PROMPT
) -> str:
    """Build a few-shot CTI-VSP prompt.

    Renders the instruction once, followed by "CVE Description: ..." / CVSS
    vector pairs for each few-shot example, then the query's CVE description
    for the model to complete.
    """
    sections = [instruction, ""]

    for ex in examples:
        context = _extract_cve_description_block(ex.get("prompt", ""))
        label = ex.get("label", ex.get("gt", ""))
        sections.append(f"{context}\n{label}")
        sections.append("")

    query_context = _extract_cve_description_block(query_prompt)
    sections.append(f"{query_context}\nCVSS:3.1/")
    return "\n".join(sections)
