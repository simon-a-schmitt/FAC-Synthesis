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
    "Analyze each CVE description and determine the CVSS v3.1 base metric values "
    "(AV, AC, PR, UI, S, C, I, A). End your response with the final CVSS v3.1 "
    "vector string, e.g. CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H."
)


def build_cti_vsp_icl_prompt(
    examples: List[dict], query_prompt: str, instruction: str = CTI_VSP_INSTRUCTION_PROMPT
) -> str:
    """Build a few-shot CTI-VSP prompt.

    Each CTI-VSP record's `prompt` is already a fully-formed instruction + CVE
    description, so examples are rendered by appending their gt CVSS vector
    string directly after it, mirroring `build_pubmedqa_icl_prompt`.
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
