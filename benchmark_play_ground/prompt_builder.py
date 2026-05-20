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
