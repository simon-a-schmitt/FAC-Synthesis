import re
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.model_wrapper import LocalModel

from ma_synthesis.common.prompt_variables import (
    SYSTEM_PROMPT,
    TASK_DESCRIPTION,
    FORMAT_INSTRUCTION,
    TARGET_REFERENCE,
    DISTRACTOR_INSTRUCTION,
    INTRODUCTION_CLAUSE,
)
from ma_synthesis.common.prompt_step2 import (
    STEP_2_PROMPT_TEMPLATE,
    STEP_2_MAX_NEW_TOKENS,
    STEP_2_TEMPERATURE,
)
from ma_synthesis.common.prompt_step3 import (
    STEP_3_PROMPT_TEMPLATE,
    STEP_3_MAX_NEW_TOKENS,
    STEP_3_TEMPERATURE,
)




def _render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _generate(
    model: LocalModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    return model.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        system=SYSTEM_PROMPT,
    ).strip()


def _parse_distractors(raw: str) -> list[str]:
    """Extract up to 4 distractor lines from model output.

    Strips leading numbering (e.g. '1.', '2)') if present; falls back to
    any non-empty line so plain-list outputs are handled too.
    """
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Remove optional leading number/bullet: '1.', '1)', '-', '*'
        cleaned = re.sub(r"^\s*(\d+[.)]\s*|[-*]\s*)", "", stripped).strip()
        if cleaned:
            lines.append(cleaned)
    return lines[:4]


def run_pipeline(model: LocalModel, step1_output: str) -> dict:
    """Run steps 2–4 on the citing context produced by step 1.

    Returns a dict with keys step_1 … step_4 plus the assembled final record.
    """
    outputs: dict = {"step_1": step1_output}

    # --- Step 2: generate correct holding ---
    step2_prompt = _render_template(
        STEP_2_PROMPT_TEMPLATE,
        TASK_DESCRIPTION=TASK_DESCRIPTION,
        CITING_CONTEXT=step1_output,
        FORMAT_INSTRUCTION=FORMAT_INSTRUCTION,
        TARGET_REFERENCE=TARGET_REFERENCE
    )

    print(step2_prompt)

    step2_output = _generate(
        model, step2_prompt, STEP_2_MAX_NEW_TOKENS, STEP_2_TEMPERATURE
    )
    outputs["step_2"] = step2_output

    # --- Step 3: generate 4 distractor holdings ---
    step3_prompt = _render_template(
        STEP_3_PROMPT_TEMPLATE,
        TASK_DESCRIPTION=TASK_DESCRIPTION,
        GENERATED_CONTEXT=step1_output,
        TRUE_LABEL=step2_output,
        DISTRACTOR_INSTRUCTION=DISTRACTOR_INSTRUCTION
    )

    print(step3_prompt)

    step3_output = _generate(
        model, step3_prompt, STEP_3_MAX_NEW_TOKENS, STEP_3_TEMPERATURE
    )
    outputs["step_3"] = step3_output

    # --- Step 4: deterministic assembly ---
    distractors = _parse_distractors(step3_output)

    options = [step2_output] + distractors  # index 0 is always the correct one before shuffle
    random.shuffle(options)

    correct_letter = chr(ord("A") + options.index(step2_output))

    labeled_options = " ".join(
        f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)
    )
    final_question = f"{INTRODUCTION_CLAUSE}\n\n{step1_output}\n\n{labeled_options}"

    outputs["step_4_options"] = options
    outputs["final_question"] = final_question
    outputs["final_label"] = correct_letter

    print("\n" + "=" * 60, flush=True)
    print("[Step 4 — Final Question]", flush=True)
    print(final_question, flush=True)
    print(f"\n[Correct Answer] {correct_letter}", flush=True)
    print("=" * 60 + "\n", flush=True)

    return outputs
