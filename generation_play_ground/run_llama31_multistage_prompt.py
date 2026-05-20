import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.model_wrapper import LocalModel
from generation_play_ground.prompt_templates import STEP_1_PROMPT_TEMPLATE, SYSTEM_PROMPT
from generation_play_ground.prompt_variables import (
    STEP_1_MAX_NEW_TOKENS,
    STEP_1_TASK_EXAMPLE_PROMPT,
    STEP_1_TASK_EXAMPLE_SOLUTION,
    STEP_1_TEMPERATURE,
    STEP_1_WEITERE_ANWEISUNG,
    TARGET_FEATURE_DESCRIPTION,
    TARGET_FEATURE_TEXT_SPANS,
    TARGET_TASK_DESCRIPTION,
)


MODEL_PATH = Path("/pfs/work9/workspace/scratch/ka_ukhtw-master_thesis_experiment/models/llama-3.1-8b")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "bfloat16"


def render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def generate_response(model: LocalModel, task_prompt: str, max_new_tokens: int, temperature: float) -> str:
    return model.generate(
        prompt=task_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        use_cache=False,
        system=SYSTEM_PROMPT,
    ).strip()


def run_pipeline() -> dict[str, str]:
    model = LocalModel(model_path=str(MODEL_PATH), device=DEVICE, dtype=DTYPE)
    outputs: dict[str, str] = {}

    step_1_prompt = render_template(
        STEP_1_PROMPT_TEMPLATE,
        TASK_BESCHREIBUNG=TARGET_TASK_DESCRIPTION,
        FEATURE_BESCHREIBUNG=TARGET_FEATURE_DESCRIPTION,
        TEXT_SPANS=TARGET_FEATURE_TEXT_SPANS,
        TASK_BEISPIEL=f"Prompt:\n{STEP_1_TASK_EXAMPLE_PROMPT}\n\nLösung:\n{STEP_1_TASK_EXAMPLE_SOLUTION}",
        WEITERE_ANWEISUNG=STEP_1_WEITERE_ANWEISUNG,
    )
    step_1_output = generate_response(
        model=model,
        task_prompt=step_1_prompt,
        max_new_tokens=STEP_1_MAX_NEW_TOKENS,
        temperature=STEP_1_TEMPERATURE,
    )
    outputs["step_1"] = step_1_output

    # TODO: Schritt 2 Prompt mit step_1_output aufbauen und ausfuehren.
    # step_2_prompt = ...
    # step_2_output = generate_response(...)
    # outputs["step_2"] = step_2_output

    # TODO: Schritt 3 Prompt mit step_2_output aufbauen und ausfuehren.
    # step_3_prompt = ...
    # step_3_output = generate_response(...)
    # outputs["step_3"] = step_3_output

    # TODO: Schritt 4 Prompt mit step_3_output aufbauen und ausfuehren.
    # step_4_prompt = ...
    # step_4_output = generate_response(...)
    # outputs["step_4"] = step_4_output

    return outputs


def main() -> None:
    outputs = run_pipeline()
    print(outputs["step_1"])


if __name__ == "__main__":
    main()