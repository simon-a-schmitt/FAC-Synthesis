import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.model_wrapper import LocalModel
from ma_synthesis.blackbox.prompt_step1 import STEP_1_PROMPT_TEMPLATE
from ma_synthesis.common.prompt_variables import SYSTEM_PROMPT, TASK_DESCRIPTION, ADDITIONAL_INSTRUCTIONS


STEP_1_MAX_NEW_TOKENS = 2048
STEP_1_TEMPERATURE = 1.0
STEP_1_TOP_P = 0.9




def _render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def run_step1(model: LocalModel, seed_context: str) -> str:
    prompt = _render_template(
        STEP_1_PROMPT_TEMPLATE,
        TASK_DESCRIPTION=TASK_DESCRIPTION,
        SEED_CONTEXT=seed_context,
        ADDITIONAL_INSTRUCTIONS=ADDITIONAL_INSTRUCTIONS,
    )

    print(prompt) 
    
    return model.generate(
        prompt=prompt,
        max_new_tokens=STEP_1_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=STEP_1_TEMPERATURE,
        top_p=STEP_1_TOP_P,
        system=SYSTEM_PROMPT,
    ).strip()
