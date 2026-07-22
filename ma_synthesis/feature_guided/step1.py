import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.model_wrapper import LocalModel
from ma_synthesis.feature_guided.prompt_step1 import STEP_1_PROMPT_TEMPLATE
from ma_synthesis.common.prompt_variables import SYSTEM_PROMPT, TASK_DESCRIPTION


STEP_1_MAX_NEW_TOKENS = 300
STEP_1_TEMPERATURE = 0.9
STEP_1_TOP_P = 0.9
STEP_1_NO_REPEAT_NGRAM_SIZE = 3
STEP_1_REPETITION_PENALTY = 1.1

STRUCTURAL_REQUIREMENTS = (
    "The context must end with a citation to a specific court decision in standard reporter format (the case may be invented), immediately followed by the placeholder in parentheses: (<HOLDING>). The placeholder stands for the cited case's holding, never for the case name or citation itself."
)


def _render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _format_style_directives(style_directives: list[str]) -> str:
    if not style_directives:
        return "(none)"
    return "\n".join(f"- {directive}" for directive in style_directives)


def run_step1(
    model: LocalModel,
    content_abstract: str,
    style_directives: list[str],
) -> tuple[str, dict]:
    prompt = _render_template(
        STEP_1_PROMPT_TEMPLATE,
        TASK_BESCHREIBUNG=TASK_DESCRIPTION,
        CONTENT_ABSTRACT=content_abstract,
        STYLE_DIRECTIVES=_format_style_directives(style_directives),
        STRUCTURAL_REQUIREMENTS=STRUCTURAL_REQUIREMENTS,
    )
    output, usage = model.generate(
        prompt=prompt,
        max_new_tokens=STEP_1_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=STEP_1_TEMPERATURE,
        top_p=STEP_1_TOP_P,
        no_repeat_ngram_size=STEP_1_NO_REPEAT_NGRAM_SIZE,
        repetition_penalty=STEP_1_REPETITION_PENALTY,
        system=SYSTEM_PROMPT,
        return_usage=True,
    )
    return output.strip(), usage
