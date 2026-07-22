import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.model_wrapper import LocalModel
from ma_synthesis.feature_guided.prompt_step0 import STEP_0_PROMPT_TEMPLATE
from ma_synthesis.common.prompt_variables import SYSTEM_PROMPT, TASK_DESCRIPTION


STEP_0_MAX_NEW_TOKENS = 500
STEP_0_TEMPERATURE = 0.9

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _parse_step0_output(raw: str) -> tuple[dict, bool]:
    """Parse the step-0 JSON output.

    Returns (parsed, accept). `accept` is False when the output is not
    parseable JSON, is the explicit failure object, or lacks a non-empty
    "abstract" — any of which means step 0 could not build an abstract
    from the given feature pair.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return {"error": "unparseable", "reason": "no JSON object found"}, False

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"error": "unparseable", "reason": "invalid JSON"}, False

    if "error" in data:
        return data, False

    abstract = str(data.get("abstract", "")).strip()
    if not abstract:
        return data, False

    style_directives = data.get("style_directives", [])
    if not isinstance(style_directives, list):
        style_directives = []

    return {"abstract": abstract, "style_directives": style_directives}, True


def run_step0(
    model: LocalModel,
    feature_1: dict,
    feature_2: dict,
) -> tuple[dict, bool, dict]:
    """Run step 0 of the feature-guided path.

    Renders the two sampled features (each a pool entry from
    feature_pool.load_feature_pool, carrying "top_tokens" and
    "reference_spans") into the abstract-writing prompt, generates, and
    parses the result. Returns (parsed_output, accept, usage); on
    rejection parsed_output holds the raw failure/parse-error info
    instead of an abstract.
    """
    prompt = _render_template(
        STEP_0_PROMPT_TEMPLATE,
        TASK_BESCHREIBUNG=TASK_DESCRIPTION,
        SEED_EXAMPLE_1_TOP_TOKENS=feature_1["top_tokens"],
        SEED_EXAMPLE_1_REFERENCE_SPANS=feature_1["reference_spans"],
        SEED_EXAMPLE_2_TOP_TOKENS=feature_2["top_tokens"],
        SEED_EXAMPLE_2_REFERENCE_SPANS=feature_2["reference_spans"],
    )

    output, usage = model.generate(
        prompt=prompt,
        max_new_tokens=STEP_0_MAX_NEW_TOKENS,
        do_sample=STEP_0_TEMPERATURE > 0,
        temperature=STEP_0_TEMPERATURE,
        system=SYSTEM_PROMPT,
        return_usage=True,
    )
    output = output.strip()

    parsed, accept = _parse_step0_output(output)
    return parsed, accept, usage
