import os
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.model_wrapper import LocalModel
from ma_synthesis.blackbox.prompt_step1 import STEP_1_PROMPT_TEMPLATE
from ma_synthesis.common.prompt_variables import (
    SYSTEM_PROMPT,
    TASK_DESCRIPTION,
    STRUCTURAL_SIGNATURE,
    OPERATORS,
)


STEP_1_MAX_NEW_TOKENS = 300
STEP_1_TEMPERATURE = 0.9
STEP_1_TOP_P = 0.9
STEP_1_NO_REPEAT_NGRAM_SIZE = 3
STEP_1_REPETITION_PENALTY = 1.1

# `benchmark_play_ground.model_wrapper` pulls in `sae_pretrain.generator_uni`,
# which calls `transformers.set_seed(42)` at import time, fixing the global
# `random` module to the same sequence on every process start. Use a private
# RNG seeded from OS entropy so seed/operator sampling actually varies across
# separate script invocations.
_rng = random.Random(os.urandom(16))


def _render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def _sample_operator() -> tuple[str, dict]:
    names = list(OPERATORS.keys())
    weights = [OPERATORS[name]["weight"] for name in names]
    name = _rng.choices(names, weights=weights, k=1)[0]
    return name, OPERATORS[name]


def run_step1(model: LocalModel, seeds: list[dict]) -> tuple[str, dict, dict]:
    """Run step 1 of the blackbox path.

    Draws two distinct seed examples and one transformation operator
    (weighted by OPERATORS[...]["weight"]) at random, renders the prompt
    template, and generates the new context. Returns the generated text,
    the sampling metadata (seed indices, operator name), and the token
    usage ({"input_tokens", "output_tokens"}) for this generation call so
    the caller can log it.
    """
    seed_index_1, seed_index_2 = _rng.sample(range(len(seeds)), 2)
    operator_name, operator = _sample_operator()

    prompt = _render_template(
        STEP_1_PROMPT_TEMPLATE,
        TASK_DESCRIPTION=TASK_DESCRIPTION,
        TRANSFORMATION_OPERATOR=operator["description"],
        SEED_EXAMPLE_1=seeds[seed_index_1]["prompt"],
        SEED_EXAMPLE_2=seeds[seed_index_2]["prompt"],
        STRUCTURAL_SIGNATURE=STRUCTURAL_SIGNATURE,
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
    output = output.strip()

    meta = {
        "seed_index_1": seed_index_1,
        "seed_index_2": seed_index_2,
        "transformation_operator": operator_name,
    }
    return output, meta, usage
