import os
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_play_ground.model_wrapper import LocalModel
from ma_synthesis_v2.blackbox.prompt_step1 import SYSTEM_PROMPT, STEP_1_PROMPT_TEMPLATE


STEP_1_MAX_NEW_TOKENS = 220
STEP_1_TEMPERATURE = 0.9
STEP_1_TOP_P = 0.9
STEP_1_NO_REPEAT_NGRAM_SIZE = 3
STEP_1_REPETITION_PENALTY = 1.1

CVE_DESCRIPTION_MARKER = "CVE Description:"

# `benchmark_play_ground.model_wrapper` pulls in `sae_pretrain.generator_uni`,
# which calls `transformers.set_seed(42)` at import time, fixing the global
# `random` module to the same sequence on every process start. Use a private
# RNG seeded from OS entropy so seed/vector sampling actually varies across
# separate script invocations.
_rng = random.Random(os.urandom(16))


def _render_template(template: str, **values: str) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def load_seeds(path: Path) -> list[dict]:
    """Load seed CVE examples from the benchmark TSV.

    Each line is `<full analyst prompt>\\t<CVSS vector>`; the description is
    recovered from the tail of the first column, after the
    "CVE Description:" marker.
    """
    seeds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            prompt_field, _, vector = line.rpartition("\t")
            marker_idx = prompt_field.rfind(CVE_DESCRIPTION_MARKER)
            if marker_idx == -1:
                continue
            description = prompt_field[marker_idx + len(CVE_DESCRIPTION_MARKER):].strip()
            seeds.append({"description": description, "vector": vector.strip()})
    return seeds


def load_gt_pool(path: Path) -> list[str]:
    """Load the pool of target CVSS vectors, one per line."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_step1(model: LocalModel, seeds: list[dict], gt_pool: list[str]) -> tuple[str, dict, dict]:
    """Run step 1 of the blackbox arm.

    Draws one random seed example (description + its true CVSS vector) and
    one random target vector from the ground-truth pool, renders the prompt
    template, and generates a new CVE description whose correct CVSS vector
    is the target vector. Returns the generated text, the sampling metadata
    (seed index, seed description/vector, target vector), and the token
    usage ({"input_tokens", "output_tokens"}) for this generation call so the
    caller can log it.
    """
    seed_index = _rng.randrange(len(seeds))
    seed = seeds[seed_index]
    target_vector = _rng.choice(gt_pool)

    prompt = _render_template(
        STEP_1_PROMPT_TEMPLATE,
        SEED_DESCRIPTION=seed["description"],
        SEED_VECTOR=seed["vector"],
        TARGET_VECTOR=target_vector,
    )

    full_prompt = model.render_prompt(prompt, system=SYSTEM_PROMPT)
    print("\n" + "=" * 60, flush=True)
    print("[Step 1] Full prompt sent to model (incl. special tokens):", flush=True)
    print(full_prompt, flush=True)
    print("=" * 60 + "\n", flush=True)

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
        "seed_index": seed_index,
        "seed_description": seed["description"],
        "seed_vector": seed["vector"],
        "target_vector": target_vector,
    }
    return output, meta, usage
