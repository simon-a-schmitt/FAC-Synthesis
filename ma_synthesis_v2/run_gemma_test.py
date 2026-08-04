"""Minimal example call against the local Gemma model.

Loads the model via `UnifiedGenerator` (see benchmark_play_ground/generator_uni.py)
and prints:
  - the exact token ids / decoded text that are fed into the model
    (i.e. after the chat template + tokenizer have run), and
  - the exact raw token ids / decoded text that come back out of the model
    (before any post-processing), plus the cleaned final text.
"""

import os
import sys

import torch as tc

WS_PATH = os.environ.get(
    "WS_PATH", "/pfs/work9/workspace/scratch/ka_ai3967-master_thesis_exp"
)
DEFAULT_MODEL_PATH = os.path.join(WS_PATH, "models", "gemma-4-31B-it")
DEFAULT_HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.join(WS_PATH, "hf_cache"))

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # FAC-Synthesis
BENCHMARK_PLAY_GROUND_DIR = os.path.join(ROOT_DIR, "benchmark_play_ground")
if BENCHMARK_PLAY_GROUND_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_PLAY_GROUND_DIR)

from generator_uni import UnifiedGenerator  # noqa: E402


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model_path = os.environ.get("GEMMA_MODEL_PATH", DEFAULT_MODEL_PATH)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Local Gemma model directory not found: {model_path}. "
            "Set GEMMA_MODEL_PATH to override."
        )

    print(f"Loading model from: {model_path}")
    gen = UnifiedGenerator(
        model_path,
        device="cuda",
        dtype="bfloat16",
        cache_dir=DEFAULT_HF_CACHE_DIR,
        local_files_only=True,
        strict_local_paths=True,
    )

    system_text = "You are a helpful assistant."
    user_text = "What are black holes?"
    messages = gen.build_messages(user_text=user_text, system_text=system_text)

    print_header("MESSAGES (chat format, before templating)")
    for m in messages:
        print(m)

    # Mirrors the gemma branch of UnifiedGenerator.generate(), but keeps every
    # intermediate value around so we can print it for debugging.
    inputs = gen._processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = {
        k: (v.to(gen._device) if isinstance(v, tc.Tensor) else v)
        for k, v in inputs.items()
    }
    input_ids = inputs["input_ids"]

    print_header("EXACT INPUT SENT TO GEMMA (after chat template + tokenization)")
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"input_ids: {input_ids.tolist()}")
    if "attention_mask" in inputs:
        print(f"attention_mask: {inputs['attention_mask'].tolist()}")
    print("\nDecoded input (including all special/template tokens):")
    print(gen.tokenizer.decode(input_ids[0], skip_special_tokens=False))

    gen_kwargs = dict(max_new_tokens=128, do_sample=False)

    with tc.no_grad():
        outputs = gen._model.generate(
            **inputs, **gen_kwargs, pad_token_id=gen.tokenizer.pad_token_id
        )

    gen_tokens = outputs[:, input_ids.shape[1] :]
    raw_output = gen.tokenizer.batch_decode(gen_tokens, skip_special_tokens=False)[0]
    cleaned_output = gen._decode_gemma_response(raw_output, prefix=input_ids[0])

    print_header("EXACT RAW OUTPUT FROM GEMMA (before any post-processing)")
    print(f"output token ids: {gen_tokens.tolist()}")
    print("\nRaw decoded output (including all special tokens):")
    print(raw_output)

    print_header("CLEANED / FINAL OUTPUT TEXT")
    print(cleaned_output)


if __name__ == "__main__":
    main()
