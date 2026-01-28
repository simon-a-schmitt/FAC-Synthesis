import json
import argparse
import os
import re
import copy
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Optional
from llama_wrapper import Llama3Wrapper
from utils.helpers import get_a_b_probs
from steering_settings import SteeringSettings
from behaviors import (
    get_ab_test_data,
    get_system_prompt,
    get_steering_vector,
    get_results_dir,
    ALL_BEHAVIORS,
)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def swap_a_b_options(text: str) -> str:
    pattern = r"(?s)(.*?)\s*\(A\)\s*(.*?)\s*\(B\)\s*(.*)"
    match = re.search(pattern, text)
    
    if match:
        preamble = match.group(1)
        content_a = match.group(2)
        content_b = match.group(3)
        return f"{preamble}\n (A) {content_b}\n (B) {content_a}"
    else:
        return text

def process_item_ab(item: Dict[str, str], model: Llama3Wrapper, system_prompt: Optional[str], a_token_id: int, b_token_id: int) -> Dict[str, str]:
    question = item["question"]
    ans_match = item["answer_matching_behavior"]
    ans_not = item["answer_not_matching_behavior"]
    
    logits = model.get_logits_from_text(user_input=question, model_output="(", system_prompt=system_prompt)
    a_prob, b_prob = get_a_b_probs(logits, a_token_id, b_token_id)
    
    pred = "(A)" if a_prob > b_prob else "(B)"
    correct = 1 if pred == ans_match else 0
    
    return {
        "question": question,
        "answer_matching_behavior": ans_match,
        "answer_not_matching_behavior": ans_not,
        "pred": pred,
        "a_prob": float(a_prob),
        "b_prob": float(b_prob),
        "correct": correct,
    }

def test_steering(layers: List[int], multipliers: List[int], settings: SteeringSettings, overwrite=False):
    save_dir = get_results_dir(settings.behavior)
    os.makedirs(save_dir, exist_ok=True)
    
    model_name_for_vector = "xxx/llama-3.1-8b-instruct"
    model_weights_path = settings.override_model_weights_path or ""
    
    print(f"Loading model from: {model_weights_path}")
    model = Llama3Wrapper(model_path=model_weights_path, hf_token=HF_TOKEN, use_chat=not settings.use_base_model)

    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")

    test_data = get_ab_test_data(settings.behavior)

    for layer in layers:
        vec_layer = settings.override_vector if settings.override_vector is not None else layer
        vector = get_steering_vector(settings.behavior, vec_layer, model_name_for_vector, normalized=True).to(model.device)

        for multiplier in multipliers:
            save_suffix = settings.make_result_save_suffix(layer=layer, multiplier=multiplier)
            save_path = os.path.join(save_dir, f"results_diagnosis_{save_suffix}.json")
            
            if os.path.exists(save_path) and not overwrite:
                print("Found existing", save_path, "- skipping")
                continue

            print(f"\n=== Testing Layer {layer} | Multiplier {multiplier} ===")
            
            model.reset_all()
            model.set_add_activations(layer, multiplier * vector)
            sys_prompt = None if settings.system_prompt == "none" else get_system_prompt(settings.behavior, settings.system_prompt)

            results = []
            
            count_robust_success = 0
            count_positional_bias = 0
            count_broken = 0
            count_simple_correct = 0

            for item in tqdm(test_data, desc="Running Swap Test"):
                res_orig = process_item_ab(item, model, sys_prompt, a_token_id, b_token_id)
                if res_orig["correct"] == 1:
                    count_simple_correct += 1
                q_swapped = swap_a_b_options(item["question"])
                
                orig_match_label = item["answer_matching_behavior"]
                swapped_match_label = "(B)" if "(A)" in orig_match_label else "(A)"
                swapped_not_match_label = "(A)" if "(B)" in swapped_match_label else "(B)"
                
                item_swapped = {
                    "question": q_swapped,
                    "answer_matching_behavior": swapped_match_label,
                    "answer_not_matching_behavior": swapped_not_match_label
                }
                
                res_swapped = process_item_ab(item_swapped, model, sys_prompt, a_token_id, b_token_id)
                is_robust = (res_orig["correct"] == 1) and (res_swapped["correct"] == 1)
                is_positional = (res_orig["pred"] == res_swapped["pred"])
                if is_robust:
                    count_robust_success += 1
                elif is_positional:
                    count_positional_bias += 1
                else:
                    count_broken += 1

                results.append({
                    "original": res_orig,
                    "swapped": res_swapped,
                    "analysis": "robust_success" if is_robust else ("positional_bias" if is_positional else "broken")
                })

            total = len(test_data)
            acc_robust = count_robust_success / total
            rate_bias = count_positional_bias / total
            acc_simple = count_simple_correct / total

            print(f"\n>>> DIAGNOSIS REPORT: Layer {layer} * {multiplier}")
            print(f"Total Samples: {total}")
            print(f"Robust Semantic Accuracy: {acc_robust:.2%}")
            print(f"Positional Bias Rate:     {rate_bias:.2%}")
            print(f"Broken/Inconsistent:      {count_broken / total:.2%}")
            print(f"Standard Accuracy:        {acc_simple:.2%} (Old Metric)")
            print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--behaviors", type=str, nargs="+", default=ALL_BEHAVIORS)
    parser.add_argument("--system_prompt", type=str, choices=["pos", "neg", "none"], default="none")
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    settings = SteeringSettings()
    settings.type = "ab"
    settings.system_prompt = args.system_prompt
    settings.override_vector = args.override_vector
    settings.override_model_weights_path = args.override_model_weights_path
    settings.model_size = "8b"
    settings.use_base_model = False

    for behavior in args.behaviors:
        settings.behavior = behavior
        test_steering(
            layers=args.layers,
            multipliers=args.multipliers,
            settings=settings,
            overwrite=args.overwrite,
        )
