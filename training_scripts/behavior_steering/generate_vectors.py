import os
import json
import argparse
import random
from typing import List, Dict, Any, Optional
import torch as t
from torch.utils.data import Dataset
from tqdm import tqdm
from llama_wrapper import Llama3Wrapper
from behaviors import (
    get_ab_data_path,
    get_vector_dir,
    get_vector_path,
    get_activations_dir,
    get_activations_path,
    ALL_BEHAVIORS,
)

class ComparisonDataset(Dataset):
    def __init__(self, path: str, tokenizer, n_train: Optional[int] = None, seed: int = 42):
        with open(path, "r") as f:
            full_data: List[Dict[str, Any]] = json.load(f)
        
        if n_train is not None and n_train > 0:
            print(f"[INFO] Seeding (PyTorch) with seed={seed}")
            t.manual_seed(seed)
            indices = t.randperm(len(full_data)).tolist()
            selected_indices = indices[:n_train]
            self.data = [full_data[i] for i in selected_indices]
            print(f"[INFO] Randomly selected {len(self.data)} samples out of {len(full_data)} available.")
        else:
            self.data = full_data
            
        self.tokenizer = tokenizer

    def _pair_to_tensors(self, question: str, answer: str) -> Dict[str, t.Tensor]:
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        enc = self.tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
        return {
            "input_ids": enc["input_ids"].to("cuda"),
            "attention_mask": enc["attention_mask"].to("cuda"),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        q = item["question"]
        p = item["answer_matching_behavior"]
        n = item["answer_not_matching_behavior"]
        return self._pair_to_tensors(q, p), self._pair_to_tensors(q, n)


@t.no_grad()
def generate_save_vectors_for_behavior(
    model: Llama3Wrapper,
    layers: List[int],
    behavior: str,
    save_activations: bool = False,
    model_name_tag: str = "llama-3.1-8b-instruct",
    target: str = "last_nonpad",
    n_train: Optional[int] = None,
    seed: int = 42,
):
    data_path = get_ab_data_path(behavior)
    os.makedirs(get_vector_dir(behavior), exist_ok=True)
    if save_activations:
        os.makedirs(get_activations_dir(behavior), exist_ok=True)

    dataset = ComparisonDataset(data_path, model.tokenizer, n_train=n_train, seed=seed)

    acts_pos: Dict[int, List[t.Tensor]] = {l: [] for l in layers}
    acts_neg: Dict[int, List[t.Tensor]] = {l: [] for l in layers}

    captured: Dict[int, t.Tensor] = {l: None for l in layers}
    target_idx: int = -1

    def make_hook(layer_id: int):
        def hook_fn(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if target_idx is not None and target_idx >= 0:
                captured[layer_id] = hidden[0, target_idx, :].detach().float().cpu()
        return hook_fn

    handles = []
    try:
        for l in layers:
            handles.append(model.model.model.layers[l].register_forward_hook(make_hook(l)))

        for p_tok, n_tok in tqdm(dataset, desc=f"[{behavior}]"):
            if target == "last_nonpad":
                target_idx = int(p_tok["attention_mask"].sum(dim=1).item()) - 1
            elif target == "minus2":
                target_idx = p_tok["input_ids"].size(1) - 2
            else:
                raise ValueError("target must be 'last_nonpad' or 'minus2'")

            captured = {l: None for l in layers}
            _ = model.model(input_ids=p_tok["input_ids"], attention_mask=p_tok["attention_mask"])
            for l in layers:
                if captured[l] is not None:
                    acts_pos[l].append(captured[l])

            if target == "last_nonpad":
                target_idx = int(n_tok["attention_mask"].sum(dim=1).item()) - 1
            elif target == "minus2":
                target_idx = n_tok["input_ids"].size(1) - 2

            captured = {l: None for l in layers}
            _ = model.model(input_ids=n_tok["input_ids"], attention_mask=n_tok["attention_mask"])
            for l in layers:
                if captured[l] is not None:
                    acts_neg[l].append(captured[l])

        for l in layers:
            if len(acts_pos[l]) == 0 or len(acts_neg[l]) == 0:
                print(f"[WARNING] No valid activations collected for layer {l}. Skipping.")
                continue
            pos = t.stack(acts_pos[l])
            neg = t.stack(acts_neg[l])
            vec = (pos - neg).mean(dim=0)
            t.save(vec, get_vector_path(behavior, l, model_name_tag))
            if save_activations:
                t.save(pos, get_activations_path(behavior, l, model_name_tag, "pos"))
                t.save(neg, get_activations_path(behavior, l, model_name_tag, "neg"))

    finally:
        for h in handles:
            h.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--target", type=str, choices=["last_nonpad", "minus2"], default="last_nonpad")
    parser.add_argument("--n_train", type=int, default=None, help="Number of training samples to use (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data selection")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN", None)
    model = Llama3Wrapper(model_path=args.model_path, hf_token=hf_token, use_chat=True)

    for behavior in args.behaviors:
        generate_save_vectors_for_behavior(
            model=model,
            layers=args.layers,
            behavior=behavior,
            save_activations=args.save_activations,
            model_name_tag="llama-3.1-8b-instruct",
            target=args.target,
            n_train=args.n_train,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
