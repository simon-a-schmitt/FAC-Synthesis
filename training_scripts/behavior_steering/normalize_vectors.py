import os
import argparse
import torch as t
from typing import List
from behaviors import ALL_BEHAVIORS, get_vector_path

def normalize_vectors(model_tag: str, layers: List[int], behaviors: List[str], use_normalized_dir=True):
    for layer in layers:
        norms = {}
        vecs = {}
        new_paths = {}

        for behavior in behaviors:
            vec_path = get_vector_path(behavior, layer, model_tag)
            if not os.path.exists(vec_path):
                print(f"[Warning] Missing vector: {vec_path}")
                continue
            
            vec = t.load(vec_path)
            norm = vec.norm().item()
            vecs[behavior] = vec
            norms[behavior] = norm

            if use_normalized_dir:
                new_path = vec_path.replace("vectors", "normalized_vectors")
            else:
                new_path = vec_path
            new_paths[behavior] = new_path

        if len(norms) == 0:
            print(f"[Warning] No vectors found for layer {layer}, skipping.")
            continue

        mean_norm = t.tensor(list(norms.values())).mean().item()

        for behavior in vecs:
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]

        for behavior in vecs:
            save_path = new_paths[behavior]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            t.save(vecs[behavior], save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default="llama-3.1-8b-instruct")
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)

    args = parser.parse_args()

    normalize_vectors(
        model_tag=args.model_tag,
        layers=args.layers,
        behaviors=args.behaviors
    )
