import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch as tc
import tqdm


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAE_PRETRAIN_DIR = os.path.join(ROOT_DIR, "sae_pretrain")
if SAE_PRETRAIN_DIR not in sys.path:
    sys.path.insert(0, SAE_PRETRAIN_DIR)

from autoencoder import TopKSAE  # noqa: E402
from generator_uni import UnifiedGenerator  # noqa: E402
from llm_surgery_uni import mount_function, switch_mode  # noqa: E402


tc.manual_seed(42)


class Collector:
    def __init__(self, layer: int):
        self.layer = layer
        switch_mode(self, "monitor")
        self.early_stop = False
        self.cache = None

    def monitor(self, x):
        self.cache = x


def _default_cache_dir() -> str:
    return os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))


def resolve_sae_checkpoint(
    local_path: Optional[str],
) -> str:
    if local_path:
        local_path = os.path.abspath(local_path)
        if os.path.exists(local_path):
            return local_path
        raise FileNotFoundError(f"Local SAE checkpoint not found: {local_path}")

    raise ValueError("--sae-ckpt-path is required. Online SAE download is disabled in this pipeline.")


def load_prompts(prompts_json: str) -> List[str]:
    with open(prompts_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    prompts: List[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict):
                prompt = item.get("prompt")
                if isinstance(prompt, str):
                    prompts.append(prompt)
    elif isinstance(payload, dict):
        items = payload.get("prompts", [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict):
                    prompt = item.get("prompt")
                    if isinstance(prompt, str):
                        prompts.append(prompt)

    if len(prompts) == 0:
        raise ValueError("No prompts found in prompts JSON. Expected list[str] or list[{\"prompt\": ...}].")
    return prompts


def extract_latent_for_prompt(prompt: str, model: UnifiedGenerator, collector: Collector, sae: TopKSAE) -> tc.Tensor:
    collector.cache = None
    try:
        model.get_activates(prompt)
    except RuntimeError:
        # Expected when collector.early_stop is true.
        pass

    if collector.cache is None:
        raise RuntimeError("Collector cache is empty. Hook may not be mounted correctly.")

    hidden = collector.cache.to(tc.float32)
    last_token_hidden = hidden[:, -1, :]
    latent = sae.encode(last_token_hidden).detach().cpu().squeeze(0)
    return latent


def write_jsonl(records: List[Dict[str, Any]], output_jsonl: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_jsonl))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FAC test pipeline scaffold for prompt -> Llama activations -> SAE latent vectors.")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    parser.add_argument("--model-name", type=str, required=True, help="Local path to model directory")
    parser.add_argument("--layer", type=int, default=16, help="1-based decoder layer index for activation extraction.")

    parser.add_argument("--sae-ckpt-path", type=str, default="", help="Local path to SAE checkpoint .pth")

    parser.add_argument("--prompts-json", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all prompts")

    parser.add_argument("--hf-cache-dir", type=str, default=_default_cache_dir())
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # This pipeline is intentionally strict local-only. Runtime network downloads are disabled.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Keep cache dir explicit so all loaders resolve to the same local location.
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir
    os.makedirs(args.hf_cache_dir, exist_ok=True)

    if args.device == "cuda":
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    model_path = os.path.abspath(args.model_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Local model directory not found: {model_path}. "
            "Provide a local model snapshot path via --model-name."
        )

    prompts = load_prompts(args.prompts_json)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    sae_ckpt = resolve_sae_checkpoint(
        local_path=args.sae_ckpt_path or None,
    )

    model = UnifiedGenerator(
        model_path,
        device=args.device,
        dtype=args.dtype,
        cache_dir=args.hf_cache_dir,
        local_files_only=True,
        strict_local_paths=True,
    )
    collector = Collector(args.layer)
    mount_function(model._model, "llama", args.layer, collector)
    collector.early_stop = True

    sae = TopKSAE.from_disk(sae_ckpt, device=args.device)
    sae.eval()

    records: List[Dict[str, Any]] = []
    with tc.no_grad():
        for idx, prompt in enumerate(tqdm.tqdm(prompts, desc="Processing prompts")):
            latent = extract_latent_for_prompt(prompt, model, collector, sae)
            records.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "layer": args.layer,
                    "latent_dim": int(latent.shape[-1]),
                    "non_zero": int((latent != 0).sum().item()),
                    "latent": latent.tolist(),
                }
            )

    write_jsonl(records, args.output_jsonl)
    print(f"Wrote {len(records)} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
