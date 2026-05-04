import os
import bisect
import tempfile
import tqdm
import torch as tc
import numpy as np
from corpus import CorpusSearchIndex
from llm_surgery import switch_mode, mount_function
from generator import Generator
from autoencoder import load_pretrained

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
CACHE_DIR = "./"

class TopKCollector:
    def __init__(self, topK=3):
        self.TopK = topK
        self.items = []
        self.vals = []

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for _ in self.items:
            yield _

    def _insert(self, val, item):
        idx = bisect.bisect_left(self.vals, val)
        self.vals.insert(idx, val)
        self.items.insert(idx, item)

    def _remove(self):
        self.vals.pop(0)
        self.items.pop(0)

    def update(self, val, item):
        if len(self) < self.TopK:
            self._insert(val, item)
        elif val > self.vals[0]:
            self._remove()
            self._insert(val, item)

IDX = tc.arange(65536)

def activations(messages, model, sae, tokenizer, size=32, shift=31):
    """
    Compute neuron activations for one conversation.
    'messages' is a list of {"role": ..., "content": ...}.
    """
    switch_mode(sae, "train")
    topk, sae.topk = sae.topk, 65536

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(model._device)

    try:
        with tc.no_grad():
            model.get_activates(ids)
    except RuntimeError:
        pass

    ids = ids[shift:]
    
    device = model._device
    IDX_dev = IDX.to(device)
    act, pos = sae.actvs.squeeze()[shift:].max(dim=0)
    choose = act > args.threshold
    idx = IDX_dev[choose].tolist()
    act = act[choose].float().cpu().tolist()
    pos = pos[choose].cpu().tolist()

    spans = [ids[max(0, p - size):p] for p in pos]
    spans = tokenizer.batch_decode(spans)
    sae.topk = topk
    return {"Neurons": idx, "Spans": spans, "Scores": act}

def collect_text_spans(corpus, sae, generator, tokenizer, model_name, subgroup, ttlgroup, max_collects):
    sae.eval()
    sae.MaskTopK = False
    generator._model.eval()
    switch_mode(sae, "train")
    sae.early_stop = True
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    root = f"./xxx/threshold_{args.threshold}"
    os.makedirs(root, exist_ok=True)
    out_path = os.path.join(root, f"textspans_group{subgroup}.tsv")
    checkpoint_every = getattr(args, "checkpoint_every", 25)

    def write_snapshot(path):
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_textspans_", suffix=".tsv", dir=root, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf8") as f:
                f.write("NeuronID\tTextID\tScore\tSpan\n")
                for c in collectors:
                    for neuron, idx, score, span in c:
                        span = span.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "")
                        f.write(f"{neuron}\t{idx}\t{score:.8f}\t{span}\n")
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    collectors = [TopKCollector(max_collects) for _ in range(65536)]
    bar = tqdm.tqdm(total=len(corpus), desc=f"Collecting ({model_name})")

    for idx, text in enumerate(corpus):
        bar.update(1)
        if idx % ttlgroup != subgroup:
            continue

        text = text.replace("\\n", "\n").replace("\\t", "\t")
        if "Human:" in text or "Assistant:" in text:
            messages = []
            current_role = None
            for line in text.split("\n"):
                if line.startswith("Human:"):
                    current_role = "user"
                    content = line[len("Human:"):].strip()
                    messages.append({"role": current_role, "content": content})
                elif line.startswith("Assistant:"):
                    current_role = "assistant"
                    content = line[len("Assistant:"):].strip()
                    messages.append({"role": current_role, "content": content})
                elif current_role is not None and line.strip():
                    messages[-1]["content"] += " " + line.strip()
        else:
            messages = [{"role": "user", "content": text}]
        
        if not messages or len(messages) == 0:
            print(f"[WARN] Empty message skipped at sample {idx}")
            continue
        try:
            results = activations(messages, generator, sae, tokenizer)
        except Exception as e:
            print(f"[WARN] Template error at sample {idx}: {e}")
            continue
        for neuron, span, score in zip(results["Neurons"], results["Spans"], results["Scores"]):
            collectors[neuron].update(score, (neuron, idx, score, span))

        if (idx + 1) % checkpoint_every == 0:
            write_snapshot(out_path)

    write_snapshot(out_path)


if __name__ == "__main__":
    """
    Usage:
            python collect_spans.py <DEVICE> <MODEL_KEY> <SUBGROUP> <TOTAL_GROUP> [--sae-path PATH_OR_ALIAS] [--sae-layer LAYER_ID]

        DEVICE can be "cpu" or "cuda".
    """
    import argparse

    parser = argparse.ArgumentParser(description="Collect top-k text spans per neuron via SAE hooks.")
    parser.add_argument("device", type=str, help="Execution device, either 'cpu' or 'cuda'")
    parser.add_argument("model_key", type=str, choices=["mistral", "llama", "qwen"], help="Backbone model key")
    parser.add_argument("subgroup", type=int, help="This worker's subgroup id (0..TOTAL_GROUP-1)")
    parser.add_argument("ttlgroup", type=int, help="Total number of groups")
    parser.add_argument("--sae-path", type=str, default=None,
                        help="Path or alias for SAE weights to load. If not set, fallback to load_pretrained(model_key).")
    parser.add_argument("--sae-layer", type=int, default=None,
                        help="Override the target layer to mount SAE on (if your checkpoint/meta doesn't encode it).")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the input dataset file (e.g., TD_train_1.tsv)")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Activation threshold (e.g., 0, 0.5, 1.0, 1.5, 2.0, 4.0)")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Write a snapshot TSV every N processed samples")
    
    args = parser.parse_args()

    args.device = args.device.lower()
    if args.device not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'")

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    model_key = args.model_key
    subgroup = args.subgroup
    ttlgroup = args.ttlgroup

    sae_source = args.sae_path if args.sae_path is not None else model_key
    print(f"[SAE] Loading from: {sae_source}")
    name, layer, sae = load_pretrained(sae_source, device=args.device)

    if args.sae_layer is not None:
        print(f"[SAE] Overriding layer: {layer} -> {args.sae_layer}")
        layer = args.sae_layer

    model_map = {
        "mistral": "mistral-7b",
        "llama": "llama3-8b",
        "qwen": "qwen2.5-7b"
    }
    if model_key not in model_map:
        raise ValueError(f"Model must be one of {list(model_map.keys())}")

    model_ckpt = model_map[model_key]

    corpus = CorpusSearchIndex(
        args.data_path,
        cache_freq=1000,
        sampling=None,
    )
    dtype = "float32" if args.device == "cpu" else "bfloat16"
    generator = Generator(model_ckpt, device=args.device, dtype=dtype)
    tokenizer = generator._tokenizer

    print(f"[HOOK] Mounting SAE to model='{model_key}', layer={layer}")
    mount_function(generator._model, model_key, int(layer), sae)

    with tc.no_grad():
        collect_text_spans(corpus, sae, generator, tokenizer, model_key, subgroup, ttlgroup, max_collects=1000)

   
