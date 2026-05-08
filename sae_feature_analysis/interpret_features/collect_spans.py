import os
import bisect
import json
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


def resolve_runtime_device(requested_device):
    requested_device = requested_device.lower()
    if requested_device == "cuda" and not tc.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested_device


def atomic_write_json(path, payload):
    root = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_state_", suffix=".json", dir=root, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_collectors_from_tsv(path, collectors):
    if not os.path.exists(path):
        return 0

    loaded = 0
    with open(path, "r", encoding="utf8") as f:
        header = f.readline()
        if not header:
            return 0

        for line in f:
            parts = line.rstrip("\n").split("\t", 3)
            if len(parts) != 4:
                continue
            try:
                neuron = int(parts[0])
                text_idx = int(parts[1])
                score = float(parts[2])
            except ValueError:
                continue
            if neuron < 0 or neuron >= len(collectors):
                continue
            span = parts[3].replace("\\n", "\n").replace("\\t", "\t")
            collectors[neuron].update(score, (neuron, text_idx, score, span))
            loaded += 1
    return loaded


def count_assigned_samples(num_rows, subgroup, ttlgroup, upper_exclusive=None):
    if upper_exclusive is None:
        upper_exclusive = num_rows
    upper_exclusive = max(0, min(upper_exclusive, num_rows))
    if upper_exclusive <= subgroup:
        return 0
    return ((upper_exclusive - 1 - subgroup) // ttlgroup) + 1

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
    progress_path = os.path.join(root, f"textspans_group{subgroup}.progress.json")
    checkpoint_every = getattr(args, "checkpoint_every", 500)
    resume_mode = getattr(args, "resume", "auto")
    max_prompts_per_run = max(0, int(getattr(args, "max_prompts_per_run", 0)))

    collectors = [TopKCollector(max_collects) for _ in range(65536)]

    progress = None
    if resume_mode in {"auto", "require"} and os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf8") as f:
            progress = json.load(f)
    elif resume_mode == "require":
        raise FileNotFoundError(
            f"resume='require' but no progress file found: {progress_path}"
        )

    loaded_rows = 0
    if resume_mode in {"auto", "require"} and os.path.exists(out_path):
        loaded_rows = load_collectors_from_tsv(out_path, collectors)

    start_idx = 0
    if progress is not None:
        same_setup = (
            progress.get("data_path") == args.data_path
            and float(progress.get("threshold", args.threshold)) == float(args.threshold)
            and int(progress.get("subgroup", subgroup)) == subgroup
            and int(progress.get("ttlgroup", ttlgroup)) == ttlgroup
        )
        if same_setup:
            start_idx = int(progress.get("last_idx", -1)) + 1
            start_idx = max(0, start_idx)
            if progress.get("completed", False):
                print(f"[INFO] Worker {subgroup}/{ttlgroup} already completed. No further work.")
                return
        else:
            print("[WARN] Existing progress file does not match current setup; restarting from scratch.")
            start_idx = 0

    if resume_mode == "never":
        start_idx = 0
        loaded_rows = 0
        collectors = [TopKCollector(max_collects) for _ in range(65536)]

    if loaded_rows > 0:
        print(f"[INFO] Restored {loaded_rows} rows from existing snapshot: {out_path}")
    if start_idx > 0:
        print(f"[INFO] Resuming from global sample index {start_idx}")

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

    def write_progress(last_idx, processed_assigned, completed):
        payload = {
            "version": 1,
            "data_path": args.data_path,
            "threshold": float(args.threshold),
            "subgroup": int(subgroup),
            "ttlgroup": int(ttlgroup),
            "last_idx": int(last_idx),
            "processed_assigned": int(processed_assigned),
            "max_collects": int(max_collects),
            "completed": bool(completed),
        }
        atomic_write_json(progress_path, payload)

    total_rows = len(corpus)
    assigned_total = count_assigned_samples(total_rows, subgroup, ttlgroup)
    already_done_assigned = count_assigned_samples(total_rows, subgroup, ttlgroup, upper_exclusive=start_idx)
    bar = tqdm.tqdm(total=assigned_total, desc=f"Collecting ({model_name})")
    if already_done_assigned > 0:
        bar.update(already_done_assigned)

    processed_this_run = 0
    processed_assigned = already_done_assigned
    last_idx = start_idx - 1
    stopped_by_budget = False

    if start_idx >= total_rows:
        write_snapshot(out_path)
        write_progress(total_rows - 1, processed_assigned, completed=True)
        return

    for idx, text in enumerate(corpus):
        if idx < start_idx:
            continue
        if idx % ttlgroup != subgroup:
            continue

        bar.update(1)
        processed_assigned += 1
        last_idx = idx

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
        
        results = None
        if not messages or len(messages) == 0:
            print(f"[WARN] Empty message skipped at sample {idx}")
        else:
            try:
                results = activations(messages, generator, sae, tokenizer)
            except Exception as e:
                print(f"[WARN] Template error at sample {idx}: {e}")

        if results is not None:
            for neuron, span, score in zip(results["Neurons"], results["Spans"], results["Scores"]):
                collectors[neuron].update(score, (neuron, idx, score, span))

        processed_this_run += 1

        if processed_this_run % checkpoint_every == 0:
            write_snapshot(out_path)
            write_progress(last_idx, processed_assigned, completed=False)

        if max_prompts_per_run > 0 and processed_this_run >= max_prompts_per_run:
            stopped_by_budget = True
            break

    write_snapshot(out_path)
    completed = (not stopped_by_budget) and (last_idx >= total_rows - 1)
    write_progress(last_idx, processed_assigned, completed=completed)
    if stopped_by_budget:
        print(
            f"[INFO] Run budget reached ({max_prompts_per_run} prompts for worker {subgroup}/{ttlgroup}). "
            "Checkpoint written; rerun to continue."
        )


if __name__ == "__main__":
    """
    Usage (SLURM-optimized):
        python collect_spans.py --device cuda --model-key llama --subgroup 0 --ttlgroup 4 \
                                 --data-path DATA.txt --threshold 0.0 --sae-path /path/to/sae.pth

    Legacy positional args still supported:
        python collect_spans.py cuda llama 0 1 --data-path DATA.txt --threshold 0.0
    """
    import argparse
    import sys
    import logging
    from datetime import datetime

    # Setup logging
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Collect top-k text spans per neuron via SAE hooks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Support both positional args (legacy) and named args (new, SLURM-friendly)
    parser.add_argument("device", nargs="?", type=str, help="[LEGACY] Execution device: 'cpu' or 'cuda'")
    parser.add_argument("model_key", nargs="?", type=str, help="[LEGACY] Model key: mistral|llama|qwen")
    parser.add_argument("subgroup", nargs="?", type=int, help="[LEGACY] Worker ID (0..TOTAL_GROUP-1)")
    parser.add_argument("ttlgroup", nargs="?", type=int, help="[LEGACY] Total workers")
    
    # Named arguments (preferred)
    parser.add_argument("--device", type=str, dest="device_named", default=None,
                        help="Execution device: 'cpu' or 'cuda' (overrides positional)")
    parser.add_argument("--model-key", type=str, dest="model_key_named", choices=["mistral", "llama", "qwen"],
                        help="Backbone model (overrides positional)")
    parser.add_argument("--subgroup", type=int, dest="subgroup_named", default=None,
                        help="Worker ID (overrides positional)")
    parser.add_argument("--ttlgroup", type=int, dest="ttlgroup_named", default=None,
                        help="Total workers (overrides positional)")
    
    # Required arguments
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to input dataset file")
    parser.add_argument("--threshold", type=float, required=True,
                        help="Activation threshold (e.g., 0, 0.5, 1.0, 1.5, 2.0, 4.0)")
    
    # Optional arguments
    parser.add_argument("--sae-path", type=str, default=None,
                        help="SAE checkpoint path (defaults to load_pretrained)")
    parser.add_argument("--sae-layer", type=int, default=None,
                        help="Override SAE layer ID")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Checkpoint every N samples")
    parser.add_argument("--resume", type=str, default="auto", choices=["auto", "never", "require"],
                        help="Resume mode: auto (default), never (start fresh), require (fail if no checkpoint)")
    parser.add_argument("--max-prompts-per-run", type=int, default=0,
                        help="Maximum prompts per worker in this run (0 = no limit)")
    parser.add_argument("--enable-slurm", action="store_true",
                        help="Enable SLURM integration logging")
    
    args = parser.parse_args()

    # Resolve device from named or positional args
    if args.device_named:
        device = args.device_named
    elif args.device:
        device = args.device
    else:
        raise ValueError("--device is required (or use positional arg)")

    # Resolve model_key
    if args.model_key_named:
        model_key = args.model_key_named
    elif args.model_key:
        model_key = args.model_key
    else:
        raise ValueError("--model-key is required (or use positional arg)")

    # Resolve subgroup
    if args.subgroup_named is not None:
        subgroup = args.subgroup_named
    elif args.subgroup is not None:
        subgroup = args.subgroup
    else:
        raise ValueError("--subgroup is required (or use positional arg)")

    # Resolve ttlgroup
    if args.ttlgroup_named is not None:
        ttlgroup = args.ttlgroup_named
    elif args.ttlgroup is not None:
        ttlgroup = args.ttlgroup
    else:
        raise ValueError("--ttlgroup is required (or use positional arg)")

    # Validate
    device = device.lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'")

    device = resolve_runtime_device(device)

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if subgroup >= ttlgroup or subgroup < 0:
        raise ValueError(f"subgroup ({subgroup}) must be in [0, {ttlgroup-1}]")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be > 0")
    if args.max_prompts_per_run < 0:
        raise ValueError("--max-prompts-per-run must be >= 0")

    logger.info(f"{'='*60}")
    logger.info(f"SAE Feature Extraction (collect_spans.py)")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_key}")
    logger.info(f"Worker: {subgroup}/{ttlgroup}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Resume mode: {args.resume}")
    logger.info(f"Checkpoint every: {args.checkpoint_every} samples")
    logger.info(f"Max prompts per run: {args.max_prompts_per_run} (0 = unlimited)")
    if args.enable_slurm:
        logger.info(f"SLURM Integration: Enabled")
        logger.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<auto>')}")
        logger.info(f"  SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', '<none>')}")
        logger.info(f"  SLURM_PROCID: {os.environ.get('SLURM_PROCID', '<none>')}")
    logger.info(f"{'='*60}")

    try:
        # Load SAE
        sae_source = args.sae_path if args.sae_path is not None else model_key
        logger.info(f"Loading SAE from: {sae_source}")
        name, layer, sae = load_pretrained(sae_source, device=device)

        if args.sae_layer is not None:
            logger.info(f"Overriding layer: {layer} -> {args.sae_layer}")
            layer = args.sae_layer

        # Load corpus & model
        model_map = {
            "mistral": "mistral-7b",
            "llama": "llama3-8b",
            "qwen": "qwen2.5-7b"
        }
        if model_key not in model_map:
            raise ValueError(f"Model must be one of {list(model_map.keys())}")

        model_ckpt = model_map[model_key]
        logger.info(f"Loading corpus from: {args.data_path}")
        corpus = CorpusSearchIndex(
            args.data_path,
            cache_freq=1000,
            sampling=None,
        )
        
        dtype = "float32" if device == "cpu" else "bfloat16"
        logger.info(f"Loading model '{model_ckpt}' in dtype={dtype}")
        generator = Generator(model_ckpt, device=device, dtype=dtype)
        tokenizer = generator._tokenizer

        logger.info(f"Mounting SAE to model layer {layer}")
        mount_function(generator._model, model_key, int(layer), sae)

        logger.info(f"Starting feature collection (worker {subgroup}/{ttlgroup})...")
        with tc.no_grad():
            collect_text_spans(corpus, sae, generator, tokenizer, model_key, subgroup, ttlgroup, max_collects=1000)

        logger.info(f"{'='*60}")
        logger.info(f"Completed successfully!")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"{'='*60}")
        logger.error(f"Fatal error in worker {subgroup}/{ttlgroup}:")
        logger.error(f"{type(e).__name__}: {e}")
        logger.error(f"{'='*60}")
        raise
