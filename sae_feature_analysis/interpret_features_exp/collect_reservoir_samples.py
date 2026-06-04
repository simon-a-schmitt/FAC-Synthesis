import base64
import json
import os
import pickle
import random
import sys
import tempfile

import torch as tc
import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "interpret_features"))
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)

from corpus import CorpusSearchIndex
from generator import Generator
from llm_surgery import mount_function, switch_mode
from autoencoder import load_pretrained


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


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


def encode_rng_state(rng):
    return base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii")


def decode_rng_state(payload):
    rng = random.Random()
    rng.setstate(pickle.loads(base64.b64decode(payload.encode("ascii"))))
    return rng


def _special_token_id_set(tokenizer):
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    special_ids.update(range(128000, 128256))
    return special_ids


def _is_newline_only_token(token):
    stripped = token.replace("Ċ", "").replace("\n", "")
    return len(stripped) == 0 and len(token) > 0


def _find_user_content_start(token_ids, tokens, tokenizer):
    marker_tokens = ("<|start_header_id|>", "user", "<|end_header_id|>")
    marker_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in marker_tokens]

    user_header_end = -1
    if all(isinstance(tok_id, int) and tok_id >= 0 for tok_id in marker_ids):
        marker_len = len(marker_ids)
        for idx in range(0, len(token_ids) - marker_len + 1):
            if list(token_ids[idx : idx + marker_len]) == marker_ids:
                user_header_end = idx + marker_len
                break

    if user_header_end < 0:
        marker_len = len(marker_tokens)
        for idx in range(0, len(tokens) - marker_len + 1):
            if tuple(tokens[idx : idx + marker_len]) == marker_tokens:
                user_header_end = idx + marker_len
                break

    if user_header_end < 0:
        return 0

    content_start = user_header_end
    while content_start < len(tokens) and _is_newline_only_token(tokens[content_start]):
        content_start += 1
    return content_start


def normalize_messages(text):
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    if "Human:" in text or "Assistant:" in text:
        messages = []
        current_role = None
        for line in text.split("\n"):
            if line.startswith("Human:"):
                current_role = "user"
                messages.append({"role": current_role, "content": line[len("Human:"):].strip()})
            elif line.startswith("Assistant:"):
                current_role = "assistant"
                messages.append({"role": current_role, "content": line[len("Assistant:"):].strip()})
            elif current_role is not None and line.strip():
                messages[-1]["content"] += " " + line.strip()
        return messages
    return [{"role": "user", "content": text}]


class ReservoirSampler:
    def __init__(self, capacity, seed, items=None, seen=0, rng_state=None):
        self.capacity = int(capacity)
        self.items = list(items) if items is not None else []
        self.seen = int(seen)
        self.rng = decode_rng_state(rng_state) if rng_state is not None else random.Random(int(seed))

    def update(self, item):
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        choice = self.rng.randrange(self.seen)
        if choice < self.capacity:
            self.items[choice] = item


def compute_samples(messages, model, sae, tokenizer, span_size=32):
    switch_mode(sae, "train")
    original_topk, sae.topk = sae.topk, 65536

    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    if isinstance(enc, dict):
        input_ids = enc["input_ids"]
    else:
        input_ids = enc

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    ids = input_ids[0].to(model._device)
    token_ids = ids.tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    special_ids = _special_token_id_set(tokenizer)
    content_start = _find_user_content_start(token_ids, tokens, tokenizer)
    token_mask = tc.tensor(
        [idx >= content_start and token_id not in special_ids for idx, token_id in enumerate(token_ids)],
        device=ids.device,
    )

    try:
        with tc.no_grad():
            model.get_activates(ids)
    except RuntimeError:
        pass

    token_actvs = sae.actvs.squeeze()
    masked_actvs = token_actvs * token_mask.to(token_actvs.dtype).unsqueeze(-1)
    active_pairs = tc.nonzero(masked_actvs > 0, as_tuple=False)

    samples = []
    if active_pairs.numel() > 0:
        for token_idx, feature_idx in active_pairs.tolist():
            score = float(masked_actvs[token_idx, feature_idx].float().item())
            span_ids = ids[max(0, token_idx - span_size + 1): token_idx + 1]
            span = tokenizer.batch_decode([span_ids])[0]
            samples.append(
                {
                    "feature_id": int(feature_idx),
                    "score": score,
                    "token_idx": int(token_idx),
                    "token": tokens[token_idx],
                    "span": span,
                }
            )

    sae.topk = original_topk
    return samples


def escape_tsv(value):
    return str(value).replace("\r", "").replace("\n", "\\n").replace("\t", "\\t")


def write_snapshot(path, sampler):
    root = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_reservoir_", suffix=".tsv", dir=root, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf8") as f:
            f.write("FeatureID\tScore\tDocID\tTokenIndex\tToken\tSpan\n")
            for sample in sampler.items:
                f.write(
                    f"{sample['feature_id']}\t{sample['score']:.8f}\t{sample['doc_id']}\t"
                    f"{sample['token_idx']}\t{escape_tsv(sample['token'])}\t{escape_tsv(sample['span'])}\n"
                )
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_progress(path, sampler, last_idx, completed, args, seed):
    payload = {
        "version": 1,
        "data_path": args.data_path,
        "last_idx": int(last_idx),
        "max_samples": int(args.max_samples),
        "completed": bool(completed),
        "sample_count": int(len(sampler.items)),
        "seed": int(seed),
        "seen": int(sampler.seen),
        "rng_state": encode_rng_state(sampler.rng),
        "samples": sampler.items,
    }
    atomic_write_json(path, payload)


def load_progress(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def collect_reservoir_samples(corpus, sae, generator, tokenizer, args, seed):
    sae.eval()
    sae.MaskTopK = False
    generator._model.eval()
    switch_mode(sae, "train")
    sae.early_stop = True

    root = os.path.join(SCRIPT_DIR, "xxx", "reservoir_samples")
    os.makedirs(root, exist_ok=True)
    out_path = os.path.join(root, "reservoir_samples.tsv")
    progress_path = os.path.join(root, "reservoir_samples.checkpoint.json")
    checkpoint_every = int(args.checkpoint_every)

    progress = None
    if args.resume in {"auto", "require"}:
        progress = load_progress(progress_path)
        if progress is None and args.resume == "require":
            raise FileNotFoundError(f"resume='require' but no progress file found: {progress_path}")

    if progress is not None:
        same_setup = (
            progress.get("data_path") == args.data_path
            and int(progress.get("max_samples", args.max_samples)) == int(args.max_samples)
            and int(progress.get("seed", seed)) == int(seed)
        )
        if not same_setup:
            print("[WARN] Existing progress file does not match current setup; restarting from scratch.")
            progress = None

    if progress is None or args.resume == "never":
        sampler = ReservoirSampler(args.max_samples, seed)
        start_idx = 0
    else:
        sampler = ReservoirSampler(
            args.max_samples,
            progress.get("seed", seed),
            items=progress.get("samples", []),
            seen=progress.get("seen", 0),
            rng_state=progress.get("rng_state"),
        )
        start_idx = max(0, int(progress.get("last_idx", -1)) + 1)
        if progress.get("completed", False):
            print("[INFO] Existing reservoir run already completed. Writing snapshot and returning.")
            write_snapshot(out_path, sampler)
            return

    total_rows = len(corpus)
    if start_idx >= total_rows:
        write_snapshot(out_path, sampler)
        write_progress(progress_path, sampler, total_rows - 1, True, args, seed)
        return

    bar = tqdm.tqdm(total=total_rows, desc="Reservoir sampling")
    if start_idx > 0:
        bar.update(start_idx)

    last_idx = start_idx - 1
    processed_this_run = 0
    stopped_by_budget = False

    for idx, text in enumerate(corpus):
        if idx < start_idx:
            continue

        bar.update(1)
        last_idx = idx
        processed_this_run += 1

        messages = normalize_messages(text)
        if not messages:
            print(f"[WARN] Empty message skipped at sample {idx}")
        else:
            try:
                samples = compute_samples(messages, generator, sae, tokenizer)
                for sample in samples:
                    sample["doc_id"] = int(idx)
                    sampler.update(sample)
            except Exception as exc:
                print(f"[WARN] Template error at sample {idx}: {exc}")

        if processed_this_run % checkpoint_every == 0:
            write_snapshot(out_path, sampler)
            write_progress(progress_path, sampler, last_idx, completed=False, args=args, seed=seed)

        if args.max_docs_per_run > 0 and processed_this_run >= args.max_docs_per_run:
            stopped_by_budget = True
            break

    write_snapshot(out_path, sampler)
    completed = (not stopped_by_budget) and (last_idx >= total_rows - 1)
    write_progress(progress_path, sampler, last_idx, completed=completed, args=args, seed=seed)
    if stopped_by_budget:
        print(
            f"[INFO] Run budget reached ({args.max_docs_per_run} docs). Checkpoint written; rerun to continue."
        )


if __name__ == "__main__":
    import argparse
    import logging

    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Reservoir sample feature activations over fineweb_edu.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("device", nargs="?", type=str, help="[LEGACY] Execution device: 'cpu' or 'cuda'")
    parser.add_argument("model_key", nargs="?", type=str, help="[LEGACY] Model key: mistral|llama|qwen")

    parser.add_argument("--device", type=str, dest="device_named", default=None,
                        help="Execution device: 'cpu' or 'cuda' (overrides positional)")
    parser.add_argument("--model-key", type=str, dest="model_key_named", choices=["mistral", "llama", "qwen"],
                        help="Backbone model (overrides positional)")
    parser.add_argument("--data-path", type=str, required=True, help="Path to fineweb_edu text file")

    parser.add_argument("--sae-path", type=str, default=None, help="SAE checkpoint path")
    parser.add_argument("--sae-layer", type=int, default=None, help="Override SAE layer ID")
    parser.add_argument("--max-samples", type=int, default=1000, help="Reservoir size")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Checkpoint every N documents")
    parser.add_argument("--resume", type=str, default="auto", choices=["auto", "never", "require"],
                        help="Resume mode: auto, never, or require")
    parser.add_argument("--max-docs-per-run", type=int, default=0,
                        help="Maximum number of documents to process in this run (0 = unlimited)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Reservoir sampling seed. If omitted, a random seed is generated.")
    parser.add_argument("--enable-slurm", action="store_true", help="Enable SLURM integration logging")

    args = parser.parse_args()

    if args.device_named:
        device = args.device_named
    elif args.device:
        device = args.device
    else:
        raise ValueError("--device is required (or use positional arg)")

    if args.model_key_named:
        model_key = args.model_key_named
    elif args.model_key:
        model_key = args.model_key
    else:
        raise ValueError("--model-key is required (or use positional arg)")

    device = device.lower()
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'")

    device = resolve_runtime_device(device)
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be > 0")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be > 0")
    if args.max_docs_per_run < 0:
        raise ValueError("--max-docs-per-run must be >= 0")

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**32 - 1)
    random.seed(seed)
    tc.manual_seed(seed)

    logger.info(f"Reservoir seed: {seed}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_key}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Reservoir size: {args.max_samples}")
    logger.info(f"Resume mode: {args.resume}")
    logger.info(f"Checkpoint every: {args.checkpoint_every} documents")

    if args.enable_slurm:
        logger.info("SLURM Integration: Enabled")
        logger.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<auto>')}")
        logger.info(f"  SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', '<none>')}")
        logger.info(f"  SLURM_PROCID: {os.environ.get('SLURM_PROCID', '<none>')}")

    try:
        sae_source = args.sae_path if args.sae_path is not None else model_key
        logger.info(f"Loading SAE from: {sae_source}")
        name, layer, sae = load_pretrained(sae_source, device=device)

        if args.sae_layer is not None:
            logger.info(f"Overriding layer: {layer} -> {args.sae_layer}")
            layer = args.sae_layer

        model_map = {
            "mistral": "mistral-7b",
            "llama": "llama3-8b",
            "qwen": "qwen2.5-7b",
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

        logger.info("Starting reservoir sampling over fineweb_edu...")
        with tc.no_grad():
            collect_reservoir_samples(corpus, sae, generator, tokenizer, args, seed)

        logger.info("Completed successfully!")

    except Exception as exc:
        logger.error("Fatal error during reservoir sampling:")
        logger.error(f"{type(exc).__name__}: {exc}")
        raise