import os
import sys
import tqdm
import torch as tc
from corpus import CorpusSearchIndex
from generator_uni import UnifiedGenerator
from llm_surgery_uni import mount_function, switch_mode


tc.manual_seed(42)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("/scratch/"))
os.makedirs(CACHE_DIR, exist_ok=True)


def collect_actvs(prompt_text, model, collectors):
    try:
        model.get_activates(prompt_text)
    except RuntimeError:
        pass
    for collector in collectors:
        if collector.cache is not None:
            yield collector.cache.detach().to(tc.float16).cpu()
        else:
            yield None


class Collector:
    def __init__(self, layer):
        self.layer = layer
        switch_mode(self, "monitor")
        self.early_stop = False
        self.cache = None

    def monitor(self, x):
        self.cache = x


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python create_actvs_uni.py <device_id> <group_idx> <jump> <model_name> <layer>")
        sys.exit(1)

    device_id = sys.argv[1]
    group_idx = int(sys.argv[2])
    jump = int(sys.argv[3])
    model_name = sys.argv[4]
    layer = int(sys.argv[5])

    corpus = CorpusSearchIndex("", cache_freq=10000, sampling=None)
    model = UnifiedGenerator(model_name, device="cuda")

    group_size = 256
    layers = [layer]

    group_actvs = []
    collectors = []

    for l in layers:
        out_dir = f"/scratch/sae_input/prompt_actvs_l{l}"
        os.makedirs(out_dir, exist_ok=True)

        c = Collector(l)
        collectors.append(c)
        group_actvs.append([])
        mount_function(model._model, "llama", l, c)
    collectors[-1].early_stop = True

    with tc.no_grad():
        bar = tqdm.tqdm(total=len(corpus))
        for i, text in enumerate(corpus):
            bar.update(1)
            try:
                if i < group_idx * group_size:
                    continue
                fpath = f"/scratch/sae_input/prompt_actvs_l{layers[0]}/group_{group_idx}.pt"
                if os.path.exists(fpath):
                    group_idx += jump
                    continue

                for g_actv, item in zip(group_actvs, collect_actvs(text, model, collectors)):
                    if item is not None:
                        g_actv.append(item)

                if len(group_actvs[0]) >= group_size:
                    for l, g_actv in zip(layers, group_actvs):
                        fpath = f""
                        os.makedirs(os.path.dirname(fpath), exist_ok=True)
                        tc.save(g_actv, fpath)
                        g_actv.clear()
                    group_idx += jump

            except Exception as e:
                print(f"Encountered error at line {i}: {e}")

        if len(group_actvs) > 0:
            for l, g_actv in zip(layers, group_actvs):
                if len(g_actv) > 0:
                    fpath = f"/scratch/sae_input/prompt_actvs_l{l}/group_{group_idx}.pt"
                    os.makedirs(os.path.dirname(fpath), exist_ok=True)
                    tc.save(g_actv, fpath)
                    g_actv.clear()
        print(f"Stopping after processing {bar.n} samples.")

