import sys
import os
import tqdm
import torch as tc
import numpy as np
import transformers as trf
from corpus import CorpusSearchIndex
from autoencoder import TopKSAE
from llm_surgery_uni import switch_mode, mount_function
from datautils import GroupActvDataset
from generator_uni import UnifiedGenerator as Generator

CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", "/scratch/huggingface")
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
print("HF Cache Dir =", os.environ["TRANSFORMERS_CACHE"])

trf.set_seed(42)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


class LinearRater:
    def __init__(self, init, final, total):
        self.init = init
        self.final = final
        self.total = max(1, total)  # avoid division by zero
        self.diff = abs(final - init)
        self.trend = 1.0 if final > init else -1.0
        self.steps = 0

    def get_value(self):
        progress = self.diff * self.steps / self.total
        value = self.init + self.trend * progress
        if self.trend > 0:
            return min(self.final, value)
        return max(self.final, value)

    def step(self, new_step=1):
        self.steps += new_step
        return self.get_value()


def testify(sae, model):
    sae.eval()
    switch_mode(sae, "generate")
    text = model.generate("Who is the president of the United States?", max_new_tokens=64, do_sample=False)
    sae.train()
    switch_mode(sae, "train")
    return text


def train_SAE(dataset, sae, generator, layer, hidden_size,
              batch_size=512, learn_rate=1e-3, epochs=5, betas=(0.9, 0.999)):

    os.makedirs("outputs", exist_ok=True)

    optimizer = tc.optim.Adam(sae.parameters(), lr=learn_rate, betas=betas,
                              eps=6.25e-10, weight_decay=0.0)
    scaler = tc.cuda.amp.GradScaler(enabled=True)
    steps_total = max(1, int(len(dataset) / batch_size * 0.5))
    rater1 = LinearRater(init=200, final=20, total=steps_total)
    sae.topk = int(rater1.step())
    sae.alpha = 0.0
    skips = (1, 0)
    freq = max(1000, batch_size)

    for epoch in range(1, 1 + epochs):
        print("Epoch=%d" % epoch, "-->", testify(sae, generator))
        bar = tqdm.tqdm(total=len(dataset), desc="Epoch=%d" % epoch)
        ttls, l0s, l1s, l2s = [], [], [], []
        dataset.shuffle(epoch)

        if epoch >= skips[0]:
            for actv in dataset.get_data():
                bar.update(1)
                if bar.n <= skips[1]:
                    continue

                seq_len = actv.shape[-2]
                if "llama" in generator._name.lower():
                    max_tokens = min(seq_len, 286)
                elif "mistral" in generator._name.lower():
                    max_tokens = min(seq_len, 256)
                elif "qwen" in generator._name.lower():
                    max_tokens = min(seq_len, 256)
                else:
                    max_tokens = min(seq_len, 256)

                x = actv[..., :max_tokens, :]
                x = x.reshape(-1, x.shape[-1])

                with tc.autocast(device_type="cuda", dtype=tc.bfloat16, enabled=True):
                    ttl, l0, l1, l2 = sae.compute_loss(x.to("cuda", non_blocking=True))
                scaler.scale(ttl).backward()

                ttls.append(ttl.item())
                l0s.append(l0.item())
                l1s.append(l1.item())
                l2s.append(l2.item())

                if bar.n % batch_size == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    sae.topk = int(rater1.step())

                if bar.n % freq == 0:
                    print("\nEpoch=%d | Step=%d | Total=%.4f | L0=%.4f | L1=%.4f | L2=%.4f" % (
                        epoch, bar.n,
                        np.mean(ttls[-freq:]),
                        np.mean(l0s[-freq:]),
                        np.mean(l1s[-freq:]),
                        np.mean(l2s[-freq:])
                    ))
                    ttls.clear(); l0s.clear(); l1s.clear(); l2s.clear()

                if bar.n % (25 * batch_size) == 0:
                    save_path = f"outputs/TopK7_l{layer}_h{hidden_size}_epoch{epoch}.pth"
                    sae.dump_disk(save_path)
                    print(f"Saved checkpoint: {save_path}")
                    print("Epoch=%d | Step=%d" % (epoch, bar.n), "-->", testify(sae, generator))
        bar.close()
    return sae

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python train_SAEs.py <gpu_id> <layer> <model_name>")
        sys.exit(1)

    gpu_id = sys.argv[1]
    layer = int(sys.argv[2])
    model_name = sys.argv[3]

    corpus = GroupActvDataset(f"/xxx/prompt_actvs_l{layer}/", layerID=None)

    generator = Generator(model_name, device="cuda")
    hidden_size = generator._model.config.hidden_size
    print("Hidden size:", hidden_size)
    print(generator.generate("Who is the president of the United States?", max_new_tokens=64, do_sample=False))

    # init SAE
    sae = TopKSAE(hidden_size, 2**16, topK=20, device="cuda")
    sae.disabled = False
    sae.logging = False

    if "llama" in model_name.lower():
        model_family = "llama"
    elif "mistral" in model_name.lower():
        model_family = "mistral"
    elif "qwen" in model_name.lower():
        model_family = "qwen"
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # mount hook for SAE monitoring/training
    mount_function(generator._model, model_family, layer, sae)

    train_SAE(corpus, sae, generator, layer, hidden_size)

