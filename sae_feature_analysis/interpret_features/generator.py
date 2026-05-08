import os
import transformers as trf
import torch as tc

trf.set_seed(42)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))


def _resolve_device(device):
    if device == "cuda" and not tc.cuda.is_available():
        print("[LLM] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return device


def _resolve_torch_dtype(dtype):
    if isinstance(dtype, tc.dtype):
        return dtype
    dtype_map = {
        "float32": tc.float32,
        "float16": tc.float16,
        "bfloat16": tc.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {dtype}; choose from {list(dtype_map.keys())}")
    return dtype_map[dtype]


def _resolve_local_model_path(model_key: str) -> str:
    if model_key == "llama3-8b":
        ws_path = os.environ.get("WS_PATH")
        if ws_path:
            local_path = os.path.join(ws_path, "models", "llama-3.1-8b")
            if os.path.isdir(local_path):
                return local_path

        local_path = os.environ.get("LLAMA_MODEL_PATH")
        if local_path and os.path.isdir(local_path):
            return local_path

        raise FileNotFoundError(
            "Local Llama model not found. Set WS_PATH or LLAMA_MODEL_PATH to a valid local snapshot."
        )

    return MODEL_MAP[model_key]["hf_name"]


MODEL_MAP = {
    "llama3-8b": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "mistral-7b": {
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.2",
    },
    "qwen-7b": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
    },
}


instruct = (
    "You are a neuron interpreter for neural networks. Each neuron looks for one particular concept/topic/theme/behavior/pattern. "
    "Look at some words the neuron activates for and summarize in a single concept/topic/theme/behavior/pattern what the neuron is looking for. "
    "The words close to the beginning are more correlated to the hidden concept/topic/theme/behavior/pattern. "
    "Don't list examples of words and keep your summary as concise as possible. "
    "If you cannot summarize more than half of the given words within one clear concept/topic/theme/behavior/pattern, you should say ``Cannot Tell.``."
)

examples = [
    ("January, terday, cember, April, July, September, December, Thursday, quished, November, Tuesday.", "Dates."),
    ("B., M., e., R., C., OK., A., H., D., S., J., al., p., T., N., W., G., a.C., or, St., K., a.m., L..", "Abbreviations and acronyms."),
    ("actual, literal, real, Real, optical, Physical, REAL, virtual, visual.", "Perception of reality."),
    ("Go, Python, C++, Java, c#, python3, cuda, java, javascript, basic.", "Programming languages."),
    ("1950, 1980, 1985, 1958, 1850, 1980, 1960, 1940, 1984, 1948.", "Years."),
]


# ========== Generator Wrapper ==========
class Generator:
    def __init__(self, model_key="mistral-7b", device="cuda", dtype="bfloat16"):
        if model_key not in MODEL_MAP:
            raise ValueError(f"Unsupported model key {model_key}, choose from {list(MODEL_MAP.keys())}")
        self._name = _resolve_local_model_path(model_key)
        self._device = _resolve_device(device)
        self._dtype = _resolve_torch_dtype(dtype)
        self.build_model()

    @tc.no_grad()
    def build_model(self):
        print(f"Initializing LLM: {self._name}")
        maps = "cpu" if self._device == "cpu" else "auto"
        load_kwargs = {"local_files_only": True} if os.path.isdir(self._name) else {}
        self._tokenizer = trf.AutoTokenizer.from_pretrained(
            self._name,
            use_fast=True,
            padding_side="right",
            cache_dir=CACHE_DIR,
            **load_kwargs,
        )
        self._model = trf.AutoModelForCausalLM.from_pretrained(
            self._name,
            cache_dir=CACHE_DIR,
            torch_dtype=self._dtype,
            device_map=maps,
            **load_kwargs,
        )
        if not self._tokenizer.eos_token:
            self._tokenizer.eos_token = "</s>"
        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.eos_token_id
        self._inp_emb = self._model.get_input_embeddings()
        self._out_emb = self._model.get_output_embeddings()
        self._out_norm = getattr(self._model.base_model, "norm", None)

    @tc.no_grad()
    def generate(self, text, **kwrds):
        messages = [{"role": "user", "content": text}]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self._device)

        outputs = self._model.generate(
            inputs,
            pad_token_id=self._tokenizer.eos_token_id,
            **kwrds
        )
        return self._tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]

    def get_activates(self, ids, return_logits=False):
        if isinstance(ids, str):
            ids = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(ids))
        outs = self._model(
            ids[:512].unsqueeze(0).to(self._device),
            output_hidden_states=True,
            return_dict=True
        )
        if not return_logits:
            return outs.hidden_states
        return outs.hidden_states, outs.logits

    def summarize_neuron(self, spans, max_new_tokens=64):
        shots = ""
        for words, label in examples:
            shots += f"Words: {words}\nConcept: {label}\n\n"

        span_text = ", ".join(spans[:20])
        prompt = f"{instruct}\n\n{shots}Words: {span_text}\nConcept:"

        return self.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
if __name__ == "__main__":
    generator = Generator("llama3-8b", device="cuda")
    print(generator.generate("Who is the current president of the United States?", max_new_tokens=128, do_sample=False))

    neuron_spans = ["January", "December", "April", "July", "September", "Tuesday"]
    summary = generator.summarize_neuron(neuron_spans)
    print("Neuron concept summary:", summary)

