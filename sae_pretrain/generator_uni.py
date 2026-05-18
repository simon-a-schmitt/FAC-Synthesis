import os
import sys
from typing import List, Dict, Any, Union
import transformers as trf
import torch as tc
from transformers import AutoTokenizer, AutoModelForCausalLM

trf.set_seed(42)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] if len(sys.argv) > 1 else "0"

CACHE_DIR = os.environ.get(
    "TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface")
)
os.makedirs(CACHE_DIR, exist_ok=True)


class UnifiedGenerator:
    """
    A unified generator supporting Llama / Mistral / Qwen families.
    Uses tokenizer.apply_chat_template for chat-format models.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_dir: str = None,
        local_files_only: bool = False,
        strict_local_paths: bool = False,
    ):
        self._name = model_name
        # Intelligenter Fallback: Wenn CUDA angefordert aber nicht verfügbar, nutze CPU
        if device == "cuda" and not tc.cuda.is_available():
            print("WARNING: CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
            device = "cpu"
        self._device = device
        self._dtype = tc.bfloat16 if dtype == "bfloat16" else tc.float16
        self._family = self.detect_family(model_name)
        self._cache_dir = cache_dir if cache_dir else CACHE_DIR
        self._local_files_only = local_files_only
        self._strict_local_paths = strict_local_paths
        self.build_model()

    def detect_family(self, model_name: str) -> str:
        name = model_name.lower()
        if "llama" in name:
            return "llama"
        if "mistral" in name or "mixtral" in name:
            return "mistral"
        if "qwen" in name:
            return "qwen"
        return "generic"

    @tc.no_grad()
    def build_model(self):
        print(f"Initializing model: {self._name}")
        print(f"Using device: {self._device}")
        maps = "cpu" if self._device == "cpu" else "auto"

        if self._strict_local_paths and not os.path.isdir(self._name):
            raise FileNotFoundError(
                f"strict_local_paths=True requires a local model directory, got: {self._name}"
            )

        tok = AutoTokenizer.from_pretrained(
            self._name,
            use_fast=True,
            cache_dir=self._cache_dir,
            local_files_only=self._local_files_only,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token else "</s>"
        tok.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            self._name,
            cache_dir=self._cache_dir,
            torch_dtype=self._dtype,
            device_map=maps,
            local_files_only=self._local_files_only,
        )
        model.config.pad_token_id = tok.pad_token_id
        model.eval()

        self._tokenizer = tok
        self._model = model
        print(f"Loaded {self._family.upper()} model successfully.")

    def build_messages(
        self,
        user_text: str = None,
        system_text: str = None,
        history: List[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        if history:
            msgs.extend(history)
        if user_text is not None:
            msgs.append({"role": "user", "content": user_text})
        return msgs

    @tc.no_grad()
    def generate(
        self,
        user_or_messages: Union[str, List[Dict[str, str]]],
        *,
        system: str = None,
        history: List[Dict[str, str]] = None,
        **kwrds,
    ) -> str:
        if isinstance(user_or_messages, str):
            messages = self.build_messages(
                user_text=user_or_messages, system_text=system, history=history
            )
        else:
            messages = user_or_messages

        enc = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Extract tensors from BatchEncoding and move to device
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        else:
            attention_mask = tc.ones_like(input_ids)
        kwrds.setdefault("pad_token_id", self._tokenizer.pad_token_id)
        kwrds.setdefault("max_new_tokens", 128)
        kwrds.setdefault("do_sample", True)
        kwrds.setdefault("temperature", 0.7)
        kwrds.setdefault("top_p", 0.9)

        outputs = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwrds,
        )

        gen_tokens = outputs[:, input_ids.shape[1]:]
        return self._tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    def get_activates(
        self,
        text_or_ids: Union[str, List[int]],
        *,
        system: str = None,
        history: List[Dict[str, str]] = None,
        add_generation_prompt: bool = False,
        return_logits: bool = False,
    ):
        attention_mask = None
        if isinstance(text_or_ids, str):
            messages = self.build_messages(
                user_text=text_or_ids, system_text=system, history=history
            )
            enc = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
            )
            # Handle both dict and tensor returns from apply_chat_template
            if isinstance(enc, dict):
                input_ids = enc["input_ids"].to(self._device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self._device)
            else:
                # enc is a tensor directly
                input_ids = enc.to(self._device)
                attention_mask = None
        else:
            input_ids = tc.tensor([text_or_ids[:512]], device=self._device)

        # Forward pass to get hidden states and optionally logits
        if attention_mask is not None:
            outs = self._model(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        else:
            outs = self._model(input_ids, output_hidden_states=True, return_dict=True)
        if not return_logits:
            return outs.hidden_states
        return outs.hidden_states, outs.logits


if __name__ == "__main__":
    # Example 1: Llama-3.1-8B-Instruct
    llama_gen = UnifiedGenerator("meta-llama/Llama-3.1-8B-Instruct", device="cuda")
    print(
        "Llama output:",
        llama_gen.generate("Who is the president of the United States?", max_new_tokens=64),
    )

    # Example 2: Qwen 2.5 7B Instruct
    qwen_gen = UnifiedGenerator("Qwen/Qwen2.5-7B-Instruct", device="cuda")
    print(
        "Qwen output:",
        qwen_gen.generate("Explain the concept of reinforcement learning.", max_new_tokens=64),
    )

    # Example 3: Mistral Instruct
    mistral_gen = UnifiedGenerator("mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
    print(
        "Mistral output:",
        mistral_gen.generate("What are black holes?", max_new_tokens=64),
    )

