import os
import sys
from typing import List, Dict, Any, Union
import transformers as trf
import torch as tc
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

trf.set_seed(42)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

CACHE_DIR = os.environ.get(
    "TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface")
)
os.makedirs(CACHE_DIR, exist_ok=True)


class UnifiedGenerator:
    """
    A unified generator supporting Llama / Mistral / Qwen / Gemma families.
    Uses tokenizer.apply_chat_template for chat-format models.
    Gemma is loaded and used as a text-only generator via AutoProcessor.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        cache_dir: str = None,
        local_files_only: bool = False,
        strict_local_paths: bool = False,
        lora_path: Optional[str] = None,
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
        self._lora_path = lora_path
        self.build_model()

    def detect_family(self, model_name: str) -> str:
        name = model_name.lower()
        if "gemma" in name:
            return "gemma"
        if "llama" in name:
            return "llama"
        if "mistral" in name or "mixtral" in name:
            return "mistral"
        if "qwen" in name:
            return "qwen"
        return "generic"

    @tc.no_grad()
    def build_model(self):
        print(f"Initializing model: {self._name}", flush=True)
        print(f"Using device: {self._device}", flush=True)
        maps = "cpu" if self._device == "cpu" else "auto"

        if self._strict_local_paths and not os.path.isdir(self._name):
            raise FileNotFoundError(
                f"strict_local_paths=True requires a local model directory, got: {self._name}"
            )

        if self._family == "gemma":
            processor = AutoProcessor.from_pretrained(
                self._name,
                cache_dir=self._cache_dir,
                local_files_only=self._local_files_only,
                padding_side="left",
            )
            self._processor = processor
            tok = processor.tokenizer
        else:
            self._processor = None
            tok = AutoTokenizer.from_pretrained(
                self._name,
                use_fast=True,
                cache_dir=self._cache_dir,
                local_files_only=self._local_files_only,
            )

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token if tok.eos_token else "</s>"
        tok.padding_side = "left"

        if self._family == "gemma":
            model = AutoModelForCausalLM.from_pretrained(
                self._name,
                cache_dir=self._cache_dir,
                dtype=self._dtype,
                device_map=maps,
                local_files_only=self._local_files_only,
                attn_implementation="sdpa",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self._name,
                cache_dir=self._cache_dir,
                torch_dtype=self._dtype,
                device_map=maps,
                local_files_only=self._local_files_only,
            )

        if self._lora_path:
            import json
            import dataclasses
            from peft import PeftModel, LoraConfig

            print(f"Loading LoRA adapter from: {self._lora_path}", flush=True)
            # The adapter was saved with a newer peft version whose LoraConfig
            # has extra fields (e.g. alora_invocation_tokens, arrow_config)
            # unknown to the peft version installed in this env. Those extra
            # fields are inactive (null/false) here, so drop anything the
            # installed LoraConfig doesn't recognize before instantiating it.
            with open(os.path.join(self._lora_path, "adapter_config.json")) as f:
                raw_config = json.load(f)
            known_fields = {f.name for f in dataclasses.fields(LoraConfig)}
            filtered_config = {k: v for k, v in raw_config.items() if k in known_fields}
            lora_config = LoraConfig(**filtered_config)
            model = PeftModel.from_pretrained(model, self._lora_path, config=lora_config)
            model = model.merge_and_unload()

        model.config.pad_token_id = tok.pad_token_id
        model.eval()

        self._tokenizer = tok
        self._model = model
        print(f"Loaded {self._family.upper()} model successfully.", flush=True)

    @property
    def tokenizer(self):
        return self._tokenizer

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

    def _decode_gemma_response(self, raw: str, prefix=None) -> str:
        try:
            parsed = self._processor.parse_response(raw, prefix=prefix if prefix is not None else "")
            return parsed["response"]
        except (AttributeError, KeyError, TypeError):
            pass
        # Fallback: strip channel markers by hand and keep the tail after the
        # last channel marker, so the pipeline doesn't hard-break on version
        # mismatches where parse_response is unavailable.
        text = raw
        if "<|channel|>" in text:
            text = text.rsplit("<|channel|>", 1)[-1]
        for tag in ("<|start|>", "<|end|>", "<|message|>", "<|channel|>"):
            text = text.replace(tag, "")
        return text.strip()

    @tc.no_grad()
    def generate(
        self,
        user_or_messages: Union[str, List[Dict[str, str]]],
        *,
        system: str = None,
        history: List[Dict[str, str]] = None,
        use_chat_template: bool = True,
        max_input_tokens: Optional[int] = None,
        return_usage: bool = False,
        **kwrds,
    ) -> Union[str, tuple]:
        if not kwrds.get("do_sample", True):
            kwrds.pop("temperature", None)
            kwrds.pop("top_p", None)

        kwrds.setdefault("pad_token_id", self._tokenizer.pad_token_id)
        kwrds.setdefault("max_new_tokens", 128)
        kwrds.setdefault("do_sample", True)
        kwrds.setdefault("temperature", 0.7)
        kwrds.setdefault("top_p", 0.9)

        # None values override transformers' defaults (e.g. use_cache=None disables KV cache),
        # causing attention mask / KV cache size mismatches during generation.
        kwrds = {k: v for k, v in kwrds.items() if v is not None}

        if self._family == "gemma":
            if isinstance(user_or_messages, str):
                messages = self.build_messages(
                    user_text=user_or_messages, system_text=system, history=history
                )
            else:
                messages = user_or_messages

            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = {
                k: (v.to(self._device) if isinstance(v, tc.Tensor) else v)
                for k, v in inputs.items()
            }

            if max_input_tokens is not None:
                max_input_tokens = int(max_input_tokens)
                if max_input_tokens > 0 and inputs["input_ids"].shape[1] > max_input_tokens:
                    inputs = {
                        k: (v[:, -max_input_tokens:] if isinstance(v, tc.Tensor) else v)
                        for k, v in inputs.items()
                    }

            input_ids = inputs["input_ids"]
            outputs = self._model.generate(**inputs, **kwrds)

            gen_tokens = outputs[:, input_ids.shape[1]:]
            raw = self._tokenizer.batch_decode(gen_tokens, skip_special_tokens=False)[0]
            text = self._decode_gemma_response(raw, prefix=input_ids[0])

            if not return_usage:
                return text
            usage = {
                "input_tokens": int(input_ids.shape[1]),
                "output_tokens": int(gen_tokens.shape[1]),
            }
            return text, usage

        if use_chat_template:
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
        else:
            # raw completion: kein Template, kein assistant-Header
            assert isinstance(user_or_messages, str), \
                "raw completion mode expects a plain string prompt"
            enc = self._tokenizer(user_or_messages, return_tensors="pt")
        # Handle both dict-like (dict / BatchEncoding) and tensor returns from apply_chat_template
        if isinstance(enc, tc.Tensor):
            input_ids = enc
            attention_mask = None
        else:
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")

        # ensure batch dimension: some tokenizers return shape (seq_len,) for single example
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self._device)

        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.to(self._device)
        else:
            attention_mask = tc.ones_like(input_ids)
        if max_input_tokens is not None:
            max_input_tokens = int(max_input_tokens)
            if max_input_tokens > 0 and input_ids.shape[1] > max_input_tokens:
                input_ids = input_ids[:, -max_input_tokens:]
                attention_mask = attention_mask[:, -max_input_tokens:]

        print("ACTUAL INPUT TAIL:", repr(self._tokenizer.decode(input_ids[0][-25:])))

        outputs = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwrds,
        )

        gen_tokens = outputs[:, input_ids.shape[1]:]
        text = self._tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        if not return_usage:
            return text
        usage = {
            "input_tokens": int(input_ids.shape[1]),
            "output_tokens": int(gen_tokens.shape[1]),
        }
        return text, usage

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
            # Handle both dict-like (dict / BatchEncoding) and tensor returns from apply_chat_template
            if isinstance(enc, tc.Tensor):
                input_ids = enc.to(self._device)
                attention_mask = None
            else:
                input_ids = enc["input_ids"].to(self._device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self._device)
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

    # Example 4: Gemma-4 (text-only)
    gemma_gen = UnifiedGenerator("google/gemma-4-31B-it", device="cuda")
    print(
        "Gemma output:",
        gemma_gen.generate("What are black holes?", max_new_tokens=64),
    )

