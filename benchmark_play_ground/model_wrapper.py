from typing import Any, List, Optional

import torch as tc

from benchmark_play_ground.generator_uni import UnifiedGenerator


class LocalModel:
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bfloat16", local_files_only: bool = True, strict_local_paths: bool = True, lora_path: Optional[str] = None):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.strict_local_paths = strict_local_paths
        self.lora_path = lora_path
        self._gen = None

    def load(self):
        self._gen = UnifiedGenerator(
            model_name=self.model_path,
            device=self.device,
            dtype=self.dtype,
            local_files_only=self.local_files_only,
            strict_local_paths=self.strict_local_paths,
            lora_path=self.lora_path,
        )

    @property
    def tokenizer(self):
        if self._gen is None:
            self.load()
        return self._gen.tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        no_repeat_ngram_size: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_input_tokens: Optional[int] = None,
        use_cache: Optional[bool] = None,
        stop_strings: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
        system: Optional[str] = None,
        return_usage: bool = False,
        use_chat_template: bool = True,
    ):
        if self._gen is None:
            self.load()
        gen = self._gen
        assert gen is not None

        kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            max_input_tokens=max_input_tokens,
            use_cache=use_cache,
            stop_strings=stop_strings,
            tokenizer=tokenizer,
            return_usage=return_usage,
            use_chat_template=use_chat_template,
        )
        if system is not None:
            kwargs["system"] = system

        return gen.generate(prompt, **kwargs)

    def render_prompt(
        self,
        prompt: str,
        system: Optional[str] = None,
        history: Optional[List[dict]] = None,
        use_chat_template: bool = True,
    ) -> str:
        """Return the exact prompt text (incl. special/template tokens) that
        `generate()` would feed to the model for this input.

        Re-runs only the chat-template + tokenize step (no forward pass), so
        this is cheap to call for logging/debugging before a real generate().
        """
        if self._gen is None:
            self.load()
        gen = self._gen
        assert gen is not None

        if not use_chat_template:
            enc = gen.tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"]
        else:
            messages = gen.build_messages(user_text=prompt, system_text=system, history=history)
            if gen._family == "gemma":
                inputs = gen._processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids = inputs["input_ids"]
            else:
                enc = gen.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                input_ids = enc if isinstance(enc, tc.Tensor) else enc["input_ids"]

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        return gen.tokenizer.decode(input_ids[0], skip_special_tokens=False)
