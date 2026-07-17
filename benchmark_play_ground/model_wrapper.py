import os
import sys
from typing import Optional

import torch as tc

# Ensure repo-local imports like UnifiedGenerator are available
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAE_PRETRAIN_DIR = os.path.join(ROOT_DIR, "..", "sae_pretrain")
if SAE_PRETRAIN_DIR not in sys.path:
    sys.path.insert(0, SAE_PRETRAIN_DIR)

from sae_pretrain.generator_uni import UnifiedGenerator


class LocalModel:
    def __init__(self, model_path: str, device: str = "cuda", dtype: str = "bfloat16", local_files_only: bool = True, strict_local_paths: bool = True):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.strict_local_paths = strict_local_paths
        self._gen = None

    def load(self):
        self._gen = UnifiedGenerator(
            model_name=self.model_path,
            device=self.device,
            dtype=self.dtype,
            local_files_only=self.local_files_only,
            strict_local_paths=self.strict_local_paths,
        )

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
        system: Optional[str] = None,
    ) -> str:
        if self._gen is None:
            self.load()
        gen = self._gen
        assert gen is not None

        if system is None:
            return gen.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                max_input_tokens=max_input_tokens,
                use_cache=use_cache,
            )

        return gen.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            max_input_tokens=max_input_tokens,
            use_cache=use_cache,
            system=system,
        )
