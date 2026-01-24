import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.helpers import add_vector_from_position, find_instruction_end_postion
from typing import Optional
import os

class AttnWrapper(t.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(t.nn.Module):
    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.activations = None
        self.add_activations = None
        self.from_position = None
        self.save_internal_decodings = False
        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]

        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (
                t.norm(last_token_activations) * t.norm(self.calc_dot_product_with)
            )
            self.dot_products.append(dot_product.cpu().item())

        if self.add_activations is not None:
            augmented_output = add_vector_from_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                from_pos=self.from_position,
            )
            output = (augmented_output,) + output[1:]

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.dot_products = []


class Llama3Wrapper:
    def __init__(
        self,
        model_path: str,
        hf_token: Optional[str] = None,
        use_chat: bool = True,
        override_model_weights_path: Optional[str] = None,
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.use_chat = use_chat

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=hf_token,
            torch_dtype=t.bfloat16,
            device_map="auto"
        )

        if override_model_weights_path is not None:
            self.model.load_state_dict(t.load(override_model_weights_path))

        self.model.eval()

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def _build_chat_input(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})
        if model_output is not None:
            messages.append({"role": "assistant", "content": model_output})
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        ).to(self.device)
        return tokens

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        for layer in self.model.model.layers:
            layer.from_position = pos

    @t.no_grad()
    def generate(self, tokens, max_new_tokens=100):
        generated = self.model.generate(
            inputs=tokens,
            max_new_tokens=max_new_tokens,
            top_k=1
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def generate_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None, max_new_tokens: int = 100) -> str:
        tokens = self._build_chat_input(user_input, model_output, system_prompt)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def get_logits(self, tokens):
        attention_mask = (tokens != self.tokenizer.pad_token_id).long()
        with t.no_grad():
            output = self.model(input_ids=tokens, attention_mask=attention_mask)
        return output.logits

    def get_logits_from_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None) -> t.Tensor:
        tokens = self._build_chat_input(user_input, model_output, system_prompt)
        return self.get_logits(tokens)

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

