import torch
import transformers.models.mistral.modeling_mistral as mistral
import transformers.models.llama.modeling_llama as llama

try:
    import transformers.models.qwen2.modeling_qwen2 as qwen2
except Exception:
    qwen2 = None
try:
    import transformers.models.qwen2_moe.modeling_qwen2_moe as qwen2_moe
except Exception:
    qwen2_moe = None
try:
    import transformers.models.qwen2_5.modeling_qwen2_5 as qwen2_5
except Exception:
    qwen2_5 = None


KEY = "__sae_surgery"


def _sae_forward_common(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self-Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # MLP
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = residual + self.mlp(hidden_states)

    # SAE hook
    if hasattr(self, KEY):
        sae_fn = getattr(self, KEY)
        if sae_fn is not None:
            hidden_states = sae_fn(hidden_states.to(torch.float32)).to(hidden_states.dtype)

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


def sae_llama_forward(self, *args, **kwargs):
    return _sae_forward_common(self, *args, **kwargs)


def sae_mistral_forward(self, *args, **kwargs):
    return _sae_forward_common(self, *args, **kwargs)


def sae_qwen_forward(self, *args, **kwargs):
    return _sae_forward_common(self, *args, **kwargs)


ops = {
    "mistral": ([mistral.MistralDecoderLayer], sae_mistral_forward),
    "llama": ([llama.LlamaDecoderLayer], sae_llama_forward),
    "qwen": ([], sae_qwen_forward),
}


if qwen2 is not None and hasattr(qwen2, "Qwen2DecoderLayer"):
    ops["qwen"][0].append(qwen2.Qwen2DecoderLayer)
if qwen2_moe is not None and hasattr(qwen2_moe, "Qwen2MoeDecoderLayer"):
    ops["qwen"][0].append(qwen2_moe.Qwen2MoeDecoderLayer)
if qwen2_5 is not None and hasattr(qwen2_5, "Qwen2_5DecoderLayer"):
    ops["qwen"][0].append(qwen2_5.Qwen2_5DecoderLayer)

for class_list, new_forward in ops.values():
    for target_class in class_list:
        setattr(target_class, "forward", new_forward)


def mount_function(model, name, layer_idx, hook):
    assert layer_idx > 0
    for attr in ["enabled", "monitoring", "computing", "early_stop"]:
        if not hasattr(hook, attr):
            print(f"Hook has no attribute {attr}, setting default.")
            setattr(hook, attr, False)
    for func in ["monitor", "compute_loss", "generate"]:
        if not hasattr(hook, func):
            print(f"Hook has no function {func}, setting default.")
            setattr(hook, func, lambda x: x)

    # hook function
    # hook -> Collector object
    def call_hook(x):
        if not hook.enabled:
            return x
        if hook.monitoring:
            hook.monitor(x)
            if hook.early_stop:
                raise RuntimeError
            return x
        if hook.computing:
            y = hook.compute_loss(x)
            if hook.early_stop:
                raise RuntimeError
            return y
        return hook.generate(x)

    # ops contains list of targeted layer classes and correct forward method for each model family
    class_list, _ = ops[name]

    # Flag: Did we find any suitable layer? 
    hit = False

    # Recursive search trough all blocks, layers, submodules
    for mod_name, layer in model.named_modules():
        # Only consider layers of the specified type (from ops)
        if any(isinstance(layer, cls) for cls in class_list):

            # Decrement layer counter for matching layers 
            layer_idx -= 1

            # Correct layer found, mount hook
            if layer_idx == 0:

                # Within the found layer instance a new attribute of name __sae_surgery (view KEY definition) is set
                setattr(layer, KEY, call_hook)
                print(f"Mounted hook at {mod_name}")
                hit = True
                break
    if not hit:
        raise RuntimeError(
            f"Failed to mount hook: no target layer matched for '{name}'. "
            f"Check model family and layer index."
        )


def switch_mode(hook, mode):
    mode = mode.lower()
    assert mode in {"turnoff", "turnon", "monitor", "train", "generate"}
    if mode == "turnoff":
        hook.enabled = False
        return
    hook.enabled = True
    if mode == "turnon":
        return
    hook.monitoring = mode == "monitor"
    hook.computing = mode == "train"

