from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from prompt_config_fn import SYSTEM_PROMPT, EXAMPLES

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def llama3_generate(prompt: str, temperature=0.8, max_new_tokens=900, num_return_sequences=2) -> list[str]:
    messages = []

    if len(EXAMPLES) > 0:
        first_content = SYSTEM_PROMPT.format(feature_content=EXAMPLES[0][0].strip())
        messages.append({"role": "user", "content": first_content})
        messages.append({"role": "assistant", "content": EXAMPLES[0][1].strip()})

        for example in EXAMPLES[1:]:
            messages.append({"role": "user", "content": example[0].strip()})
            messages.append({"role": "assistant", "content": example[1].strip()})

    if len(messages) == 0:
        messages.append({"role": "user", "content": SYSTEM_PROMPT.format(feature_content=prompt.strip())})
    else:
        messages.append({"role": "user", "content": prompt.strip()})

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )
    outputs = outputs[:, input_ids.shape[1]:]
    return [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]

