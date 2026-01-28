import os
import sys
import random
import torch
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../useful_code"))
token=os.environ.get("HF_TOKEN")
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers.utils import PaddingStrategy
from useful_code.eval_reward_bench_bt import run_rewardbench_eval


class EvalRewardBenchCallback(TrainerCallback):
    def __init__(self, eval_interval=100, tokenizer=None, model_name="", seed=None):
        self.eval_interval = eval_interval
        self.tokenizer = tokenizer
        self.model_name_safe = (model_name or "").replace("/", "_")
        self.seed = seed

    def on_step_end(self, args, state, control, model=None, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is not None and not getattr(trainer, "is_world_process_zero", False):
            return control
        if getattr(args, "local_rank", -1) not in (-1, 0):
            return control
        if os.environ.get("RANK", "0") not in ("0", ""):
            return control

        if state.global_step > 0 and state.global_step % self.eval_interval == 0:
            print(f"\n[RewardBench] step {state.global_step} evaluating...")
            m = model.module if hasattr(model, "module") else model
            record_path = f"./logs_SAE_llama/rewardbench/{self.model_name_safe}/seed{self.seed}/step{state.global_step}.txt"
            os.makedirs(os.path.dirname(record_path), exist_ok=True)
            res = run_rewardbench_eval(
                model_or_path=m,
                tokenizer=self.tokenizer,
                dataset_name="allenai/reward-bench",
                split="filtered",
                record_path=record_path,
                batch_size=8,
                limit=None,
            )
            if "by_section" in res:
                print("[RewardBench] section:", res["by_section"])
            else:
                acc_section = res.get("by_section_acc", {})
                f1_section = res.get("by_section_f1", {})
                print("[RewardBench] section (Acc):", acc_section)
                print("[RewardBench] section (F1):", f1_section)
        return control


@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1)
    deepspeed: Optional[str] = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 8e-5
    weight_decay: float = 0.01
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    bf16: bool = True
    num_train_epochs: int = 5.0
    train_set_path: str = "xxx"
    output_path: str = "./models"
    gradient_checkpointing: bool = False
    optim: str = "adamw_torch_fused"
    lr_scheduler_type: str = "cosine"
    max_length: int = 1024
    seed: int = 42
    save_every_steps: int = 10000
    eval_every_steps: int = 50
    init_model_path: Optional[str] = "xxx"


parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed)
print(args.seed)

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name,
    token=os.environ["HF_TOKEN"],
    use_fast=False,
)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.truncation_side = "left"
tokenizer.model_max_length = args.max_length


def build_dataset(tokenizer, train_path):
    def tokenize(sample):
        pos_text = tokenizer.apply_chat_template(sample["chosen"], tokenize=False).replace(tokenizer.bos_token, "")
        neg_text = tokenizer.apply_chat_template(sample["rejected"], tokenize=False).replace(tokenizer.bos_token, "")
        pos_tok = tokenizer(pos_text, truncation=True)
        neg_tok = tokenizer(neg_text, truncation=True)
        sample.update(
            {
                "input_ids_j": pos_tok["input_ids"],
                "attention_mask_j": pos_tok["attention_mask"],
                "input_ids_k": neg_tok["input_ids"],
                "attention_mask_k": neg_tok["attention_mask"],
            }
        )
        return sample

    ds = load_dataset(train_path, token=os.environ["HF_TOKEN"], split="train", keep_in_memory=True).shuffle(seed=args.seed)

    ds = ds.map(tokenize, num_proc=8, desc="Tokenizing dataset")
    split = ds.train_test_split(test_size=0.1, seed=args.seed)
    return ds, split["test"]

train_dataset, eval_dataset = build_dataset(tokenizer, args.train_set_path)

model = AutoModelForSequenceClassification.from_pretrained(
    args.init_model_path,
    token=os.environ["HF_TOKEN"],
    num_labels=1,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
)

model.config.use_cache = not args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id

for n, p in model.named_parameters():
    if not (n.startswith("score") or ".score" in n or ".classifier" in n):
        p.requires_grad = False

model.gradient_checkpointing_enable()


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = []
        for f in features:
            merged.append({"input_ids": f["input_ids_j"], "attention_mask": f["attention_mask_j"]})
            merged.append({"input_ids": f["input_ids_k"], "attention_mask": f["attention_mask_k"]})
        padded = self.tokenizer.pad(
            merged,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        }


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    preds = np.asarray(preds).squeeze()
    assert preds.size % 2 == 0, "Predictions length must be even (pairs of j/k)."
    pos = preds[0::2]
    neg = preds[1::2]
    acc = float(np.mean(pos > neg))
    return {"accuracy": acc}


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])[0]
        jidx = torch.arange(0, rewards.size(0), 2, device=rewards.device)
        kidx = jidx + 1
        loss = -torch.nn.functional.logsigmoid(rewards[jidx] - rewards[kidx]).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards[jidx], "rewards_k": rewards[kidx]}
        return loss


training_args = TrainingArguments(
    output_dir=f"{args.output_path}/seed{args.seed}",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    num_train_epochs=args.num_train_epochs,
    weight_decay=args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=args.eval_every_steps,
    save_strategy="no",
    save_steps=args.save_every_steps,
    save_total_limit=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    deepspeed=args.deepspeed,
    local_rank=args.local_rank,
    remove_unused_columns=False,
    bf16=args.bf16,
    logging_strategy="steps",
    logging_steps=1000,
    optim=args.optim,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=0.1,
    report_to="none",
    seed=args.seed,
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_length),
    compute_metrics=compute_metrics,
    callbacks=[EvalRewardBenchCallback(eval_interval=args.eval_every_steps, tokenizer=tokenizer, model_name=args.model_name, seed=args.seed)],
)

trainer.train()

print("\n RewardBench ...")
m = model.module if hasattr(model, "module") else model
record_path = f"./logs_base/rewardbench/{args.model_name.replace('/', '_')}/seed{args.seed}/final_eval.txt"
os.makedirs(os.path.dirname(record_path), exist_ok=True)
res = run_rewardbench_eval(
    model_or_path=m,
    tokenizer=tokenizer,
    dataset_name="allenai/reward-bench",
    split="filtered",
    record_path=record_path,
    batch_size=8,
    limit=None,
)
if "by_section" in res:
    print("[RewardBench] Final section:", res["by_section"])
else:
    print("[RewardBench] Final section (Acc):", res.get("by_section_acc", {}))
    print("[RewardBench] Final section (F1):", res.get("by_section_f1", {}))

