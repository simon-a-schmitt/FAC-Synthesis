import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from dataclasses import dataclass
from sklearn.metrics import average_precision_score
from scipy.special import softmax
from datasets import concatenate_datasets
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class Args:
    base_model_dir: str = ".../models/toxic_cls"
    valid_data_path: str = "...tsv"
    test_data_path: str = "...tsv"
    output_dir: str = "./model_finetuned"
    seed: int = 42
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 8e-5
    num_train_epochs: float = 15.0
    max_length: int = 512
    bf16: bool = True


parser = HfArgumentParser((Args,))
args = parser.parse_args_into_dataclasses()[0]

print(args.seed)

base_path_prefix = ".../models/toxic_cls/"
args.base_model_dir = os.path.join(base_path_prefix, "checkpoint")
print("args.base_model_dir", args.base_model_dir)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.truncation_side = "right"
tokenizer.model_max_length = args.max_length


def load_tsv(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    df["label"] = df["label"].astype(int)
    return Dataset.from_dict(df.to_dict(orient="list"))


def load_and_sample_safe_dataset(real_data_path, num_pos_samples=1000, seed=42):
    df = pd.read_csv(real_data_path, sep="\t", header=None, names=["text", "label"])
    df["label"] = df["label"].astype(int)
    pos_samples = df[df["label"] == 0].sample(n=num_pos_samples, random_state=seed)
    return Dataset.from_dict(pos_samples.to_dict(orient="list"))


train_ds_pos = load_tsv(
    "xxx.tsv"
)
td_pos_subset = load_and_safe_toxic_dataset(
    real_data_path="xxx.tsv",
    num_pos_samples=1000,
    seed=args.seed,
)
train_ds = concatenate_datasets([td_pos_subset, train_ds_pos.shuffle(args.seed).select(range(200))])
train_ds = train_ds.shuffle(args.seed)
valid_ds = load_tsv(args.valid_data_path)
test_ds = load_tsv(args.test_data_path)


def preprocess(example):
    prompt = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": example["text"]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        if hasattr(tokenizer, "apply_chat_template")
        else example["text"]
    )
    enc = tokenizer(prompt, truncation=True, padding=False, max_length=args.max_length)
    enc["labels"] = int(example["label"])
    return enc


train_ds = train_ds.map(preprocess, remove_columns=["text", "label"], num_proc=8, desc="tokenizing-train")
train_ds = train_ds.shuffle(seed=args.seed)
valid_ds = valid_ds.map(preprocess, remove_columns=["text", "label"], num_proc=8, desc="tokenizing-valid")
test_ds = test_ds.map(preprocess, remove_columns=["text", "label"], num_proc=8, desc="tokenizing-test")

model = AutoModelForSequenceClassification.from_pretrained(
    args.base_model_dir,
    torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    num_labels=2,
)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

for n, p in model.named_parameters():
    if not (n.startswith("score") or ".score" in n or ".classifier" in n):
        p.requires_grad = False

model.gradient_checkpointing_enable()


def compute_metrics(eval_pred):
    bias = 0.0
    logits, labels = eval_pred
    logits = np.array(logits)
    labels = np.array(labels)

    save_dir = os.path.join(args.output_dir, "logits")
    os.makedirs(save_dir, exist_ok=True)

    logits_path = os.path.join(save_dir, "eval_logits.tsv")
    print(logits_path)
    np.savetxt(
        logits_path,
        np.column_stack((logits, labels)),
        delimiter="\t",
        fmt="%.6f",
        header="logit0\tlogit1\tlabel",
        comments="",
    )
    print(f"[INFO] Saved logits to: {logits_path}")

    try:
        auprc = float(average_precision_score(labels, softmax(logits, axis=1)[:, 1]))
    except Exception as e:
        auprc = float("nan")
        print(f"[WARN] auprc failed: {e}")

    return {"auprc": auprc}


training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    save_strategy="no",
    evaluation_strategy="steps",
    logging_steps=500,
    bf16=args.bf16,
    report_to="none",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

print("Evaluating on test set…")
metrics = trainer.evaluate(eval_dataset=test_ds)
print(metrics)

