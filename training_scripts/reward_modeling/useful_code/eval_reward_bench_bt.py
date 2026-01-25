from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, List
import warnings
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    pipeline,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from peft import PeftModel

tqdm.pandas()
warnings.filterwarnings("ignore", message=".*promote has been superseded.*")

# -------- Constants --------
CATEGORIES: Dict[str, List[str]] = {
    "chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"],
    "chat-hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse", "xstest-should-respond", "donotanswer"],
    "reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"],
}

EXAMPLE_COUNTS: Dict[str, int] = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING: Dict[str, List[str]] = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}


def _calc_section_scores(
    example_counts: Dict[str, int],
    subset_mapping: Dict[str, List[str]],
    metrics: Dict[str, float],
) -> Dict[str, float]:
    section_scores: Dict[str, float] = {}
    for section, tests in subset_mapping.items():
        total_weighted, total_examples = 0.0, 0
        for test in tests:
            if test in metrics:
                total_weighted += float(metrics[test]) * int(example_counts[test])
                total_examples += int(example_counts[test])
        section_scores[section] = round(100.0 * total_weighted / total_examples, 2) if total_examples > 0 else 0.0
    return section_scores


def run_rewardbench_eval(
    model_or_path: Union[str, PreTrainedModel],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    dataset_name: str = "allenai/reward-bench",
    split: str = "filtered",
    record_path: Optional[str] = None,
    device: Optional[int] = None,
    batch_size: int = 8,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    if not isinstance(model_or_path, str) and isinstance(model_or_path, PeftModel):
        print("[RewardBench] Detected LoRA model, merging adapters into base model...")
        model_or_path = model_or_path.merge_and_unload()

    if device is None:
        device = -1  # enforce CPU-safe

    if tokenizer is None:
        if isinstance(model_or_path, str):
            tokenizer = AutoTokenizer.from_pretrained(model_or_path, use_fast=True)
        else:
            raise ValueError("When passing a model instance, please also pass a matching tokenizer.")

    rm_pipe = pipeline(
        task="text-classification",
        model=model_or_path,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        model_kwargs={"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {},
        return_all_scores=False,
        function_to_apply="none",
        batch_size=batch_size,
    )

    ds = load_dataset(dataset_name, split=split, keep_in_memory=True)
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    def _format_chat(prompt: str, resp: str) -> str:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp},
        ]
        s = tokenizer.apply_chat_template(messages, tokenize=False)
        bos = getattr(tokenizer, "bos_token", None)
        return s.replace(bos, "") if bos else s

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for ex in tqdm(ds, desc="RewardBench"):
        prompt, chosen, rejected = ex.get("prompt"), ex.get("chosen"), ex.get("rejected")
        subset = ex.get("subset", "unknown")
        if not prompt or not chosen or not rejected:
            skipped += 1
            continue
        texts = [_format_chat(prompt, chosen), _format_chat(prompt, rejected)]
        outputs = rm_pipe(texts)
        scores = [o["score"] for o in outputs]

        if scores[0] == scores[1]:
            pred = 0.5
        elif scores[0] > scores[1]:
            pred = 1.0
        else:
            pred = 0.0

        rows.append({"subset": subset, "pred": pred})

    if len(rows) == 0:
        raise RuntimeError("No valid examples to evaluate. Check dataset fields.")

    df = pd.DataFrame(rows)

    by_subset_acc, by_subset_f1 = {}, {}
    for subset_name in sorted(df["subset"].unique()):
        preds = df.loc[df["subset"] == subset_name, "pred"].values
        labels = np.ones_like(preds)
        acc = np.mean(preds == labels)
        f1 = acc
        by_subset_acc[subset_name] = float(acc)
        by_subset_f1[subset_name] = float(f1)

    section_acc = _calc_section_scores(EXAMPLE_COUNTS, SUBSET_MAPPING, by_subset_acc)
    section_f1 = _calc_section_scores(EXAMPLE_COUNTS, SUBSET_MAPPING, by_subset_f1)

    subset_to_category = {}
    for category, subset_list in SUBSET_MAPPING.items():
        for subset in subset_list:
            subset_to_category[subset] = category

    df_acc = []
    for subset, acc in by_subset_acc.items():
        row = {
            "category": subset_to_category.get(subset, "unknown"),
            "subset": subset,
            "accuracy": round(acc * 100, 2),
            "n": EXAMPLE_COUNTS.get(subset, 0),
        }
        df_acc.append(row)

    df_acc = pd.DataFrame(df_acc)

    print("\n==== Accuracy by Subset (with Category) ====")
    print(df_acc)
    
    result = {
        "by_subset_acc": by_subset_acc,
        "by_subset_f1": by_subset_f1,
        "by_section_acc": section_acc,
        "by_section_f1": section_f1,
        "overall_mean_acc": float(np.mean(list(by_subset_acc.values()))),
        "overall_mean_f1": float(np.mean(list(by_subset_f1.values()))),
        "skipped": skipped,
        "evaluated": len(df),
    }

    if record_path:
        with open(record_path, "a") as f:
            title = model_or_path if isinstance(model_or_path, str) else "in_memory_model"
            f.write(f"{title}\n")
            for k in ["Chat", "Chat Hard", "Safety", "Reasoning"]:
                f.write(f"{k}\tAcc={section_acc.get(k, 0):.2f}\tF1={section_f1.get(k, 0):.2f}\n")

    return result


@dataclass
class ScriptArguments:
    data_set_name: Optional[str] = field(default="allenai/reward-bench")
    record_dir: Optional[str] = field(default="./bench_mark_eval.txt")
    reward_name_or_path: Optional[str] = field(
    default="REWARD_MODEL_PATH"
    )
    split: Optional[str] = field(default="filtered")
    batch_size: Optional[int] = field(default=8)
    limit: Optional[int] = field(default=0)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    res = run_rewardbench_eval(
        model_or_path=args.reward_name_or_path,
        tokenizer=None,
        dataset_name=args.data_set_name,
        split=args.split,
        record_path=args.record_dir,
        batch_size=args.batch_size,
        limit=(args.limit if args.limit and args.limit > 0 else None),
    )

    print("\n==== RewardBench Result ====")
    print("By section (Acc):", res["by_section_acc"])
    print("By section (F1):", res["by_section_f1"])
    print("Overall mean Acc:", res["overall_mean_acc"])
    print("Overall mean F1:", res["overall_mean_f1"])
    print(f"Evaluated={res['evaluated']}  Skipped={res['skipped']}")

