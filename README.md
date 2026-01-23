# Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs

This is the official implementation of the paper: Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs.

---

## Introduction

The diversity of post-training data is critical for effective downstream performance in large language models (LLMs). Existing metrics primarily quantify diversity in surface-level linguistic features in text space, which only serve as weak proxies for the task-relevant latent representations that ultimately determine downstream performance.
In this work, we first introduce **_Feature Activation Coverage_ (FAC)** that measures data diversity in an interpretable feature space. 
Building upon this metric, we further propose a diversity-driven data synthesis framework, **FAC Synthesis**, that first uses a sparse autoencoder to identify missing features from a seed dataset, and then generates synthetic samples explicitly reflects these missing features.
Experiments show that our approach consistently improves diversity of generated synthetic data and task performance on various downstream tasks, including instruction following, toxicity detection, reward modeling, and behavior steering. 
Interestingly, we identify a shared, interpretable feature space across model families (i.e., LLaMA, Mistral, and Qwen), enabling cross-model knowledge transfer.
Our work provides a solid and practical methodology for exploring data-centric optimization of LLMs.

---

## Getting Started

### Installation

```bash
git clone https://github.com/<org-or-user>/FAC-Synthesis.git
cd FAC-Synthesis
pip install -r requirements.txt
```

### Quick Run (Example)

```bash
# 1) Identify missing features
python scripts/identify_missing_features.py --task <task_name>

# 2) Generate feature-guided synthetic data
python scripts/synthesize.py --task <task_name>

# 3) Train / evaluate (if applicable)
python scripts/train_eval.py --task <task_name>
```

---

## Project Layout

```text
.
├── prompts/              # Prompt templates (synthesis / annotation / evaluation)
├── feature_analysis/     # Feature coverage + missing-feature selection
├── synthesis/            # Feature-guided generation + filtering
├── tasks/                # Task-specific train/eval/steering code
├── scripts/              # End-to-end runnable entry points
└── README.md
```

---

## Prompts

We provide prompt templates for:

- Text span explanation (feature summarization)
- Feature relevance annotation (toxicity / helpfulness / sycophancy / survival instinct / instruction following)
- Task-specific synthesis (toxicity / reward modeling / instruction following)
- Behavior steering evaluation prompts (sycophancy / survival instinct)

---

## Notes on Reproducibility

- All baselines are configured to use the **same synthetic data budget** as our method to control variables.
- Some generated candidates are removed during filtering if the target missing features are not reliably activated or fall below an activation threshold.

---

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{fac_synthesis_2026,
  title     = {FAC-Synthesis},
  author    = {Anonymous Authors},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## License

Released under the **MIT License**. See `LICENSE` for details.

---

## Acknowledgements

This project uses public benchmarks and open-source tooling from the HuggingFace ecosystem.
