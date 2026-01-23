# FAC-Synthesis

Code and prompts for **feature-guided synthetic data generation** using **Sparse Autoencoder (SAE) features**, with downstream evaluation on multiple LLM tasks.

This repo is organized to be **simple, reproducible, and modular**, following the “pipeline-first” style commonly used in open research codebases (e.g., HF Open-R1).

---

## Overview

The core workflow is:

1. **Feature extraction** with SAE activations  
2. **Missing-feature identification** by comparing coverage between an anchor set and task-relevant data  
3. **Feature-guided synthesis** + activation-based quality control  
4. **Downstream training / evaluation** (or steering-vector construction)

Supported tasks:
- Toxicity Detection
- Reward Modeling
- Behavior Steering (Sycophancy / Survival Instinct)
- Instruction Following

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
