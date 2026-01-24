# Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs

This is the official implementation of the paper: 'Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs'.

---

## Abstract

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
git clone https://anonymous.4open.science/r/Anonymous_1196
cd Anonymous_1196
pip install -r requirements.txt
```

---

### Repository Structure

```text
FAC-Synthesis/
├── LICENSE
├── README.md
├── requirements.txt
│
├── sae_pretrain/                 # SAE pretraining
│   ├── datasets/                 # pretraining corpora (constructed from public sources)
│   └── outputs/                  # SAE pre-trained weights
│
├── sae_feature_analysis/         # SAE feature analysis pipeline
│   ├── interpret_features/       # feature interpretation (span collection + annotation)
│   ├── identify_task_relevant_features/   # task-relevant feature identification
│   └── identify_missing_features/         # missing-feature discovery (coverage gap)
│
├── fac_synthesis/                # FAC synthesis pipeline
│   ├── step1_contrastive_pair_construction/      # Step-1: contrastive pair construction
│   └── step2_feature_covered_sample_synthesis/   # Step-2: feature-covered synthesis
│
└── training_scripts/             # Downstream training / evaluation scripts
    ├── toxicity_detection/
    ├── reward_modeling/
    ├── instruction_following/
    └── behavior_steering/
```

### Pre-training Sparse Autoencoders
Most of the scripts for SAE pretraining are located in `sae_pretrain/`. To pre-train SAEs, run the following commands:

```bash
```bash
# Step-1: Collect hidden activations from the backbone LLM (e.g., layer 16)
python create_actvs_uni.py 0 0 1 meta-llama/Llama-3.1-8B-Instruct 16

# Step-2: Train SAEs on the target layer (e.g., layer 16)
python train_SAEs.py 0 16 meta-llama/Llama-3.1-8B-Instruct /sae_input/prompt_actvs_l16
```

### Analyzing the features of SAE
Feature analysis scripts are located in `sae_feature_analysis/`. To group activation spans and generate human-readable feature interpretations, run:

```bash
# Step-1: Group extracted activation spans
python groupby_textspans.py /xxx/threshold_0.0

# Step-2: Annotate feature explanations based on grouped spans
python annotate_explanations.py /xxx/threshold_0.0.tsv

# Step-3: Identify task-relevant features from the explanations
python annotate_toxicity.py /xxx/threshold_0.0_explained.tsv

# Step-4: Identify missing features via FAC analysis
python identify_fac.py anchor_features.tsv (complete) task_features.tsv (currently available)
```

### Coverage-guided data synthesis
Coverage-guided synthesis scripts are located in fac_synthesis/. To generate synthetic queries, run

```bash
# Step-1 (1): Contrastive Pair Construction
python generate_data_llama_r1.py \
  --features xxx.tsv \
  --out xxx \
  --temperature 0.8
# Step-1 (2): Feature-Covered Sample Synthesis
python analyze_step1_synthetic_data.py
python merge_step1_failed_cases.py

# Step-2: Feature-Covered Sample Synthesis
python generate_data_llama_r2.py \
  --features xxx.tsv \
  --out xxx \
  --temperature 0.8
```

---

## Acknowledgements

In the evaluation stage, our downstream training and testing scripts are adapted from the following open-source repositories:

- [RLHF-Reward-Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling)
- [CAA](https://github.com/nrimsky/CAA)
- [LLaMAFactory](https://github.com/hiyouga/LLaMAFactory)

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
