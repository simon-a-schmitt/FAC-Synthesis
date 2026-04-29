# FAC Test Pipeline (Scaffold)

This folder provides a lightweight test pipeline that is intentionally close to the style and building blocks in `sae_pretrain`:

- `UnifiedGenerator` from `sae_pretrain/generator_uni.py`
- Layer hook mounting from `sae_pretrain/llm_surgery_uni.py`
- `TopKSAE` loading from `sae_pretrain/autoencoder.py`

## What this scaffold does

1. Loads prompts from a JSON file.
2. Runs prompts through a Llama model checkpoint (default: Llama 3.1 8B Instruct).
3. Extracts hidden activations at decoder layer 16 (1-based index, configurable).
4. Maps the last-token activation into Top-K SAE latent space.
5. Writes prompt + latent vectors to a JSONL output file.

## Install dependencies

```bash
pip install -r fac_test_pipeline/requirements.txt
```

The dependencies are selected to work on both Windows and Linux. GPU support depends on your local PyTorch/CUDA setup.

## Strict local-only operation (cluster friendly)

This pipeline now runs in strict local-only mode:

1. Download the model once to shared/local storage (outside compute jobs).
2. Point `--model-name` to that local model directory.
3. Provide SAE via local `--sae-ckpt-path`.

Recommended environment variables in cluster jobs:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Optional cache location:

```bash
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
```

## Prompt file format

Use one of these formats:

- `[{"prompt": "..."}, {"prompt": "..."}]`
- `{"prompts": [{"prompt": "..."}]}`
- `["prompt a", "prompt b"]`

A small example is provided in `prompts_template.json`.

## Run example

### Linux/macOS

```bash
python fac_test_pipeline/run_fac_test_pipeline.py \
  --device cuda \
  --device-id 0 \
  --model-name /shared/models/llama-3.1-8b-instruct \
  --layer 16 \
  --prompts-json fac_test_pipeline/prompts_template.json \
  --output-jsonl fac_test_pipeline/outputs/latents.jsonl \
  --sae-ckpt-path /shared/sae/your_checkpoint.pth \
  --hf-cache-dir /scratch/$USER/hf_cache
```

### Windows (PowerShell)

```powershell
python fac_test_pipeline/run_fac_test_pipeline.py `
  --device cuda `
  --device-id 0 `
  --model-name D:\models\llama-3.1-8b-instruct `
  --layer 16 `
  --prompts-json fac_test_pipeline/prompts_template.json `
  --output-jsonl fac_test_pipeline/outputs/latents.jsonl `
  --sae-ckpt-path D:\sae\your_checkpoint.pth `
  --hf-cache-dir D:\hf_cache
```

## SLURM example (single GPU)

```bash
#!/bin/bash
#SBATCH --job-name=fac-offline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache

cd /path/to/FAC-Synthesis

python fac_test_pipeline/run_fac_test_pipeline.py \
  --device cuda \
  --device-id 0 \
  --dtype bfloat16 \
  --model-name /shared/models/llama-3.1-8b-instruct \
  --layer 16 \
  --prompts-json fac_test_pipeline/prompts_template.json \
  --output-jsonl fac_test_pipeline/outputs/latents.jsonl \
  --sae-ckpt-path /shared/sae/your_checkpoint.pth \
  --hf-cache-dir /scratch/$USER/hf_cache
```

## SAE checkpoint options

Only local file is supported:

- Local file: `--sae-ckpt-path path/to/your_checkpoint.pth`

## Notes

- Layer indexing follows the existing `mount_function` logic (1-based).
- The scaffold currently stores the **last token** latent representation for each prompt.
- If you later need full-sequence latents, this can be added with minimal changes.
