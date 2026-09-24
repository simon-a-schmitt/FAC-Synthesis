#!/bin/bash
#SBATCH --job-name=fac_get_active_features
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "====== DIAGNOSTICS START ======"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "Requested GPUs: 1"
echo ""

echo "Loading environment from start_llama.sh..."
source "$(ws_find master_thesis_exp)/start_llama.sh"
echo "Workspace Path: $WS_PATH"
echo ""

echo "====== ENVIRONMENT VARIABLES ======"
echo "HF_HOME: ${HF_HOME:-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"
echo "====== DIAGNOSTICS END ======"
echo ""

MODEL_PATH="$WS_PATH/models/llama-3.1-8b"
SAE_PATH="$WS_PATH/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth"

# Override INPUT_TSV / THRESHOLD when submitting, e.g.:
#   sbatch --export=ALL,INPUT_TSV=...,THRESHOLD=1.5 get_active_features_job.sh
INPUT_TSV="${INPUT_TSV:-$WS_PATH/code/FAC-Synthesis/data_synthesis/data/seed_groups/toxicity_detection/toxicity_seed_group_01.tsv}"
THRESHOLD="${THRESHOLD:-1.5}"

mkdir -p logs

python "$WS_PATH/code/FAC-Synthesis/active_feature_identification/get_active_features/run_get_active_features.py" \
    --model-name "$MODEL_PATH" \
    --sae-ckpt-path "$SAE_PATH" \
    --input-tsv "$INPUT_TSV" \
    --threshold "$THRESHOLD"

echo "Script finished with exit code: $?"
