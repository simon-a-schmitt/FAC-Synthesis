#!/bin/bash
#SBATCH --job-name=fg_generation
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

echo "Job ID: $SLURM_JOB_ID | Node: $(hostname)"

source "$(ws_find master_thesis_exp)/start_llama.sh"

MODEL_PATH="$WS_PATH/models/llama-3.1-8b"
SAE_PATH="$WS_PATH/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth"
SCRIPT_DIR="$WS_PATH/code/FAC-Synthesis/data_synthesis/feature_guided"

# Adjust per run.
DOMAIN="claudette_tos"
SEED_SET="k5"   # k5 | k10
SEED_GROUP="01"
N=50
THRESHOLD=0.0
ROUGE_THRESHOLD=0.5
MODEL="llama"  # llama | deepseek
PREFIX="claudette_reporting_test"
MAX_CONCURRENT_REQUESTS=2

mkdir -p logs

python "$SCRIPT_DIR/run_generation.py" \
    --domain "$DOMAIN" \
    --seed-set "$SEED_SET" \
    --seed-group "$SEED_GROUP" \
    --n "$N" \
    --threshold "$THRESHOLD" \
    --rouge-threshold "$ROUGE_THRESHOLD" \
    --model "$MODEL" \
    --prefix "$PREFIX" \
    --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
    --model-name "$MODEL_PATH" \
    --sae-ckpt-path "$SAE_PATH" \
    --env-file "$WS_PATH/code/FAC-Synthesis/.env" \
    "$@"

echo "Script finished with exit code: $?"
