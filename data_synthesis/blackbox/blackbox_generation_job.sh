#!/bin/bash
#SBATCH --job-name=bb_generation
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID | Node: $(hostname)"

source "$(ws_find master_thesis_exp)/start_llama.sh"

MODEL="deepseek"

SCRIPT_DIR="$WS_PATH/code/FAC-Synthesis/data_synthesis/blackbox"

TEMPERATURE=1.0
TOP_P=0.95

# Adjust per run.
DOMAIN="claudette_tos"
SEED_SET="k5"   # k5 | k10
SEED_GROUP="01"
N=50
ROUGE_THRESHOLD=0.5
PREFIX="claudette_tos_bb_deepseek_prompt_verification"
MAX_CONCURRENT_REQUESTS=2

mkdir -p logs

python "$SCRIPT_DIR/run_generation.py" \
    --domain "$DOMAIN" \
    --seed-set "$SEED_SET" \
    --seed-group "$SEED_GROUP" \
    --n "$N" \
    --rouge-threshold "$ROUGE_THRESHOLD" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --prefix "$PREFIX" \
    --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
    --model "$MODEL" \
    --env-file "$WS_PATH/code/FAC-Synthesis/.env" \
    "$@"

echo "Script finished with exit code: $?"
