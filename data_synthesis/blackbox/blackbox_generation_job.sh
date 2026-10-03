#!/bin/bash
#SBATCH --job-name=bb_generation
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID | Node: $(hostname)"

source "$(ws_find master_thesis_exp)/start_llama.sh"

SCRIPT_DIR="$WS_PATH/code/FAC-Synthesis/data_synthesis/blackbox"

# Adjust per run.
MODEL="deepseek"  # llama | deepseek
DOMAIN="toxicity_detection"  # claudette_tos | cti_vsp | toxicity_detection
SEED_SET="k5"  # k5 | k10
SEED_GROUP="01"
N=400
ROUGE_THRESHOLD=0.7
PREFIX="bb_test"
MAX_CONCURRENT_REQUESTS=8

mkdir -p logs

python "$SCRIPT_DIR/run_generation.py" \
    --model "$MODEL" \
    --domain "$DOMAIN" \
    --seed-set "$SEED_SET" \
    --seed-group "$SEED_GROUP" \
    --n "$N" \
    --rouge-threshold "$ROUGE_THRESHOLD" \
    --prefix "$PREFIX" \
    --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
    --env-file "$WS_PATH/code/FAC-Synthesis/.env" \
    "$@"

echo "Script finished with exit code: $?"
