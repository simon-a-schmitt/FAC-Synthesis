#!/bin/bash
#SBATCH --job-name=labeling
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID | Node: $(hostname)"

source "$(ws_find master_thesis_exp)/start_llama.sh"

SCRIPT_DIR="$WS_PATH/code/FAC-Synthesis/data_synthesis/labeling"

# Adjust per run.
SOURCE="hybrid"   # blackbox | feature_guided | hybrid
DOMAIN="claudette_tos"
INPUT_JSON="claudette_tos_reporting_test_accepted.json"
MODEL="deepseek"               # gpt | deepseek
MAX_CONCURRENT_REQUESTS=2

mkdir -p logs

python "$SCRIPT_DIR/run_labeling.py" \
    --source "$SOURCE" \
    --domain "$DOMAIN" \
    --input-json "$INPUT_JSON" \
    --model "$MODEL" \
    --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
    --env-file "$WS_PATH/code/FAC-Synthesis/.env" \
    "$@"

echo "Script finished with exit code: $?"
