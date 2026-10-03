#!/bin/bash
#SBATCH --job-name=labeling
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

SCRIPT_DIR="$WS_PATH/code/FAC-Synthesis/data_synthesis/labeling"

# Adjust per run.
SOURCE="feature_guided"   # blackbox | feature_guided | hybrid
DOMAIN="toxicity_detection"  # claudette_tos | cti_vsp | toxicity_detection
INPUT_JSON="toxicity_fg_llama_d0_6_t0_0_accepted.json"
MODEL="gpt"               # gpt | deepseek
MAX_CONCURRENT_REQUESTS=8

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
