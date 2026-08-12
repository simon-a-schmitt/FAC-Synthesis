#!/bin/bash
#SBATCH --job-name=fac_blackbox_features
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

INPUT_JSONL="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/input/prompts_gen_blackbox_casehold.jsonl"
OUTPUT_JSONL="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/output/features_gen_blackbox.jsonl"

mkdir -p logs "$(dirname "$OUTPUT_JSONL")"

# Resumable: re-running this job continues from the examples not yet present
# in $OUTPUT_JSONL, so it is safe to resubmit after hitting the time limit.
python "$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/run_fac_test_pipeline_blackbox_features.py" \
    --input-jsonl "$INPUT_JSONL" \
    --output-jsonl "$OUTPUT_JSONL"

echo "Script finished with exit code: $?"
