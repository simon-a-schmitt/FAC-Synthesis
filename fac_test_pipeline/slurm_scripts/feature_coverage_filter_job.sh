#!/bin/bash
#SBATCH --job-name=fac_feature_coverage_filter
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
INPUT_JSON="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_coverage_filter/input/cti_vsp_bb_pool_deepseek_1025.json"
OUTPUT_JSONL="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_coverage_filter/output/cti_vsp_bb_pool_deepseek_1025_feature_magnitudes.jsonl"
BASELINE_TSV="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_activation_baseline_agg.tsv"

mkdir -p logs "$(dirname "$OUTPUT_JSONL")"

python "$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_coverage_filter/run_feature_coverage_filter.py" \
    --model-name "$MODEL_PATH" \
    --sae-ckpt-path "$SAE_PATH" \
    --input-json "$INPUT_JSON" \
    --baseline-tsv "$BASELINE_TSV" \
    --output-jsonl "$OUTPUT_JSONL"

echo "Script finished with exit code: $?"
