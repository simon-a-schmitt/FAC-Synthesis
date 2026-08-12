#!/bin/bash
#SBATCH --job-name=fac_feature_coverage
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
INPUT_TSV="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_coverage/input/cti_vsp_ft_200.tsv"
OUTPUT_TSV="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_coverage/output/cti_vsp_ft_200_max_magnitude.tsv"
BASELINE_TSV="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_activation_baseline_agg.tsv"

mkdir -p logs "$(dirname "$OUTPUT_TSV")"

python "$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_coverage/run_feature_coverage.py" \
    --model-name "$MODEL_PATH" \
    --sae-ckpt-path "$SAE_PATH" \
    --input-tsv "$INPUT_TSV" \
    --baseline-tsv "$BASELINE_TSV" \
    --output-tsv "$OUTPUT_TSV"

echo "Script finished with exit code: $?"
