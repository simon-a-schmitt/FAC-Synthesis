#!/bin/bash
#SBATCH --job-name=fac_feature_stats
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
PROMPTS_JSON="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/input/prompts_seed_pubmedqa.json"
OUTPUT_BASE="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/output"
BASELINE_TSV="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_activation_baseline_agg.tsv"
FEAT_DESC_TSV="$WS_PATH/code/FAC-Synthesis/sae_feature_analysis/interpret_features/xxx/threshold_1.0/threshold_1.0.tsv"

mkdir -p logs "$OUTPUT_BASE"

python "$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/run_fac_test_pipeline_feature_stats.py" \
    --model-name "$MODEL_PATH" \
    --sae-ckpt-path "$SAE_PATH" \
    --prompts-json "$PROMPTS_JSON" \
    --baseline-tsv "$BASELINE_TSV" \
    --feature-descriptions-tsv "$FEAT_DESC_TSV" \
    --output-classification-jsonl "$OUTPUT_BASE/feature_classification.jsonl" \
    --save-intermediate \
    --output-jsonl "$OUTPUT_BASE/feature_stats_intermediate.jsonl"

echo "Script finished with exit code: $?"
