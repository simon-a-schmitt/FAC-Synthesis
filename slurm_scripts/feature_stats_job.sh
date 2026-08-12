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
PROMPTS_JSON="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/input/prompts_cti_vsp_seed_group_03.json"
OUTPUT_BASE="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/output"
BASELINE_TSV="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_activation_baseline_agg.tsv"
FEAT_DESC_TSV="$WS_PATH/code/FAC-Synthesis/sae_feature_analysis/interpret_features/xxx/threshold_1.0/threshold_1.0.tsv"

CLASSIFICATION_JSONL="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/output/feature_classification_cti_vsp_seed_group_03.jsonl"
SEED_PROMPTS_JSON="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/input/prompts_cti_vsp_seed_group_03.json"
BB_JSON="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/seed_gen_feature_comparison/input/cti_vsp_bb_deepseek.json"
OUTPUT_JSONL="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/seed_gen_feature_comparison/output/cti_vsp_bb_deepseek_feature_evolution.jsonl"


INPUT_TSV="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_coverage/input/labeled_full_bb_deepseek_02_200.tsv"
OUTPUT_TSV="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_coverage/output/bb_deepseek_02_200_fc_content.tsv"

INPUT_JSON="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/input/cti_vsp_fg_examples.json"
OUTPUT_JSONL="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/output/cti_vsp_fg_feature_check.jsonl"


mkdir -p logs "$OUTPUT_BASE"



# FEATURE STATS

# python "$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/run_fac_test_pipeline_feature_stats.py" \
#     --model-name "$MODEL_PATH" \
#     --sae-ckpt-path "$SAE_PATH" \
#     --prompts-json "$PROMPTS_JSON" \
#     --baseline-tsv "$BASELINE_TSV" \
#     --feature-descriptions-tsv "$FEAT_DESC_TSV" \
#     --output-classification-jsonl "$OUTPUT_BASE/feature_classification_cti_vsp_seed_group_03_content.jsonl" \
#     --opening-phrase "CVE Description: "


# SYNTHETIC FEATURE ACTIVATION CHECK

# python "$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/run_synthetic_feature_activation_check.py" \
#     --model-name "$MODEL_PATH" \
#     --sae-ckpt-path "$SAE_PATH" \
#     --input-json "$INPUT_JSON" \
#     --baseline-tsv "$BASELINE_TSV" \
#     --output-jsonl "$OUTPUT_JSONL"


# SEED GEN EVOLUTION

# python "$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/seed_gen_feature_comparison/run_seed_gen_feature_comparison.py" \
#     --model-name "$MODEL_PATH" \
#     --sae-ckpt-path "$SAE_PATH" \
#     --classification-jsonl "$CLASSIFICATION_JSONL" \
#     --seed-prompts-json "$SEED_PROMPTS_JSON" \
#     --bb-json "$BB_JSON" \
#     --baseline-tsv "$BASELINE_TSV" \
#     --feature-descriptions-tsv "$FEAT_DESC_TSV" \
#     --output-jsonl "$OUTPUT_JSONL"


# FEATURE COVERAGE

# python "$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_coverage/run_feature_coverage.py" \
#     --model-name "$MODEL_PATH" \
#     --sae-ckpt-path "$SAE_PATH" \
#     --input-tsv "$INPUT_TSV" \
#     --baseline-tsv "$BASELINE_TSV" \
#     --output-tsv "$OUTPUT_TSV" \
#     --opening-phrase "CVE Description: "
    
    
    # --save-intermediate \
    # --output-jsonl "$OUTPUT_BASE/feature_stats_intermediate.jsonl"


# FEATURE COVERAGE FILTER

python $WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_coverage_filter/run_feature_coverage_filter.py \
    --model-name "$MODEL_PATH" \
    --sae-ckpt-path "$SAE_PATH" \
    --input-json "/pfs/work9/workspace/scratch/ka_ai3967-master_thesis_exp/code/FAC-Synthesis/fac_test_pipeline/feature_coverage_filter/input/cti_vsp_bb_pool_deepseek_1025.json" \
    --opening-phrase "CVE Description: " \
    --output-jsonl "feature_coverage_filter/output/cti_vsp_bb_pool_1025_fc_filter.jsonl."


echo "Script finished with exit code: $?"
