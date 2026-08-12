#!/bin/bash
#SBATCH --job-name=fac_test          # Name des Jobs
#SBATCH --partition=gpu_a100_short       # Partition für Single-GPU Jobs
#SBATCH --nodes=1                    # 1 Rechenknoten
#SBATCH --ntasks-per-node=1          # 1 Prozess
#SBATCH --gres=gpu:1
#SBATCH --mem=30gb                   # Etwas Puffer für das 8B Modell beim Laden
#SBATCH --time=00:15:00              # 30 Minuten Zeitlimit
#SBATCH --output=logs/%j_out.txt     # Log-Dateien (Ordner 'logs' muss existieren)
#SBATCH --error=logs/%j_err.txt

echo "====== DIAGNOSTICS START ======"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "Requested GPUs: $SLURM_GPUS"
echo ""

# 1. Umgebung über dein existierendes Skript laden
echo "Loading environment from start_llama.sh..."
source "$(ws_find master_thesis_exp)/start_llama.sh"
echo "Workspace Path: $WS_PATH"
echo ""



# 4. Environment Variables
echo "====== ENVIRONMENT VARIABLES ======"
echo "HF_HOME: $HF_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "====== DIAGNOSTICS END ======"
echo ""

# 5. Den Python-Befehl ausführen
echo "Starting FAC test pipeline..."
# python $WS_PATH/code/FAC-Synthesis/fac_test_pipeline/run_fac_test_pipeline.py \
# --model-name $WS_PATH/models/llama-3.1-8b \
# --sae-ckpt-path $WS_PATH/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth \
# --prompts-json $WS_PATH/code/FAC-Synthesis/fac_test_pipeline/input/prompts_feature_description.json \
# --output-jsonl $WS_PATH/code/FAC-Synthesis/fac_test_pipeline/output/results_prompts_feature_description.jsonl

python $WS_PATH/code/FAC-Synthesis/sae_feature_analysis/interpret_features/smoke_test_consistency.py \
    --model-key llama \
    --sae-path $WS_PATH/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth \
    --device cuda

echo "Script finished with exit code: $?"