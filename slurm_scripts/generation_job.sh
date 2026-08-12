#!/bin/bash
#SBATCH --job-name=generation_test          # Name des Jobs
#SBATCH --partition=gpu_a100_short        # Partition für Single-GPU Jobs
#SBATCH --nodes=1                    # 1 Rechenknoten
#SBATCH --ntasks-per-node=1          # 1 Prozess
#SBATCH --gres=gpu:4
#SBATCH --mem=160gb                   # Etwas Puffer für das 8B Modell beim Laden
#SBATCH --time=00:30:00              # 30 Minuten Zeitlimit
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
echo "Starting Generation."

python -u $WS_PATH/code/FAC-Synthesis/generation_play_ground/run_llama31_prompt.py

echo "Script finished with exit code: $?"