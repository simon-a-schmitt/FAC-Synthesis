#!/bin/bash -l
#SBATCH --job-name=ma_synthesis_job
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            # 16 Kerne/GPU × 4
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --output=logs/%j_out.txt
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

cd $WS_PATH

# 5. Den Python-Befehl ausführen
echo "Starting Generation."

RUN_PATH="blackbox"
N_SAMPLES=12
OUT_FILE="code/FAC-Synthesis/ma_synthesis/output/blackbox.jsonl"

SAMPLES_BEFORE=$(grep -c '"accept": true' "$OUT_FILE" 2>/dev/null)
SAMPLES_BEFORE=${SAMPLES_BEFORE:-0}

START_TIME=$(date +%s)

python -u code/FAC-Synthesis/ma_synthesis/run.py --path "$RUN_PATH" --n "$N_SAMPLES" --out "$OUT_FILE" \
    --model-path "$WS_PATH/models/llama-3.1-70b" \
    --token-log "${SLURM_SUBMIT_DIR}/token_usage_log.json"
EXIT_CODE=$?

END_TIME=$(date +%s)

SAMPLES_AFTER=$(grep -c '"accept": true' "$OUT_FILE" 2>/dev/null)
SAMPLES_AFTER=${SAMPLES_AFTER:-0}
SAMPLES_GENERATED=$((SAMPLES_AFTER - SAMPLES_BEFORE))

ELAPSED_SECONDS=$((END_TIME - START_TIME))
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi

# 6. Genutzte GPU-Stunden und generierte Samples ins Log-File schreiben
# (liegt im Verzeichnis, aus dem das SLURM-Skript per sbatch übergeben wurde;
# ein Eintrag pro RUN_PATH, wird über mehrere Jobs hinweg aufaddiert)
python code/FAC-Synthesis/ma_synthesis/log_gpu_hours.py \
    --log-file "${SLURM_SUBMIT_DIR}/gpu_hours_log.json" \
    --run-path "$RUN_PATH" \
    --elapsed-seconds "$ELAPSED_SECONDS" \
    --num-gpus "$NUM_GPUS" \
    --samples "$SAMPLES_GENERATED" \
    --job-id "$SLURM_JOB_ID"

echo "Script finished with exit code: $EXIT_CODE"