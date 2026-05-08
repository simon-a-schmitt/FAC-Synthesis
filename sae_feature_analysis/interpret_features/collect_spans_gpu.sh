#!/bin/bash

##############################################################################
# GPU-optimiertes SLURM Job Script für SAE Feature Extraction
# Target: H100 GPU Partition (gpu_h100)
# 
# Usage: sbatch collect_spans_gpu.sh
#        sbatch --array=0-3 collect_spans_gpu.sh  (für manuelle Worker-Verteilung)
##############################################################################

#SBATCH --job-name=sae_spans_gpu
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6           # 6 CPUs pro GPU (24 CPUs / 4 GPUs)
#SBATCH --mem=180000mb        
#SBATCH --time=02:00:00             
#SBATCH --output=%x_%j.out          # sae_spans_gpu_<jobid>.out
#SBATCH --error=%x_%j.err


set -euo pipefail

##############################################################################
# Konfiguration
##############################################################################

# Optimieren Sie diese Parameter für Ihren Usecase:
NUM_GPUS="${SLURM_GPUS:-4}"           # Anzahl der verfügbaren GPUs
TTL_GROUPS="${NUM_GPUS}"              # Ein Worker pro GPU
CHECKPOINT_EVERY=25                   # Snapshot alle N Samples
RESUME_MODE="${RESUME_MODE:-auto}"    # auto|never|require
THRESHOLDS=(2.0)                      # Aktivierungs-Thresholds zum Testen

##############################################################################
# 1. Umgebung laden
##############################################################################

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============ Job Start ============"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loading environment from start_llama.sh..."

source "$(ws_find master_thesis_experiment)/start_llama.sh"

if [[ -z "${WS_PATH:-}" ]]; then
    echo "[ERROR] WS_PATH not set. Aborting."
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Workspace Path: $WS_PATH"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SLURM Job ID: $SLURM_JOB_ID"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SLURM Task ID: ${SLURM_ARRAY_TASK_ID:-none (bulk run)}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Number of GPUs: $NUM_GPUS"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resume mode: $RESUME_MODE"
echo ""

##############################################################################
# 2. CUDA & Device Setup
##############################################################################

# Automatisches CUDA-Setup
DEVICE="cuda"
if ! python - 2>/dev/null <<'PY_CHECK'
import torch
if torch.cuda.is_available():
    print(f"CUDA Available. GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
    exit(0)
else:
    print("CUDA Not Available!")
    exit(1)
PY_CHECK
then
    echo "[WARN] CUDA check failed. Falling back to CPU."
    DEVICE="cpu"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Device: $DEVICE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA_DEVICE_ORDER: ${CUDA_DEVICE_ORDER}"
echo ""

##############################################################################
# 3. Modelle & Daten validieren
##############################################################################

LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-$WS_PATH/models/llama-3.1-8b}"
SAE_PATH="${SAE_PATH:-$WS_PATH/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth}"
DATA_PATH="${DATA_PATH:-$WS_PATH/code/FAC-Synthesis/feature_annotation_data/fineweb_edu/fineweb_edu_test.txt}"

# Fallback-Pfade
if [[ ! -f "$SAE_PATH" && -f "$WS_PATH/code/FAC-Synthesis/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth" ]]; then
    SAE_PATH="$WS_PATH/code/FAC-Synthesis/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth"
fi

if [[ ! -f "$DATA_PATH" && -f "$WS_PATH/code/FAC-Synthesis/feature_annotation_data/fineweb_edu/fineweb_edu_sample.txt" ]]; then
    DATA_PATH="$WS_PATH/code/FAC-Synthesis/feature_annotation_data/fineweb_edu/fineweb_edu_sample.txt"
fi

# Validierung
if [[ ! -d "$LLAMA_MODEL_PATH" ]]; then
    echo "[ERROR] Llama model not found: $LLAMA_MODEL_PATH"
    exit 1
fi

if [[ ! -f "$SAE_PATH" ]]; then
    echo "[ERROR] SAE checkpoint not found: $SAE_PATH"
    exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
    echo "[ERROR] Data file not found: $DATA_PATH"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ====== Environment Setup ======"
echo "HF_HOME: ${HF_HOME:-<unset>}"
echo "LLAMA_MODEL_PATH: $LLAMA_MODEL_PATH"
echo "SAE_PATH: $SAE_PATH"
echo "DATA_PATH: $DATA_PATH"
echo ""

##############################################################################
# 4. Arbeitsverzeichnis
##############################################################################

cd "$WS_PATH/code/FAC-Synthesis/sae_feature_analysis/interpret_features" || exit 1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Working directory: $(pwd)"
echo ""

##############################################################################
# 5. Parallele Worker starten (ein Worker pro GPU)
##############################################################################

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ====== Starting Workers ======"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Total Workers: $TTL_GROUPS"
echo ""

# Array für Background Process IDs
declare -a PIDS
FAILED=0

for THRESHOLD in "${THRESHOLDS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing threshold: $THRESHOLD"

    for SUBGROUP in $(seq 0 $((TTL_GROUPS - 1))); do
        # GPU-Zuordnung: Worker 0 → GPU 0, Worker 1 → GPU 1, etc.
        GPU_ID=$((SUBGROUP % NUM_GPUS))
        
        LOG_FILE="logs/collect_spans_gpu_group${SUBGROUP}_thr${THRESHOLD}.log"
        mkdir -p logs

        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Starting Worker $SUBGROUP on GPU $GPU_ID (log: $LOG_FILE)"

        # Worker in Background starten mit dedizierter GPU
        CUDA_VISIBLE_DEVICES=$GPU_ID python collect_spans.py \
            --device "$DEVICE" \
            --model-key "llama" \
            --subgroup "$SUBGROUP" \
            --ttlgroup "$TTL_GROUPS" \
            --data-path "$DATA_PATH" \
            --threshold "$THRESHOLD" \
            --sae-path "$SAE_PATH" \
            --checkpoint-every "$CHECKPOINT_EVERY" \
            --resume "$RESUME_MODE" \
            --enable-slurm \
            > "$LOG_FILE" 2>&1 &

        PIDS+=($!)
    done

    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   All workers started for threshold $THRESHOLD. Waiting for completion..."
    
    # Auf alle Worker warten
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}" || {
            EXIT_CODE=$?
            echo "[WARN] Worker ${PIDS[$i]} exited with code $EXIT_CODE"
            ((FAILED++))
        }
    done

    PIDS=()
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Threshold $THRESHOLD complete."
    echo ""
done

##############################################################################
# 6. Zusammenfassung
##############################################################################

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============ Job Complete ============"

if [[ $FAILED -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All workers completed successfully."
    exit 0
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $FAILED worker(s) failed."
    exit 1
fi
