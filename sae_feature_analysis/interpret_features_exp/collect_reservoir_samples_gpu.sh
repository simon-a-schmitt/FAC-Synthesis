#!/bin/bash
#SBATCH --job-name=fac_reservoir
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=120gb
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j_%t.out
#SBATCH --error=logs/%x_%j_%t.err

set -euo pipefail

NUM_WORKERS="${SLURM_NTASKS:-4}"
REQUESTED_GPUS="${SLURM_GPUS:-4}"

if [[ "${NUM_WORKERS}" -le 0 ]]; then
    echo "[ERROR] NUM_WORKERS must be > 0"
    exit 1
fi

echo "====== DIAGNOSTICS START ======"
echo "Job ID: ${SLURM_JOB_ID:-<none>}"
echo "Job Name: ${SLURM_JOB_NAME:-<none>}"
echo "Partition: ${SLURM_JOB_PARTITION:-<none>}"
echo "Node: $(hostname)"
echo "Requested GPUs: ${REQUESTED_GPUS}"
echo "Workers: ${NUM_WORKERS}"
echo ""

echo "Loading environment from start_llama.sh..."
source "$(ws_find master_thesis_exp)/start_llama.sh"
export WS_PATH
echo "Workspace Path: ${WS_PATH}"
echo ""

echo "====== ENVIRONMENT VARIABLES ======"
echo "HF_HOME: ${HF_HOME:-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"
echo "====== DIAGNOSTICS END ======"
echo ""

DATA_PATH="${WS_PATH}/code/FAC-Synthesis/feature_annotation_data/fineweb_edu/fineweb_edu_sample.txt"
MODEL_PATH="${WS_PATH}/models/llama-3.1-8b"
SAE_PATH="${WS_PATH}/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth"
OUT_BASE="${WS_PATH}/code/FAC-Synthesis/sae_feature_analysis/interpret_features_exp/xxx/reservoir_samples_4gpu"

mkdir -p logs "${OUT_BASE}"

export OUT_BASE DATA_PATH MODEL_PATH SAE_PATH

echo "Starting 4-way reservoir sampling..."
srun --ntasks="${NUM_WORKERS}" --gpus-per-task=1 --cpu-bind=none bash -lc '
set -euo pipefail

TASK_ID="${SLURM_PROCID:-0}"
LOCAL_ID="${SLURM_LOCALID:-0}"
SHARD_COUNT="${SLURM_NTASKS:-4}"
SHARD_DIR="${OUT_BASE}/shard_${TASK_ID}"

mkdir -p "${SHARD_DIR}"

echo "Task ${TASK_ID} on local GPU ${LOCAL_ID} of ${SHARD_COUNT}"

python "${WS_PATH}/code/FAC-Synthesis/sae_feature_analysis/interpret_features_exp/collect_reservoir_samples.py" \
    --model-key llama \
    --sae-path "${SAE_PATH}" \
    --device cuda \
    --data-path "${DATA_PATH}" \
    --shard-id "${TASK_ID}" \
    --shard-count "${SHARD_COUNT}" \
    --seed "$((10000 + TASK_ID))" \
    --enable-slurm
' 

echo "Script finished with exit code: $?"