#!/bin/bash
#SBATCH --job-name=generation_test          # Name des Jobs
#SBATCH --partition=gpu_a100_short        # Partition für Single-GPU Jobs
#SBATCH --nodes=1                    # 1 Rechenknoten
#SBATCH --ntasks-per-node=1          # 1 Prozess
#SBATCH --gres=gpu:4
#SBATCH --mem=300gb                   # Etwas Puffer für das 8B Modell beim Laden
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

_return_or_exit() {
    local code="$1"
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        return "$code"
    fi
    exit "$code"
}

# 1. Pfad automatisch finden
if [[ -z "${WS_PATH:-}" ]]; then
    if ! command -v ws_find >/dev/null 2>&1; then
        echo "FEHLER: 'ws_find' ist nicht im PATH verfügbar."
        _return_or_exit 1
    fi
    export WS_PATH="$(ws_find master_thesis_exp)"
fi

if [[ ! -d "$WS_PATH" ]]; then
    echo "FEHLER: Workspace-Pfad '$WS_PATH' existiert nicht."
    _return_or_exit 1
fi

# 2. Mamba/Conda Shell-Funktionen laden
# Wir prüfen erst, ob der Pfad existiert
if [ -f "$WS_PATH/software/miniforge3/bin/mamba" ]; then
    export MAMBA_ROOT_PREFIX="$WS_PATH/software/miniforge3"
    if [[ ! -f "$WS_PATH/software/miniforge3/etc/profile.d/mamba.sh" ]]; then
        echo "FEHLER: Mamba-Profilskript fehlt unter '$WS_PATH/software/miniforge3/etc/profile.d/mamba.sh'."
        _return_or_exit 1
    fi

    source "$WS_PATH/software/miniforge3/etc/profile.d/mamba.sh"

    if [[ ! -d "$WS_PATH/software/envs/ft_pipeline_env" ]]; then
        echo "FEHLER: Conda-Umgebung fehlt unter '$WS_PATH/software/envs/ft_pipeline_env'."
        _return_or_exit 1
    fi
    
    # 3. Umgebung aktivieren
    mamba activate "$WS_PATH/software/envs/ft_pipeline_env"

    if [[ "$(realpath "$CONDA_PREFIX")" != "$(realpath "$WS_PATH/software/envs/ft_pipeline_env")" ]]; then
        echo "FEHLER: 'mamba activate' hat nicht die erwartete Umgebung aktiviert."
        echo "Aktives CONDA_PREFIX: ${CONDA_PREFIX:-<leer>}"
        echo "Erwartet:             $WS_PATH/software/envs/ft_pipeline_env"
        _return_or_exit 1
    fi

    # 4. Pfade für Hugging Face setzen
    export HF_HOME="$WS_PATH/hf_cache"
    export HF_DATASETS_CACHE="$WS_PATH/hf_cache/datasets"

    # Env-eigene libstdc++ vor der (älteren) System-libstdc++ suchen lassen,
    # sonst schlägt der Import von z.B. optree mit GLIBCXX-Versionsfehlern fehl.
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

    echo "----------------------------------------------------"
    echo "Fine-Tuning Umgebung erfolgreich geladen!"
    echo "Workspace: $WS_PATH"
    echo "----------------------------------------------------"
else
    echo "FEHLER: Miniforge wurde nicht unter $WS_PATH/software/miniforge3 gefunden."
    _return_or_exit 1
fi


echo "Workspace Path: $WS_PATH"
echo ""



# 4. Environment Variables
echo "====== ENVIRONMENT VARIABLES ======"
echo "HF_HOME: $HF_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "====== GPU DIAGNOSTICS ======"
echo "Active python: $(command -v python)"
nvidia-smi || echo "FEHLER: nvidia-smi fehlgeschlagen (keine GPU sichtbar?)."
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"
echo ""

echo "====== DIAGNOSTICS END ======"
echo ""

cd "$WS_PATH"

set -euo pipefail

MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"
DATASET="casehold_gold_50"
DATASET_DIR="$WS_PATH/software/LLaMA-Factory/data"
TEMPLATE="llama3"


# Training output (LoRA adapters)
OUTPUT_DIR="$WS_PATH/code/FAC-Synthesis/fine_tuning/casehold_gold/lora"

# Export merged model
EXPORT_DIR="$WS_PATH/code/FAC-Synthesis/fine_tuning/casehold_gold/merged"


echo "[INFO] Starting SFT training..."
FORCE_TORCHRUN=1 llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET}" \
  --dataset_dir "${DATASET_DIR}" \
  --template "${TEMPLATE}" \
  --finetuning_type lora \
  --lora_target all \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 5.0 \
  --save_steps 100 \
  --logging_steps 10 \
  --bf16 \
  --max_samples 2000 \
  --val_size 0.1 \
  --do_eval \
  --eval_strategy steps \
  --eval_steps 100 \
  --per_device_eval_batch_size 4

echo "[INFO] Training done."
echo "[INFO] Exporting merged model (LoRA -> full weights)..."

llamafactory-cli export \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --adapter_name_or_path "${OUTPUT_DIR}" \
  --template "${TEMPLATE}" \
  --finetuning_type lora \
  --export_dir "${EXPORT_DIR}" \
  --export_size 5 \
  --export_device cpu

echo "[INFO] Export done: ${EXPORT_DIR}"

