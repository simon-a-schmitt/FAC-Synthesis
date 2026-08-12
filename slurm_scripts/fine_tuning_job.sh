#!/bin/bash
#SBATCH --job-name=fine_tuning_job         # Name des Jobs
#SBATCH --partition=gpu_a100_short        # Partition für Single-GPU Jobs
#SBATCH --nodes=1                    # 1 Rechenknoten
#SBATCH --ntasks-per-node=1          # 1 Prozess
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb                   # Etwas Puffer für das 8B Modell beim Laden
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


set -euo pipefail

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
    export HF_HUB_OFFLINE=1

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


MODEL_NAME_OR_PATH="$WS_PATH/models/llama-3.1-8b"
DATASET="cti_vsp_bb_deepseek_03_min200"
DATASET_DIR="$WS_PATH/software/LLaMA-Factory/data"
TEMPLATE="llama3"

echo "====== DATASET DIAGNOSTICS ======"
echo "Dataset: ${DATASET}"
python3 - "${DATASET_DIR}" "${DATASET}" <<'PYEOF'
import json
import sys

dataset_dir, dataset_name = sys.argv[1], sys.argv[2]

with open(f"{dataset_dir}/dataset_info.json") as f:
    info = json.load(f)

entry = info.get(dataset_name)
if entry is None:
    print(f"FEHLER: Dataset '{dataset_name}' nicht in dataset_info.json gefunden.")
    sys.exit(0)

file_names = entry["file_name"]
if isinstance(file_names, str):
    file_names = [file_names]

total = 0
for fn in file_names:
    with open(f"{dataset_dir}/{fn}") as f:
        data = json.load(f)
    print(f"  {fn}: {len(data)} Beispiele")
    total += len(data)

print(f"Trainings-Set Größe (gesamt): {total}")
PYEOF
echo ""


# Training output (LoRA adapters)
OUTPUT_DIR="$WS_PATH/code/FAC-Synthesis/fine_tuning/cti_vsp_bb_deepseek_03_min200/lora"

# Export merged model
EXPORT_DIR="$WS_PATH/code/FAC-Synthesis/fine_tuning/cti_vsp_bb_deepseek_03_min200/merged"

EPOCHS=10
LORA_RANK=16
LORA_ALPHA=32
LORA_TARGET=q_proj,v_proj
SEED=1004



echo "[INFO] Starting SFT training..."
llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET}" \
  --dataset_dir "${DATASET_DIR}" \
  --template llama3 \
  --cutoff_len 1024 \
  --finetuning_type lora \
  --lora_target "${LORA_TARGET}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout 0.1 \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lr_scheduler_type linear \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --num_train_epochs "${EPOCHS}" \
  --bf16 \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --logging_steps 5 \
  --save_strategy no \
  --save_only_model \
  --plot_loss \
  --overwrite_output_dir \
  --report_to none


echo "[INFO] Training done."
# echo "[INFO] Exporting merged model (LoRA -> full weights)..."

# llamafactory-cli export \
#   --model_name_or_path "${MODEL_NAME_OR_PATH}" \
#   --adapter_name_or_path "${OUTPUT_DIR}" \
#   --template "${TEMPLATE}" \
#   --finetuning_type lora \
#   --export_dir "${EXPORT_DIR}" \
#   --export_size 5 \
#   --export_device cpu

# echo "[INFO] Export done: ${EXPORT_DIR}"


