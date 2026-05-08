#!/bin/bash
#SBATCH --job-name=sae_spans_cpu     # Name des Jobs
#SBATCH --partition=dev_cpu              # CPU-Partition auf bwUniCluster
#SBATCH --nodes=1                    # 1 Rechenknoten
#SBATCH --ntasks-per-node=1          # 1 Prozess
#SBATCH --cpus-per-task=4            # Anzahl CPU-Kerne
#SBATCH --mem=45gb                   # Puffer fuer das Modell und die Verarbeitung
#SBATCH --time=00:20:00              # Zeitlimit fuer den Testlauf
#SBATCH --output=%x-%j.out           # Log-Dateien direkt im Arbeitsverzeichnis
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "Loading environment from start_llama.sh..."
source "$(ws_find master_thesis_experiment)/start_llama.sh"
echo "Workspace Path: $WS_PATH"
echo ""

DEVICE="${DEVICE:-auto}"
if [[ "$DEVICE" == "auto" ]]; then
    DEVICE="$(python - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
fi

echo "Selected device: $DEVICE"

export LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-$WS_PATH/models/llama-3.1-8b}"
if [[ ! -d "$LLAMA_MODEL_PATH" ]]; then
	echo "ERROR: Local Llama model directory not found: $LLAMA_MODEL_PATH"
	exit 1
fi

SAE_PATH="${SAE_PATH:-$WS_PATH/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth}"
if [[ ! -f "$SAE_PATH" && -f "$WS_PATH/code/FAC-Synthesis/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth" ]]; then
	SAE_PATH="$WS_PATH/code/FAC-Synthesis/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth"
fi

DATA_PATH="${DATA_PATH:-$WS_PATH/code/FAC-Synthesis/feature_annotation_data/fineweb_edu/fineweb_edu_text.txt}"
if [[ ! -f "$DATA_PATH" && -f "$WS_PATH/code/FAC-Synthesis/feature_annotation_data/fineweb_edu/fineweb_edu_test.txt" ]]; then
	DATA_PATH="$WS_PATH/code/FAC-Synthesis/feature_annotation_data/fineweb_edu/fineweb_edu_test.txt"
fi

if [[ ! -f "$SAE_PATH" ]]; then
	echo "ERROR: SAE checkpoint not found: $SAE_PATH"
	exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
	echo "ERROR: Data file not found: $DATA_PATH"
	exit 1
fi

echo "====== ENVIRONMENT VARIABLES ======"
echo "HF_HOME: ${HF_HOME:-<unset>}"
echo "LLAMA_MODEL_PATH: $LLAMA_MODEL_PATH"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"
echo "DATA_PATH: $DATA_PATH"
echo "SAE_PATH: $SAE_PATH"
echo ""

echo "====== DIAGNOSTICS END ======"
echo ""

echo "Starting SAE feature extraction..."
cd "$WS_PATH/code/FAC-Synthesis/sae_feature_analysis/interpret_features"
python collect_spans.py "$DEVICE" llama 0 1 \
	--data-path "$DATA_PATH" \
	--threshold 0.0 \
	--sae-path "$SAE_PATH"

echo "Script finished with exit code: $?"
