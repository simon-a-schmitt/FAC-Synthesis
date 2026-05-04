#!/bin/bash
#SBATCH --job-name=sae_spans_cpu     # Name des Jobs
#SBATCH --partition=dev_cpu              # CPU-Partition auf bwUniCluster
#SBATCH --nodes=1                    # 1 Rechenknoten
#SBATCH --ntasks-per-node=1          # 1 Prozess
#SBATCH --cpus-per-task=4            # Anzahl CPU-Kerne
#SBATCH --mem=45gb                   # Puffer fuer das Modell und die Verarbeitung
#SBATCH --time=00:20:00              # Zeitlimit fuer den Testlauf
#SBATCH --output=logs/%j_out.txt     # Log-Dateien (Ordner 'logs' muss existieren)
#SBATCH --error=logs/%j_err.txt

set -euo pipefail

echo "Loading environment from start_llama.sh..."
source $(ws_find master_thesis_experiment)/start_llama.sh
echo "Workspace Path: $WS_PATH"
echo ""

echo "====== ENVIRONMENT VARIABLES ======"
echo "HF_HOME: $HF_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "====== DIAGNOSTICS END ======"
echo ""

mkdir -p logs

echo "Starting SAE feature extraction..."
cd $WS_PATH/code/FAC-Synthesis/code/sae_feature_analysis/interpret_features
python collect_spans.py cpu llama 0 1 \
	--data-path $WS_PATH/code/FAC-Synthesis/feature_annotation_data/fineweb_edu/fineweb_edu_text.txt \
	--threshold 0.0 \
	--sae-path $WS_PATH/code/FAC-Synthesis/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth

echo "Script finished with exit code: $?"
