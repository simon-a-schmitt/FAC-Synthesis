#!/bin/bash
#SBATCH --job-name=groupby_textspans_full
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120GB
#SBATCH --time=00:25:00
#SBATCH --output=logs/groupby_%j.log
#SBATCH --error=logs/groupby_%j.err

source "$(ws_find master_thesis_exp)/start_llama.sh"

if [[ -z "${WS_PATH:-}" ]]; then
    echo "[ERROR] WS_PATH not set. Aborting."
    exit 1
fi

echo "===== Processing full aggregated dataset ====="
echo "Hostname: $(hostname)"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

mkdir -p logs

cd "$WS_PATH/code/FAC-Synthesis/sae_feature_analysis/interpret_features"

# Ersetze /path/to/folder mit dem Ordner der deine full.tsv enthält
python groupby_textspans.py "xxx/threshold_1.0" \
    --output-path "xxx/threshold_1.0/threshold_1.0.tsv" \
    --checkpoint-every 500

echo "Completed at $(date)"