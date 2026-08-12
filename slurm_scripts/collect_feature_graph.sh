#!/bin/bash
##############################################################################
# Multi-GPU feature-graph activation collection.
#
# Launches one rank per GPU via torchrun; each rank owns a disjoint,
# contiguous block of source documents and writes only to its own
# shards/rank{R}/ directory (no cross-rank IO, so ranks need no
# torch.distributed process group - torchrun is used purely as a process
# launcher here).
#
# Usage: sbatch run_collect.sh
#        (rerun the same command to resume - completed shards are skipped)
##############################################################################

#SBATCH --job-name=feature_graph_collect
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=180000mb
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

NUM_GPUS="${SLURM_GPUS:-4}"
RESUME_MODE="${RESUME_MODE:-auto}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============ Job Start ============"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loading environment from start_llama.sh..."
source "$(ws_find master_thesis_exp)/start_llama.sh"

if [[ -z "${WS_PATH:-}" ]]; then
    echo "[ERROR] WS_PATH not set. Aborting."
    exit 1
fi

SAE_PATH="${SAE_PATH:-$WS_PATH/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth}"
if [[ ! -f "$SAE_PATH" && -f "$WS_PATH/code/FAC-Synthesis/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth" ]]; then
    SAE_PATH="$WS_PATH/code/FAC-Synthesis/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth"
fi
DATA_PATH="${DATA_PATH:-$WS_PATH/code/FAC-Synthesis/feature_annotation_data/fineweb_edu/fineweb_edu_sample.txt}"
OUT_DIR="${OUT_DIR:-$WS_PATH/code/FAC-Synthesis/sae_feature_analysis/feature_graph/shards}"

if [[ ! -f "$SAE_PATH" ]]; then
    echo "[ERROR] SAE checkpoint not found: $SAE_PATH"
    exit 1
fi
if [[ ! -f "$DATA_PATH" ]]; then
    echo "[ERROR] Data file not found: $DATA_PATH"
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] SAE_PATH:  $SAE_PATH"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DATA_PATH: $DATA_PATH"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] OUT_DIR:   $OUT_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] NUM_GPUS:  $NUM_GPUS"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resume:    $RESUME_MODE"

cd "$WS_PATH/code/FAC-Synthesis/sae_feature_analysis/feature_graph" || exit 1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching $NUM_GPUS ranks via torchrun..."
torchrun --standalone --nproc_per_node="$NUM_GPUS" collect_feature_activations.py \
    --data-path "$DATA_PATH" \
    --sae-path "$SAE_PATH" \
    --sae-layer 16 \
    --out-dir "$OUT_DIR" \
    --resume "$RESUME_MODE" \
    --enable-slurm

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============ All ranks complete ============"

##############################################################################
# Fallback without torchrun (e.g. interactive node, no distributed launcher):
# manually pin one GPU per background process, matching the pattern used by
# ../interpret_features/collect_spans_gpu.sh:
#
#   for RANK in $(seq 0 $((NUM_GPUS - 1))); do
#       CUDA_VISIBLE_DEVICES=$RANK python collect_feature_activations.py \
#           --data-path "$DATA_PATH" --sae-path "$SAE_PATH" --sae-layer 16 \
#           --out-dir "$OUT_DIR" --resume "$RESUME_MODE" \
#           --rank "$RANK" --world-size "$NUM_GPUS" --local-rank 0 \
#           > "logs/collect_rank${RANK}.log" 2>&1 &
#   done
#   wait
#
# (--local-rank 0 there because CUDA_VISIBLE_DEVICES already restricts the
# process to a single GPU, which is then device index 0 within it.)
##############################################################################
