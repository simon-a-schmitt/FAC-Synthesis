#!/bin/bash
#SBATCH --job-name=fac_test
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

echo "====== DIAGNOSTICS START ======"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "Requested GPUs: ${SLURM_GPUS:-4}"
echo "SLURM_NTASKS: ${SLURM_NTASKS:-4}"
echo ""

echo "Loading environment from start_llama.sh..."
source "$(ws_find master_thesis_exp)/start_llama.sh"
echo "Workspace Path: $WS_PATH"
echo ""

echo "====== ENVIRONMENT VARIABLES ======"
echo "HF_HOME: ${HF_HOME:-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"
echo "====== DIAGNOSTICS END ======"
echo ""

export MODEL_PATH="$WS_PATH/models/llama-3.1-8b"
export SAE_PATH="$WS_PATH/models/sae_llama_l16/TopK7_l16_h4096_epoch3.pth"
export PROMPTS_JSON="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/input/prompts_seed_pubmedqa.json"
export OUTPUT_BASE="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/output"
export HTML_BASE="$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/html_output"

mkdir -p logs "$OUTPUT_BASE" "$HTML_BASE"

WORKER_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/fac_token_matrix_worker.XXXXXX.sh")"
cat > "$WORKER_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail

TASK_ID="${SLURM_PROCID}"
LOCAL_ID="${SLURM_LOCALID:-0}"
SHARD_COUNT="${SLURM_NTASKS:-4}"

export CUDA_VISIBLE_DEVICES="${LOCAL_ID}"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PROMPTS_SHARD="${WORK_DIR}/prompts_shard_${TASK_ID}.json"
OUTPUT_JSONL="${OUTPUT_BASE}/results_prompts_seed_pubmedqa_token_shard${TASK_ID}.jsonl"
HTML_DIR="${HTML_BASE}/shard_${TASK_ID}"

python - "${PROMPTS_JSON}" "${PROMPTS_SHARD}" "${TASK_ID}" "${SHARD_COUNT}" <<'PY'
import json
import sys

src = sys.argv[1]
dst = sys.argv[2]
task_id = int(sys.argv[3])
shard_count = int(sys.argv[4])

with open(src, "r", encoding="utf-8") as f:
    payload = json.load(f)

def shard_items(items):
    return [item for idx, item in enumerate(items) if idx % shard_count == task_id]

if isinstance(payload, list):
    out = shard_items(payload)
elif isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
    payload["prompts"] = shard_items(payload["prompts"])
    out = payload
else:
    raise ValueError("Unsupported prompts JSON format")

with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
PY

mkdir -p "${HTML_DIR}"

python "$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/run_fac_test_pipeline_log_odds.py" \
    --model-name "$MODEL_PATH" \
    --sae-ckpt-path "$SAE_PATH" \
    --prompts-json "$PROMPTS_SHARD" \
    --output-jsonl "$OUTPUT_JSONL" \
    --baseline-tsv "$WS_PATH/code/FAC-Synthesis/fac_test_pipeline/feature_activation_baseline_agg.tsv" \
    --feature-descriptions-tsv "$WS_PATH/code/FAC-Synthesis/sae_feature_analysis/interpret_features/xxx/threshold_1.0/threshold_1.0.tsv" \
    --emit-html \
    --html-dir "$HTML_DIR"

echo "Task ${TASK_ID} finished with exit code: $?"
EOF

chmod +x "$WORKER_SCRIPT"

echo "Starting 4 parallel tasks..."
srun --ntasks=4 --gpus-per-task=1 --cpu-bind=none "$WORKER_SCRIPT"

rm -f "$WORKER_SCRIPT"

echo "Script finished with exit code: $?"