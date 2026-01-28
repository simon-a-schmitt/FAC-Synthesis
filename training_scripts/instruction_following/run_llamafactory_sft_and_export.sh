#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3-8B"
DATASET="dataset"
DATASET_DIR=""
TEMPLATE="alpaca"

# Training output (LoRA adapters)
OUTPUT_DIR=""

# Export merged model
EXPORT_DIR=""

echo "[INFO] Starting SFT training..."
llamafactory-cli train \
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
  --max_samples 2000

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

