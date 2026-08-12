#!/bin/bash
#SBATCH --job-name=benchmark_casehold          # Name des Jobs
#SBATCH --partition=gpu_a100_short             # Partition für Single-GPU Jobs
#SBATCH --nodes=1                              # 1 Rechenknoten
#SBATCH --ntasks-per-node=1                    # 1 Prozess
#SBATCH --gres=gpu:4
#SBATCH --mem=300gb                             # Etwas Puffer für das 8B Modell beim Laden
#SBATCH --time=00:30:00                        # 30 Minuten Zeitlimit
#SBATCH --output=logs/%j_out.txt               # Log-Dateien (Ordner 'logs' muss existieren)
#SBATCH --error=logs/%j_err.txt

export PYTHONUNBUFFERED=1

echo "====== DIAGNOSTICS START ======"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $(hostname)"
echo "Requested GPUs: $SLURM_GPUS"
echo ""

# 1. Umgebung über dein existierendes Skript laden
echo "Loading environment from start_llama.sh..."
source "$(ws_find master_thesis_exp)/start_llama.sh"
echo "Workspace Path: $WS_PATH"
echo ""

echo "====== PYTHON / CUDA DIAGNOSTICS ======"
which python
python - <<'PY'
import os
import torch

print("python_executable:", os.sys.executable)
print("torch_version:", torch.__version__)
print("torch_cuda_version:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"gpu_{i}_name:", props.name)
        print(f"gpu_{i}_capability:", f"{props.major}.{props.minor}")
else:
    print("gpu_status: CUDA is not visible from this Python environment")
PY

echo "====== NVIDIA-SMI ======"
nvidia-smi
echo ""

# 2. Environment Variables
echo "====== ENVIRONMENT VARIABLES ======"
echo "HF_HOME: $HF_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "====== DIAGNOSTICS END ======"
echo ""

# # 5. CaseHold
# echo "Starting Benchmark."

# cd $WS_PATH/code/FAC-Synthesis

# python benchmark_play_ground/run_casehold_benchmark.py --model-path $WS_PATH/models/llama-3.1-8b --data-tsv benchmarks/casehold/casehold_extract.tsv --mode plain   --max-prompts 1000 --device cuda --output-jsonl benchmark_play_ground/casehold_bench_plain.jsonl
# # --few-shot-tsv benchmarks/casehold/casehold_few_shot.tsv --icl-k 5
# echo "Script finished with exit code: $?"


# # 5. PubMedQA
# echo "Starting Benchmark."

# cd $WS_PATH/code/FAC-Synthesis

# python benchmark_play_ground/run_pubmedqa_benchmark.py --model-path $WS_PATH/models/llama-3.1-8b --lora-path $WS_PATH/code/FAC-Synthesis/fine_tuning/blackbox_gen_02/lora --data-tsv benchmarks/pubmedqa/pubmedqa_test_500.tsv --mode plain --few-shot-tsv benchmark_play_ground/pubmedqa_seed_group_01.tsv --icl-k 5 --max-prompts 1000 --device cuda --output-jsonl benchmark_play_ground/pubmedqa_bench_plain.jsonl
# # --few-shot-tsv benchmarks/casehold/casehold_few_shot.tsv --icl-k 5
# # --lora-path $WS_PATH/code/FAC-Synthesis/fine_tuning/casehold_gold_50/lora
# echo "Script finished with exit code: $?"


# 5. CTI-VSP
echo "Starting Benchmark."

cd $WS_PATH/code/FAC-Synthesis

python benchmark_play_ground/run_cti_vsp_benchmark.py --model-path $WS_PATH/models/llama-3.1-8b --lora-path $WS_PATH/code/FAC-Synthesis/fine_tuning/cti_vsp_bb_deepseek_03_min200/lora --data-tsv benchmarks/cti_vsp/cti_vsp_benchmark_test_500.tsv --mode fine_tuned --few-shot-tsv benchmark_play_ground/seed_examples/cti_vsp_benchmark_seed_group_03.tsv --icl-k 5 --max-prompts 1000 --device cuda --output-jsonl benchmark_play_ground/cti_vsp_benchmark_run_01/cti_vsp_bench_bb_deepseek_03_min200.jsonl --max-new-tokens 48 --resume
# --few-shot-tsv benchmarks/casehold/casehold_few_shot.tsv --icl-k 5
# --lora-path $WS_PATH/code/FAC-Synthesis/fine_tuning/casehold_gold_50/lora
echo "Script finished with exit code: $?"




