#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0

DATASETS=(
    # "xxx.tsv"
)

for dataset_path in "${DATASETS[@]}"
do
    echo "Running with dataset: $dataset_path"
    
    for seed in 42 43 44
    do
        echo "Running with seed: $seed"
        python finetune_with_synthetic_lora.py --seed $seed --negative_data_path "$dataset_path"
    done
done
