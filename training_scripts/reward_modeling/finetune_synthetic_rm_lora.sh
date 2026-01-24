#!/bin/bash

datasets=(
    # "xxx"
    # "xxx"
)

for dataset in "${datasets[@]}"
do
    safe_name=${dataset//\//_}
    
    for seed in 42 43 44
    do
        echo "Running dataset: $dataset with seed $seed..."
        
        python finetune_synthetic_rm_lora.py \
            --seed $seed \
            --train_set_path "$dataset" \
            --output_path "./models/$safe_name"
    done
done
