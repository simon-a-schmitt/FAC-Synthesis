#!/bin/bash

for seed in 42 43 44
do
    echo "Running with seed $seed..."
    python finetune_synthetic_rm_head_only.py --seed $seed
done
