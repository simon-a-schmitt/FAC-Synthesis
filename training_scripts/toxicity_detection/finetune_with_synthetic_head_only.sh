#!/bin/bash
set -e

for seed in 42 43 44
do
    echo "Running with seed: $seed"
    python finetune_with_synthetic_head_only.py --seed $seed
done
