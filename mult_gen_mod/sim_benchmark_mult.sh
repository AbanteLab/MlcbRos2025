#!/bin/bash

set -x # Prints each command to the terminal as it is executed 

# Distributions
outdir='/pool01/projects/abante_lab/snDGM/mlcb_2025/results/benchmark_multimodal/'
mkdir -p "$outdir"

# common parameters
lregs=(1e7 1e8 1e9)
latent_dims=(4 8 16 32 64 128)
seeds=(0 1 2 3 4 5 6 7 8 9)

# Loop over models, lreg values, seeds, and latent_dim dimensions
for seed in "${seeds[@]}"
do
    for latent_dim in "${latent_dims[@]}"
    do
        for lreg in "${lregs[@]}"
        do
            python train_multimodal_sup.py --seed "$seed" --fluo_noise "1.0" --outdir "$outdir" --latent_dim "$latent_dim" --lreg "$lreg"
        done
    done
done
