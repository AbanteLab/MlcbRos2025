#!/bin/bash

set -x # Prints each command to the terminal as it is executed

# Distributions
outdir='/pool01/projects/abante_lab/snDGM/mlcb_2025/results/benchmark_unimodal/'
mkdir -p "$outdir"

# common parameters
seeds=(0 1 2 3 4 5 6 7 8 9)
latent_dims=(8 16 32 64 128 256)
fluo_noises=(0.5 1.0 1.5 2.0 2.5)

# Loop over seed values
for seed in "${seeds[@]}"
do
    # Loop over latent_dim dimensions
    for latent_dim in "${latent_dims[@]}"
    do
        # Parallelize over fluo_noise values
        for fluo_noise in "${fluo_noises[@]}"
        do
            python ../python/train_effa.py --seed "$seed" --fluo_noise "$fluo_noise" --outdir "$outdir" --latent_dim "$latent_dim"
        done
    done
done
