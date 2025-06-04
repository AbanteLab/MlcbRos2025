# MlcbRos2025
Code used in Ros 2025 (submitted to MLCB 2025).

## Overview
---
The code in this repository shows how to load the data associated with the paper 'Data driven prediciton of battery cycle life before capacity degradation' by K.A. Severson, P.M. Attia, et al. The data is available at https://data.matr.io/1/. There you can also find more details about the data.

This repository contains a the codes used for training and benchmarking models mentioned in Ros 2025, including:

- Single neuron unimodal VAE and BFA trained on synthetic data
- Single neuron multimodal VAE models trained on synthetic data
- Single neuron unimodal VAE trained on real data

Models and utils files can be found in [this link](https://github.com/AbanteLab/CaGenNet).

## 📂 Folder structure
---
<pre><code>
├── gen_mod_ca_img/
|   └── train_dgm.py
|   └── train_effa.py
|   └── sim_benchmark_dgm.sh
|   └── sim_benchmark_effa.sh
│   └── benchmark.py
├── mult_gen_mod/
│   └── train_multimodal.py
|   └── sim_benchmark_mult.sh
|   └── benchmark_mult.py
├── real_data/
│   └── train_dgm_real_data.py
│   └── benchmark_real_data.sh
├── requirements.txt
└── README.md
</code></pre>

### Single neuron unimodal DGM

#### VAE models
Here one can find tools to execute and benchmark 6 different VAE architectures originated from combinations of the following features:
- **Learned/ Fixed Var**: The variance of the likelihood $p_\theta(x\mid z)$ is Learned/Fixed.
- **Supervised model**: The batch labels are concatenated with the input to remove batch effect in the latent representation.
- **Denoising model**: A random mask $m$ is applied at each epoch, concatenaed with the data ($[x,m]$) and fed into the encoder. The goal is to make the model more robust to missing data.

`train_dgm.py` is used for training the chosen out of the 6 models and evaluating the latent representation and reconstruction using different metrics.
`sim_benchmark_dgm.sh` runs `train_dgm.py` for specified parameters
`train_effa.py` is used to train Exponent Factor Analysis model with FALTAA
`sim_benchmark_effa.sh` runs `train_effa.py` for specified parameters
`benchmark.py` generates plots with representation and reconstruction metrics

### Single neuron multimodal DGM
This model presents an extension of the Supervised unimodal VAE (SVAE) by adding transcriptomic data as input, therefore building a multimodal SVAE (MSVAE). In this case, 2 encoders map each modality concatenated with the labels to their own latent space and a third encoder takes [x_cal, x_rna] and maps to a shared latent space. There are 2 decoders, which take [z_cal/rna, z_shared, y_cal/rna] as input and reconstruct x_cal/rna.

`train_multimodal.py` is used for training the MSVAE and evaluating the latent representation and reconstruction using different metrics.
`sim_benchmark_mult.sh` runs `train_multimodal.py` for specified parameters
`benchmark_mult.py` generates plots with representation and reconstruction metrics


## Data

Data available upon request.
- Real data from primary cortical neurons fro rat and mouse DIV 7 and 12 respectively.
- Simulated multimodal data: calcium traces simulated with the Izhikevic model and scRNA-seq matching data.

## Installation and use
To execute any code, first proceed with installation of [CaGenNet](https://github.com/AbanteLab/CaGenNet).

To execute any `train*.py` (first choose if executing a single time or for several parameter combinations). In the first case, comment argument parser and uncomment fixed parameters. In the second case:
1. Comment fixed parameters & uncomment argument parser
2. Set paramter loops in corresponding `.sh` file
3. Run `bash *.sh` on terminal from the directory

