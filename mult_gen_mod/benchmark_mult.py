#%%
# Import dependencies
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import sem, t

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Set the style for seaborn
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlepad'] = 10
plt.rcParams['axes.labelpad'] = 5
plt.rcParams['font.family'] = 'Helvetica'
sns.set_context("notebook")
sns.set_style("ticks")

############################################################################################################################################
# Compare hybrid_z supervised with unimodal models
############################################################################################################################################

# Upload unimodal results
df_unimodal = pd.read_csv('/pool01/projects/abante_lab/snDGM/mlcb_2025/results/benchmark_unimodal/round_2/results_summary_model_FixedVarSupMlpVAE.txt', sep='\t')

df_unimodal['modality_type'] = 'unimodal'

# Reaname mae_x column to mae_xcal
df_unimodal = df_unimodal.rename(columns={'mae_x': 'mae_xcal'})#, 'kbet_val': 'kbet_val_cal'})
df_unimodal = df_unimodal.rename(columns={'med_ent_firing_val': 'med_entropy_val'})

# Filter by latent dim != 256
df_unimodal = df_unimodal[df_unimodal['latent_dim'] != 256]

# Filter by fluo_noise = 1
df_unimodal = df_unimodal[df_unimodal['fluo_noise'] == 1]


# Read the csv file for multimodal results
df_multimodal = pd.read_csv('/pool01/projects/abante_lab/snDGM/mlcb_2025/results/benchmark_multimodal/supervised/results_summary_multimodal_sup_combined.csv', sep=',')

df_multimodal['modality_type'] = 'multimodal'

# Filter by z_lreg = 1e7
df_multimodal = df_multimodal[df_multimodal['z_lreg'] == 1e7]


# Combine the two dataframes
df_combined = pd.concat([df_unimodal, df_multimodal], ignore_index=True)
# Drop index
df_combined = df_combined.reset_index(drop=True)

# kBET for calcium and RNA by model type and model
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Calcium
sns.boxplot(
    data=df_combined,
    x='modality_type',
    y='mae_xcal',
    hue='latent_dim',
    showfliers=False,
    palette=sns.color_palette("ch:s=.25,rot=-.25", len(df_combined['latent_dim'].unique())),
    ax=axes[0]
)
axes[0].set_title('MAE Validation for Calcium')
axes[0].set_ylabel('MAE')
axes[0].set_xlabel('Model')
axes[0].legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
# RNA   
sns.boxplot(
    data=df_combined,
    x='modality_type',
    y='ari_val',
    hue='latent_dim',
    showfliers=False,
    palette=sns.color_palette("ch:s=.25,rot=-.25", len(df_combined['latent_dim'].unique())),
    ax=axes[1]
)
axes[1].set_title('ARI Validation')
axes[1].set_ylabel('ARI')
axes[1].set_xlabel('Model')
axes[1].legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()

# %%
