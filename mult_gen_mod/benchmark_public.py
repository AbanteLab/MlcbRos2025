#%%
# Import depndencies
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


#%% 
################################################################################################################################
# Load results
################################################################################################################################

data_dir = '/data/results_summary_multimodal_ok.csv'
outdir = '/results/'

# Read the csv file
df = pd.read_csv(data_dir)

# remove index to avoid duplicates
df = df.reset_index(drop=True)

################################################################################################################################
# MAE Plots
################################################################################################################################

# 1x2 plot for MAE (calcium and RNA) by model and latent dimension
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE for calcium
sns.boxplot(
    data=df,
    x='model',
    y='mae_xcal',
    hue='latent_dim',
    showfliers=False,
    order=['indep_z', 'shared_z', 'hybrid_z'],
    palette=sns.color_palette("ch:s=.25,rot=-.25", len(df['latent_dim'].unique())),
    ax=axes[0]
)
axes[0].set_ylabel('MAE (Calcium)')
axes[0].set_xlabel('')  # Remove x-axis label
axes[0].legend_.remove()  # Remove legend from left plot

sns.boxplot(
    data=df,
    x='model',
    y='mae_xrna',
    hue='latent_dim',
    showfliers=False,
    order=['indep_z', 'shared_z', 'hybrid_z'],
    palette=sns.color_palette("ch:s=.25,rot=-.25", len(df['latent_dim'].unique())),
    ax=axes[1]
)

axes[1].set_ylabel('MAE (RNA)')
axes[1].set_xlabel('')  # Remove x-axis label
axes[1].legend(title='Latent Dim', bbox_to_anchor=(0.98, 1), loc='upper right')

fig.tight_layout()
# Save the figure
fig.savefig(os.path.join(outdir, 'mae_multimodal.pdf'), dpi=300, bbox_inches='tight')
plt.show()


##################################################################################################################################################################################################
#  Model shared_z_VAE vs hybrid_z_VAE
##################################################################################################################################################################################################
# Define order
model_order = ['shared_z', 'hybrid_z']

# Plot kbet for shared_z and hybrid_z models for each latent dimension
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df[df['model'].isin(['shared_z', 'hybrid_z'])],
    x='model',
    y='kbet_val',
    hue='latent_dim',
    showfliers=False,
    order=model_order,
    palette=sns.color_palette("ch:s=.25,rot=-.25", len(df['latent_dim'].unique())),
    ax=ax
)
ax.set_title('kBET of shared z')
ax.set_ylabel('kBET')
ax.legend(title='Latent Dim', bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()
# Save the figure
fig.savefig(os.path.join(outdir, 'kbet_shared_vs_hybrid.pdf'), dpi=300, bbox_inches='tight')
plt.show()


##################################################################################################################################################################################################
#  Model indep_z_VAE vs hybrid_z_VAE
##################################################################################################################################################################################################
# Plot kbet for indep_z and hybrid_z models for calcium and RNA in a 1x2 grid
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Define order
model_order = ['indep_z', 'hybrid_z']

# Calcium
sns.boxplot(
    data=df[df['model'].isin(model_order)],
    x='model',
    y='kbet_val_cal',
    hue='latent_dim',
    showfliers=False,
    order=model_order,
    palette=sns.color_palette("ch:s=.25,rot=-.25", len(df['latent_dim'].unique())),
    ax=axes[0]
)
axes[0].set_title('kBET Validation for Calcium')
axes[0].set_ylabel('kBET')
axes[0].set_xlabel('Model')
axes[0].legend(title='Latent Dim', bbox_to_anchor=(1.05, 1), loc='upper left')

# RNA
sns.boxplot(
    data=df[df['model'].isin(model_order)],
    x='model',
    y='kbet_val_rna',
    hue='latent_dim',
    showfliers=False,
    order=model_order,
    palette=sns.color_palette("ch:s=.25,rot=-.25", len(df['latent_dim'].unique())),
    ax=axes[1]
)
axes[1].set_title('kBET Validation for RNA')
axes[1].set_ylabel('kBET')
axes[1].set_xlabel('Model')
axes[1].legend(title='Latent Dim', bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()
# Save the figure
fig.savefig(os.path.join(outdir, 'kbet_indep_vs_hybrid.pdf'), dpi=300, bbox_inches='tight')
plt.show()

#%%
################################################################################################################################
# Sihouette vs kBET plot for MultiModMlpVAE models
################################################################################################################################

# Prepare data for combined plot
df_cal = df.copy()
df_cal['modality'] = 'cal'
df_cal['sil_val_mod'] = df_cal['sil_cal_val']
df_cal['kbet_val_mod'] = df_cal['kbet_val_cal']

df_rna = df.copy()
df_rna['modality'] = 'rna'
df_rna['sil_val_mod'] = df_rna['sil_rna_val']
df_rna['kbet_val_mod'] = df_rna['kbet_val_rna']

df_mod = pd.concat([df_cal, df_rna], axis=0)

# Compute means and CIs for each group
grouped = df_mod.groupby(['model', 'latent_dim', 'modality'], observed=True)
df_mean = grouped.mean(numeric_only=True).reset_index()
n = grouped.size().values
se_x = grouped['sil_val_mod'].sem().values
se_y = grouped['kbet_val_mod'].sem().values
h_x = se_x * t.ppf((1 + 0.95) / 2., n - 1)
h_y = se_y * t.ppf((1 + 0.95) / 2., n - 1)
df_mean['sil_val_mod_ci'] = h_x
df_mean['kbet_val_mod_ci'] = h_y

# Define marker shapes for latent_dim and colors for model
latent_shapes = {dim: marker for dim, marker in zip(sorted(df_mean['latent_dim'].unique()), ['o', 's', 'D', '^', 'v', 'P', '*', 'X'])}
model_palette = {
    'indep_z': '#66c2a5',
    'shared_z': '#fc8d62',
    'hybrid_z': '#8da0cb'
}
# Use more distinct outline colors for modalities (red, blue, black)
modality_outline = {'cal': '#e41a1c', 'rna': '#ff7f0e', 'shared': '#000000'}

# Make plots share y axis
fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'wspace': 0.1})
for _, row in df_mean.iterrows():
    marker = latent_shapes.get(row['latent_dim'], 'o')
    facecolor = model_palette.get(row['model'], '#cccccc')
    edgecolor = modality_outline.get(row['modality'], '#333333')
    ax[0].scatter(
        row['sil_val_mod'], row['kbet_val_mod'],
        s=100, marker=marker, facecolors=facecolor, edgecolors=edgecolor, linewidths=1.5, alpha=0.9,
        label=None
    )
    # Error bars
    ax[0].errorbar(
        row['sil_val_mod'], row['kbet_val_mod'],
        xerr=row['sil_val_mod_ci'], yerr=row['kbet_val_mod_ci'],
        fmt='none', ecolor=edgecolor, alpha=0.5, capsize=3, zorder=0
    )


# Legend for model (fill color)
model_handles = [
    mpatches.Patch(color=color, label=label)
    for label, color in zip(['indep_z', 'shared_z', 'hybrid_z'], [model_palette['indep_z'], model_palette['shared_z'], model_palette['hybrid_z']])
]

# Legend for latent_dim (marker shape)
latent_handles = [
    mlines.Line2D([], [], color='gray', marker=marker, linestyle='None', markersize=7, label=f'Latent {dim}', markerfacecolor='white', markeredgewidth=2)
    for dim, marker in latent_shapes.items()
]

# Legend for modality (outline color)
modality_handles = [
    mlines.Line2D([], [], color=edgecolor, marker='o', linestyle='None', markersize=7, markerfacecolor='white', markeredgewidth=2, label=mod.upper())
    for mod, edgecolor in modality_outline.items()
]

ax[0].set_title('Silhouette vs kBET\nShape: Latent Dim, Fill: Model, Outline: Modality')
ax[0].set_xlabel('Silhouette')
ax[0].set_ylabel('kBET')
ax[0].tick_params(axis='y', labelsize=10)  # Make y-tick labels smaller
ax[0].tick_params(axis='x', labelsize=10)  # Make x-tick labels smaller

# Plot sil_val vs kbet_val (shared) in ax[1]
grouped_shared = df.groupby(['model', 'latent_dim'], observed=True)
df_mean_shared = grouped_shared.mean(numeric_only=True).reset_index()

n_shared = grouped_shared.size().values
se_x_shared = grouped_shared['sil_val'].sem().values
se_y_shared = grouped_shared['kbet_val'].sem().values
h_x_shared = se_x_shared * t.ppf((1 + 0.95) / 2., n_shared - 1)
h_y_shared = se_y_shared * t.ppf((1 + 0.95) / 2., n_shared - 1)

df_mean_shared['sil_val_ci'] = h_x_shared
df_mean_shared['kbet_val_ci'] = h_y_shared
# Define marker shapes for latent_dim and colors for model
latent_shapes_shared = {dim: marker for dim, marker in zip(sorted(df_mean_shared['latent_dim'].unique()), ['o', 's', 'D', '^', 'v', 'P', '*', 'X'])}

for _, row in df_mean_shared.iterrows():
    marker = latent_shapes_shared.get(row['latent_dim'], 'o')
    facecolor = model_palette.get(row['model'], '#cccccc')
    ax[1].scatter(
        row['sil_val'], row['kbet_val'],
        s=100, marker=marker, facecolors=facecolor, alpha=0.9, edgecolors='black', linewidths=1.0,
        label=None
    )
    ax[1].errorbar(
        row['sil_val'], row['kbet_val'],
        xerr=row['sil_val_ci'], yerr=row['kbet_val_ci'],
        fmt='none', ecolor='black', alpha=0.3, capsize=3, zorder=0
    )

ax[1].set_title('Silhouette vs kBET (Shared)')
ax[1].set_xlabel('Silhouette')
ax[1].tick_params(axis='y', labelsize=10)  # Make y-tick labels smaller
ax[1].tick_params(axis='x', labelsize=10)  # Make x-tick labels smaller

first_legend = ax[1].legend(handles=model_handles, title='Model', bbox_to_anchor=(0.99, 1), loc='upper left', fontsize=7, title_fontsize=9, frameon=False)
second_legend = ax[1].legend(handles=latent_handles, title='Latent Dim', bbox_to_anchor=(1.0, 0.7), loc='upper left', fontsize=7, title_fontsize=9, frameon=False)
third_legend = ax[1].legend(handles=modality_handles, title='Modality', bbox_to_anchor=(1.0, 0.25), loc='upper left', fontsize=7, title_fontsize=9, frameon=False)
ax[1].add_artist(first_legend)
ax[1].add_artist(second_legend)
# Save rasterized figure
fig.savefig(os.path.join(outdir, 'silhouette_vs_kbet_multimodal.pdf'), dpi=300, bbox_inches='tight')
plt.show()
#%%