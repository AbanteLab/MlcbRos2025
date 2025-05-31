#%%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import sem, t

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

# boxplots per noise
def boxplots_per_noise(x,y):
    
    noise_levels = x['fluo_noise'].unique()
    lat_levels = x['latent_dim'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

    for i, noise in enumerate(noise_levels):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        x_noise = x[x['fluo_noise'] == noise]
        sns.boxplot(x='model', 
                    y=y, 
                    hue='latent_dim', 
                    data=x_noise, 
                    showfliers=False, 
                    palette=sns.color_palette("ch:s=.25,rot=-.25",len(lat_levels)),
                    ax=ax)
        ax.set_title(f'Noise Level: {noise}')
        ax.set_xlabel('Model')
        if col == 0:
            ax.set_ylabel(y)
        else:
            ax.set_ylabel('')
        ax.legend_.remove()  # Remove individual legends
        ax.tick_params(axis='x', rotation=90)

    # Hide any unused subplots
    for i in range(len(noise_levels), 4):
        row, col = divmod(i, 2)
        fig.delaxes(axes[row, col])

    # Add a single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Latent Dimension', loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.show()
    
    return fig

def scatter_with_ci(df, metric_x, metric_y, conf=0.95):
    grouped = df.groupby(['model', 'latent_dim'], observed=True)
    df_mean = grouped.mean(numeric_only=True).reset_index()

    # Compute standard error and confidence intervals
    n = grouped.size().values
    se_x = grouped[metric_x].sem().values
    se_y = grouped[metric_y].sem().values
    h_x = se_x * t.ppf((1 + conf) / 2., n - 1)
    h_y = se_y * t.ppf((1 + conf) / 2., n - 1)

    df_mean[f'{metric_x}_ci'] = h_x
    df_mean[f'{metric_y}_ci'] = h_y

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df_mean,
        x=metric_x,
        y=metric_y,
        hue='model',
        style='latent_dim',
        palette="pastel",
        s=100,
        alpha=0.8,
        ax=ax
    )

    # Add error bars
    for _, row in df_mean.iterrows():
        ax.errorbar(
            row[metric_x], row[metric_y],
            xerr=row[f'{metric_x}_ci'], yerr=row[f'{metric_y}_ci'],
            fmt='none', ecolor='gray', alpha=0.7, capsize=3, zorder=0
        )

    ax.set_title(f'Model Performance (Mean ± 95% CI): {metric_x} vs {metric_y}')
    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    ax.legend(title='Model / Latent Dim', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    
    return fig

# Load data
results_dir = '/pool01/projects/abante_lab/snDGM/mlcb_2025/results/benchmark_unimodal/round_2/'

# TODO: get from round 2!
with open(os.path.join(results_dir, 'results_summary_BFA.txt'), 'r') as f:
    df_time_effa_t = pd.read_csv(f, sep='\t')
    df_time_effa_t['model'] = 'BFA'
    
with open(os.path.join(results_dir, 'results_summary_model_FixedVarMlpVAE.txt'), 'r') as f:
    df_time_fixedvarmlpvae = pd.read_csv(f, sep='\t')
    df_time_fixedvarmlpvae['model'] = 'VAE'

with open(os.path.join(results_dir, 'results_summary_model_FixedVarSupMlpVAE.txt'), 'r') as f:
    df_time_fixedvarsupmlpvae = pd.read_csv(f, sep='\t')
    df_time_fixedvarsupmlpvae['model'] = 'SVAE'

with open(os.path.join(results_dir, 'results_summary_model_FixedVarSupMlpDenVAE.txt'), 'r') as f:
    df_time_FixedVarSupMlpDenVAE = pd.read_csv(f, sep='\t')
    df_time_FixedVarSupMlpDenVAE['model'] = 'DSVAE'

# with open(os.path.join(results_dir, 'results_summary_model_LearnedVarMlpVAE.txt'), 'r') as f:
#     df_time_learnedvarmlpvae = pd.read_csv(f, sep='\t')
#     df_time_learnedvarmlpvae['model'] = 'LVVAE'

# with open(os.path.join(results_dir, 'results_summary_model_LearnedVarSupMlpVAE.txt'), 'r') as f:
#     df_time_learnedvarsupmlpvae = pd.read_csv(f, sep='\t')
#     df_time_learnedvarsupmlpvae['model'] = 'LVVAE (Sup)'

# with open(os.path.join(results_dir, 'results_summary_model_LearnedVarSupMlpDenVAE.txt'), 'r') as f:
#     df_time_LearnedVarSupMlpDenVAE = pd.read_csv(f, sep='\t')
#     df_time_LearnedVarSupMlpDenVAE['model'] = 'DLVVAE (Sup)'

# with open(os.path.join(results_dir, 'results_summary_model_LearnedVarSupLstmVAE.txt'), 'r') as f:
#     df_time_learnedvarsuplstmvae = pd.read_csv(f, sep='\t')
    
# with open(os.path.join(results_dir, 'results_summary_model_LearnedVarLstmVAE.txt'), 'r') as f:
#     df_time_learnedvarlstmvae = pd.read_csv(f, sep='\t')

# merge dataframes
df = pd.concat([df_time_effa_t, df_time_fixedvarmlpvae, df_time_fixedvarsupmlpvae, 
                df_time_FixedVarSupMlpDenVAE],axis=0).reset_index(drop=True)

# order models
df['model'] = pd.Categorical(
    df['model'], 
    categories=['BFA', 'VAE', 'SVAE', 'DSVAE'], 
    ordered=True
)

# filter out noise levels 0 and 2.5
df = df[df['fluo_noise'].isin([0.5, 1.0, 1.5, 2.0])]

# latent dims
df = df[df['latent_dim'].isin([4, 8, 16, 32, 64, 128])]

# keep only rows with mae_x<10
# df = df[df['mae_x'] < 10]

# metrics to plot
# 'ari_train': [ari_train],
# 'ari_val': [ari_val],
# 'sil_train': [sil_train_firing],
# 'sil_val': [sil_val_firing],
# 'sil_train_batch': [sil_train_batch],
# 'sil_val_batch': [sil_val_batch],
# 'med_ent_firing_train': [med_ent_firing_tr],
# 'med_ent_batch_train': [med_ent_batch_tr],
# 'med_ent_firing_val': [med_ent_firing_val],
# 'med_ent_batch_val': [med_ent_batch_val],
# 'kbet_train': [kbet_train],
# 'kbet_val': [kbet_val]

# # Construct a table with average values for BA, Precision, Recall, F1 
# # with mean and standard error
# metrics = ['ba', 'precision', 'recall', 'f1']
# df_metrics = df.groupby(['model', 'latent_dim']).agg(
#     {metric: ['mean', sem] for metric in metrics}
# ).reset_index()

# # Flatten MultiIndex columns
# df_metrics.columns = ['_'.join(col).strip() for col in df_metrics.columns.values]

# # boxplots for different metrics
# p = boxplots_per_noise(df, 'mae_x')
# p = boxplots_per_noise(df, 'ari_val')
# p = boxplots_per_noise(df, 'sil_val')
# p = boxplots_per_noise(df, 'ba')
# p = boxplots_per_noise(df, 'precision')
# p = boxplots_per_noise(df, 'recall')
# p = boxplots_per_noise(df, 'f1')
# p = boxplots_per_noise(df, 'med_ent_firing_val')
# p = boxplots_per_noise(df, 'kbet_val')

# # scatter plots with confidence intervals
# scatter_with_ci(df, 'mae_x', 'ari_val')
# scatter_with_ci(df, 'mae_x', 'sil_val')
# scatter_with_ci(df, 'mae_x', 'ba')
# scatter_with_ci(df, 'mae_x', 'precision')
# scatter_with_ci(df, 'mae_x', 'recall')
# scatter_with_ci(df, 'mae_x', 'f1')
# scatter_with_ci(df, 'mae_x', 'med_ent_firing_val')
# scatter_with_ci(df, 'mae_x', 'kbet_val')

# fig 2b mlcb 2025
df_fig2 = df[df['fluo_noise'].isin([1])]
df_fig2 = df_fig2[df_fig2['model'].isin(['BFA', 'VAE', 'SVAE'])]
df_fig2['model'] = pd.Categorical(df_fig2['model'], categories=['BFA', 'VAE', 'SVAE'], ordered=True)
p = scatter_with_ci(df_fig2, 'sil_val', 'kbet_val')
p.savefig(f'{results_dir}/scatter_silval_kbetval.pdf', bbox_inches='tight')

# supp figure mae vs latent dim
lat_levels = df['latent_dim'].unique()
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x='model', y='mae_x', hue='latent_dim', data=df_fig2, showfliers=False, 
            palette=sns.color_palette("ch:s=.25,rot=-.25",len(lat_levels)), ax=ax)
fig.tight_layout()
fig.savefig(f'{results_dir}/boxplot_mae.pdf', bbox_inches='tight')
plt.show()

# %%
