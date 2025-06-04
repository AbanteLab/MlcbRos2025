#%%

import os
import umap
import time
import psutil
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Pyro
import pyro

# PyTorch
import torch
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Our modules
from ca_sn_gen_models.utils import superprint
from ca_sn_gen_models.evaluation import evaluate_latent_svm,get_hdbscan_ari,compute_local_entropy,kBET
from ca_sn_gen_models.multimodal import MultiModSupMlpVAE_v2 as VaeModel

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superprint(f"Using device: {device}")

###############################################################################################
# arguments
###############################################################################################

# Create the parser
parser = argparse.ArgumentParser(description="MlpVAE training script")

# required arguments
parser.add_argument(
    "--fluo_noise",
    type=float,
    required=True,
    default=1.0,
    help="Noise level (default: 1.0)"
)

# optional arguments
parser.add_argument(
    "-l",
    "--latent_dim", 
    type=int,
    required=False,
    default=100,
    help="Dimension of latent space (default: 100)"
)

parser.add_argument(
    "-e",
    "--num_epochs", 
    type=int, 
    required=False,
    default=5000,
    help="Dimension of latent space (default: 5000)"
)

parser.add_argument(
    "-s",
    "--seed", 
    type=int, 
    required=False, 
    default=0,
    help="RNG seed (default: 0)"
)

parser.add_argument(
    "-r",
    "--rate", 
    type=float, 
    required=False, 
    default=1e-4,
    help="Learning rate (default: 1e-4)"
)

parser.add_argument(
    "--lreg", 
    type=float, 
    required=False, 
    default=1e5,
    help="Regularization in Z-space (default: 1e5)"
)

parser.add_argument(
    "-b",
    "--batch_size", 
    type=int, 
    required=False, 
    default=20000,
    help="Batch size for training (default: 20000)"
)

parser.add_argument(
    '--outdir', 
    type=str, 
    required=False,
    default='./output', 
    help='Folder to save output files'
)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
lr = args.rate
lreg = args.lreg
seed = args.seed
outdir = args.outdir
batch_size = args.batch_size
num_epochs = args.num_epochs
fluo_noise = args.fluo_noise
latent_dim = args.latent_dim

# # Comment out the following lines if you want to run the script without command line arguments
# seed = 1
# lreg = 1e8
# lr = 0.0001
# num_epochs = 2
# latent_dim = 64
# fluo_noise = 1.0
# batch_size = 1000
# outdir = "/pool01/projects/abante_lab/snDGM/mlcb_2025/results/benchmark_multimodal/supervised/"

#########################################################################################################
# Read in
#########################################################################################################
#%%
## Simulated calcium traces

superprint('Reading in data...')

## Metadata

# multimodal data directory
multi_dir = '/pool01/projects/abante_lab/snDGM/mlcb_2025/data/multimodal/'

# metadata
metadata_path_sims = f'{multi_dir}/metadata_merged.csv.gz'
meta_df = pd.read_csv(metadata_path_sims)

# create label combining group and sample
firing_labels = meta_df['firing_type'].values
group_labels = meta_df['group'].values
sample_labels = meta_df['sample'].values

# create group sample labels
group_sample_labels = np.array([f'{g}_{s}' for g, s in zip(group_labels, sample_labels)])

# Encode group_sample_labels using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(group_sample_labels)
encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)

# Convert to one-hot encoding
num_classes = len(np.unique(encoded_labels))
oh_encoded_labels = torch.nn.functional.one_hot(encoded_labels,num_classes=num_classes)

## Simulated RNA-seq data

# data directory
rna_dir = '/pool01/projects/abante_lab/snDGM/mlcb_2025/data/scrna_seq/'

# read in data
rna_data = pd.read_csv(f'{rna_dir}/scRNAseq_simulated_counts.csv', sep=',', header=0, index_col=0)

# keep only columns in metadata and transpose
rna_data = rna_data.loc[:, rna_data.columns.isin(meta_df['CellID'].values)].T

# create dataframe with rows 
rna_data = rna_data.loc[meta_df['CellID'].values]

# create tensor
xrna = torch.tensor(rna_data.values, dtype=torch.float32)

# normalize rows: log(x/sum(x)+1)
xrna = torch.log1p(xrna / xrna.sum(dim=1, keepdim=True))

## Simulated calcium traces

# included groups
group_set = [1, 2, 3]
sample_set = [0, 1, 2]

# directory
data_dir = '/pool01/projects/abante_lab/snDGM/mlcb_2025/data/caimg/normalized/'

# traces
data_list = []
for group in group_set:
    for sample in sample_set:
        data_path_sims = f'{data_dir}/sigma_{fluo_noise}/fluo_group_{group}_sample_{sample}.tsv.gz'
        data_sims = pd.read_csv(data_path_sims, sep='\t', header=None).values
        data_list.append(torch.tensor(data_sims, dtype=torch.float32))

# concatenate data
xcal = torch.cat(data_list, dim=0)

# subsample data
xcal = xcal[:,:10000]

# compute average signal for all xs
xcal_mean = xcal.mean(dim=1, keepdim=True)

# plot three examples from each firing type firing_labels
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
for i, firing_type in enumerate(np.unique(firing_labels)):
    firing_indices = np.where(firing_labels == firing_type)[0]
    random_indices = np.random.choice(firing_indices, 3, replace=False)
    for j, idx in enumerate(random_indices):
        axs[i].plot(xcal[idx], alpha=0.5)
        axs[i].set_title(f"Firing Type {firing_type} Example {idx}")
        axs[i].set_xlabel("Frequency")
        axs[i].set_ylabel("Amplitude")
plt.tight_layout()
plt.show()

#########################################################################################################
# NORMALIZE AND SPLIT DATA
#########################################################################################################

superprint('Splitting data...')

# split and create data loader
train_indices, val_indices = train_test_split(np.arange(xcal.shape[0]), test_size=0.2, random_state=seed)

# split data
xcal_train_data = xcal[train_indices]
xrna_train_data = xrna[train_indices]
xcal_val_data = xcal[val_indices]
xrna_val_data = xrna[val_indices]

# split labels
train_labels = oh_encoded_labels[train_indices]
val_labels = oh_encoded_labels[val_indices]

# create datasets
train_dataset = TensorDataset(xcal_train_data, xrna_train_data, train_labels)
val_dataset = TensorDataset(xcal_val_data, xrna_val_data, val_labels)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#########################################################################################################
# TRAIN
#########################################################################################################
#%%
superprint('Training model...')

# model path
model_suff = f'MultiModSupMlpVAE_seed_{seed}_latent_{latent_dim}_lreg_{lreg}'
model_path = f'{outdir}/models/{model_suff}.pt'

# Initialize the FA model
model = VaeModel(xcal.shape[1], xrna.shape[1], num_classes, latent_dim, device=device)

if not os.path.exists(model_path):
    
    superprint(f'Model does not exist at {model_path}. Starting training...')

    # Start the timer
    start_time = time.time()

    # clear cuda memory
    torch.cuda.empty_cache()

    # Clear Pyro parameters
    pyro.clear_param_store()

    # Train the model
    loss_tr,loss_val = model.train_model(train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=50, min_delta=1e-2, lreg=lreg)

    # Monitor peak RAM usage
    process = psutil.Process()

    # Get peak memory usage in MB
    peak_memory = process.memory_info().peak_wset / (1024 ** 2) if hasattr(process.memory_info(), 'peak_wset') else process.memory_info().rss / (1024 ** 2)
    superprint(f"Peak RAM usage: {peak_memory:.2f} MB")

    # stop timer
    end_time = time.time()
    training_time = end_time - start_time
    superprint(f'Training time: {training_time:.2f} seconds')

    # Save the model
    torch.save(model.state_dict(), model_path)
    superprint(f'Model saved to {model_path}')

    # plot training and validation loss
    fig, ax1 = plt.subplots()

    # Plot train loss on the first y-axis
    ax1.plot(range(len(loss_tr)), loss_tr, label='Train Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(range(len(loss_val)), loss_val, label='Validation Loss', color='red')
    ax2.set_ylabel('Validation Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add a title and show the plot
    plt.title('Train and Validation Loss')
    fig.tight_layout()
    plt.show()

else:
    
    superprint(f'Model already exists at {model_path}. Loading model...')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

#########################################################################################################
# Clean data
#########################################################################################################

superprint('Reading in zero noise data...')

# traces
data_list = []
for group in group_set:
    for sample in sample_set:
        data_path_sims = f'{data_dir}/sigma_0.0/fluo_group_{group}_sample_{sample}.tsv.gz'
        data_sims = pd.read_csv(data_path_sims, sep='\t', header=None).values
        data_list.append(torch.tensor(data_sims, dtype=torch.float32))

# concatenate data
x0 = torch.cat(data_list, dim=0)

# subsample data
x0 = x0[:,:10000]

# split into train and validation
x0_train = x0[train_indices]
x0_val = x0[val_indices]

#########################################################################################################
# INFERENCE
#########################################################################################################
#%%

superprint('Getting posterior estimates...')

# forward pass
zloc_tr, zloc_tr_cal, zloc_tr_rna, xhat_tr_cal, xhat_tr_rna = model.forward(train_loader)
zloc_val, zloc_val_cal, zloc_val_rna, xhat_val_cal, xhat_val_rna = model.forward(val_loader)


#########################################################################################################
# EMBEDDING
#########################################################################################################

# Perform UMAP dimensionality reduction
reducer = umap.UMAP(n_components=2)
Z_umap = reducer.fit_transform(zloc_tr.cpu().numpy())

# get labels for training data
group_labels = meta_df['group'].values[train_indices]
sample_labels = meta_df['sample'].values[train_indices]
firing_type_labels = meta_df['firing_type'].values[train_indices]
mean_firing_label = xcal_mean[train_indices].squeeze(1).cpu().numpy()

# Create a DataFrame for easier plotting with seaborn
umap_df = pd.DataFrame({
    'UMAP1': Z_umap[:, 0],
    'UMAP2': Z_umap[:, 1],
    'Group': group_labels,
    'Sample': sample_labels,
    'FiringType': firing_type_labels,
    'MeanFiring': mean_firing_label,
    'Modality': ['Calcium+RNA'] * zloc_tr.shape[0],
    'GroupSample': group_sample_labels[train_indices]
})

# Plot using seaborn for Group and Sample, without splitting by Modality
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='Group',
    style='Sample',
    palette='Set2',
    s=10
)
plt.title("UMAP of Latent Variables Z (Colored by Group and Sample)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Groups and Samples", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot using seaborn for Group and Sample, without splitting by Modality
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='FiringType',
    palette='Set2',
    s=10
)
plt.title("UMAP of Latent Variables Z (Colored by Group and Sample)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Modality and Firing", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot using seaborn for Group and Sample, without splitting by Modality
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='GroupSample',
    style='Modality',
    palette='Set2',
    s=30
)
plt.title("UMAP of Latent Variables Z (Colored by Group and Sample)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Modality and Firing", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# UMAP for calcium
umap_cal = umap.UMAP(n_components=2, random_state=seed)
zloc_tr_cal_umap = umap_cal.fit_transform(zloc_tr_cal.cpu().detach().numpy())
zloc_val_cal_umap = umap_cal.transform(zloc_val_cal.cpu().detach().numpy())
# UMAP for RNA
umap_rna = umap.UMAP(n_components=2, random_state=seed)
zloc_tr_rna_umap = umap_rna.fit_transform(zloc_tr_rna.cpu().detach().numpy())
zloc_val_rna_umap = umap_rna.transform(zloc_val_rna.cpu().detach().numpy())

# # Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))  
# Calcium
sns.scatterplot(
    x=zloc_tr_cal_umap[:, 0], 
    y=zloc_tr_cal_umap[:, 1], 
    hue=group_sample_labels[train_indices], 
    ax=axs[0], 
    palette="viridis", 
    alpha=0.7, 
    s=10
)
axs[0].set_title('Latent Space - Calcium (Train)')

sns.scatterplot(
    x=zloc_tr_rna_umap[:, 0], 
    y=zloc_tr_rna_umap[:, 1], 
    hue=group_sample_labels[train_indices], 
    ax=axs[1], 
    palette="viridis", 
    alpha=0.7, 
    s=10
)
axs[1].set_title('Latent Space - RNA (Train)')

plt.tight_layout()
plt.show()

##############################################################################################################
# Reconstruction of X
##############################################################################################################
#%%
superprint('Evaluating reconstruction...')

## Calcium data

# compute reconstruction error with noiseless data
mae_xrna = torch.nn.functional.l1_loss(xrna_val_data, xhat_val_rna.cpu(), reduction='mean').item()

## Calcium data

# plot a few examples of the data
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
random_indices = np.random.choice(xcal_val_data.shape[0], 4, replace=False)
for i, idx in enumerate(random_indices):
    axs[i].plot(x0_val[idx], label='GT', color='green')
    axs[i].plot(xhat_val_cal.cpu()[idx], label='Reconstructed FA', color='red', alpha=0.5)
    axs[i].set_title(f'Sample {idx}')
    axs[i].set_xlabel('Frequency')
    axs[i].set_ylabel('Amplitude')
    axs[i].set_ylim(-1, 1)
plt.legend()
plt.tight_layout()
plt.show()

# compute reconstruction error with noiseless data
mae_xcal = torch.nn.functional.l1_loss(x0_val, xhat_val_cal.cpu(), reduction='mean').item()

###############################################################################################
# Evaluation of the latent representation (RNA)
###############################################################################################


superprint('Evaluating latent representation (Cal+RNA)...')

## 1. silhouette score

sil_train = silhouette_score(zloc_tr.cpu(), firing_labels[train_indices])
sil_val = silhouette_score(zloc_val.cpu(), firing_labels[val_indices])
superprint(f'Silhouette training: {sil_train:.4f}')
superprint(f'Silhouette validation: {sil_val:.4f}')

# 2. train linear SVM classifier on zloc_tr and evaluate on zloc_val

ba,prec,rec,f1 = evaluate_latent_svm(zloc_tr.cpu(), firing_labels[train_indices], zloc_val.cpu(), firing_labels[val_indices])
superprint(f'Balanced accuracy: {ba:.4f}')
superprint(f'Precision: {prec:.4f}')
superprint(f'Recall: {rec:.4f}')
superprint(f'F1: {f1:.4f}')

## 3. ARI for HDBSCAN

clust_train, ari_train = get_hdbscan_ari(zloc_tr.cpu(), firing_labels[train_indices])
clust_val, ari_val = get_hdbscan_ari(zloc_val.cpu(), firing_labels[val_indices])
superprint(f'ARI train: {ari_train:.4f}')
superprint(f'ARI val: {ari_val:.4f}')

## 4. Local entropy

med_entropy_tr = compute_local_entropy(zloc_tr.cpu(), firing_labels[train_indices], k=100)
med_entropy_val = compute_local_entropy(zloc_val.cpu(), firing_labels[val_indices], k=100)
superprint(f'Median local entropy in training: {med_entropy_tr:.4f} bits')
superprint(f'Median local entropy in validation: {med_entropy_val:.4f} bits')

## 5. kBET
kbet_tr = kBET(zloc_tr.cpu(), group_sample_labels[train_indices])
kbet_val = kBET(zloc_val.cpu(), group_sample_labels[val_indices])
superprint(f"Rejection rate kBET (train): {kbet_tr:.3f}")
superprint(f"Rejection rate kBET (val): {kbet_val:.3f}")


###############################################################################################
# Evaluation of the latent representation (RNA)
###############################################################################################
        
superprint('Evaluating latent representation (RNA)...')

## 1. silhouette score

sil_train_rna = silhouette_score(zloc_tr_rna.cpu(), firing_labels[train_indices])
sil_val_rna = silhouette_score(zloc_val_rna.cpu(), firing_labels[val_indices])
superprint(f'Silhouette training (RNA): {sil_train_rna:.4f}')
superprint(f'Silhouette validation (RNA): {sil_val_rna:.4f}')

# 2. train linear SVM classifier on zloc_tr and evaluate on zloc_val

ba_rna,prec_rna,rec_rna,f1_rna = evaluate_latent_svm(zloc_tr_rna.cpu(), firing_labels[train_indices], zloc_val_rna.cpu(), firing_labels[val_indices])
superprint(f'Balanced accuracy (RNA): {ba_rna:.4f}')
superprint(f'Precision (RNA): {prec_rna:.4f}')
superprint(f'Recall (RNA): {rec_rna:.4f}')
superprint(f'F1 (RNA): {f1_rna:.4f}')

## 3. ARI for HDBSCAN

clust_train_rna, ari_train_rna = get_hdbscan_ari(zloc_tr_rna.cpu(), firing_labels[train_indices])
clust_val_rna, ari_val_rna = get_hdbscan_ari(zloc_val_rna.cpu(), firing_labels[val_indices])
superprint(f'ARI train (RNA): {ari_train_rna:.4f}')
superprint(f'ARI val (RNA): {ari_val_rna:.4f}')

## 4. Local entropy

med_entropy_tr_rna = compute_local_entropy(zloc_tr_rna.cpu(), firing_labels[train_indices], k=100)
med_entropy_val_rna = compute_local_entropy(zloc_val_rna.cpu(), firing_labels[val_indices], k=100)
superprint(f'Median local entropy in training (RNA): {med_entropy_tr_rna:.4f} bits')
superprint(f'Median local entropy in validation (RNA): {med_entropy_val_rna:.4f} bits')

## 5. kBET
kbet_tr_rna = kBET(zloc_tr_rna.cpu(), group_sample_labels[train_indices])
kbet_val_rna = kBET(zloc_val_rna.cpu(), group_sample_labels[val_indices])
superprint(f"Rejection rate kBET (train RNA): {kbet_tr_rna:.3f}")
superprint(f"Rejection rate kBET (val RNA): {kbet_val_rna:.3f}")



###############################################################################################
# Evaluation of the latent representation (Calcium)
###############################################################################################


superprint('Evaluating latent representation (Calcium)...')

## 1. silhouette score

sil_train_cal = silhouette_score(zloc_tr_cal.cpu(), firing_labels[train_indices])
sil_val_cal = silhouette_score(zloc_val_cal.cpu(), firing_labels[val_indices])
superprint(f'Silhouette training (Cal): {sil_train_cal:.4f}')
superprint(f'Silhouette validation (Cal): {sil_val_cal:.4f}')

# 2. train linear SVM classifier on zloc_tr and evaluate on zloc_val

ba_cal,prec_cal,rec_cal,f1_cal = evaluate_latent_svm(zloc_tr_cal.cpu(), firing_labels[train_indices], zloc_val_cal.cpu(), firing_labels[val_indices])
superprint(f'Balanced accuracy (Cal): {ba_cal:.4f}')
superprint(f'Precision (Cal): {prec_cal:.4f}')
superprint(f'Recall (Cal): {rec_cal:.4f}')
superprint(f'F1 (Cal): {f1_cal:.4f}')

## 3. ARI for HDBSCAN

clust_train_cal, ari_train_cal = get_hdbscan_ari(zloc_tr_cal.cpu(), firing_labels[train_indices])
clust_val_cal, ari_val_cal = get_hdbscan_ari(zloc_val_cal.cpu(), firing_labels[val_indices])
superprint(f'ARI train (Cal): {ari_train_cal:.4f}')
superprint(f'ARI val (Cal): {ari_val_cal:.4f}')

## 4. Local entropy

med_entropy_tr_cal = compute_local_entropy(zloc_tr_cal.cpu(), firing_labels[train_indices], k=100)
med_entropy_val_cal = compute_local_entropy(zloc_val_cal.cpu(), firing_labels[val_indices], k=100)
superprint(f'Median local entropy in training (Cal): {med_entropy_tr_cal:.4f} bits')
superprint(f'Median local entropy in validation (Cal): {med_entropy_val_cal:.4f} bits')

## 5. kBET

kbet_tr_cal = kBET(zloc_tr_cal.cpu(), group_sample_labels[train_indices])
kbet_val_cal = kBET(zloc_val_cal.cpu(), group_sample_labels[val_indices])
superprint(f"Rejection rate kBET (train): {kbet_tr_cal:.3f}")
superprint(f"Rejection rate kBET (val): {kbet_val_cal:.3f}")


#########################################################################################################
# Save results
#########################################################################################################

superprint('Storing summary...')

# Create a dataframe with the required information
results_df = pd.DataFrame({
    'seed': [seed],
    'fluo_noise': [fluo_noise],
    'model': 'MultiModSupMlpVAE',
    'z_lreg': [lreg],
    'beta': [1.0],
    'lr': [lr],
    'time': [training_time],
    'ram': [peak_memory],
    'latent_dim': [latent_dim],
    'num_epochs': [len(loss_tr)],
    'last_train_loss': [loss_tr[-1]],
    'last_val_loss': [loss_tr[-1]],
    'mae_x': [mae_xcal],
    'mae_xrna': [mae_xrna],
    'ari_train': [ari_train],
    'ari_val': [ari_val],
    'ari_cal_train': [ari_train_cal],
    'ari_cal_val': [ari_val_cal],
    'ari_rna_train': [ari_train_rna],
    'ari_rna_val': [ari_val_rna],
    'sil_train': [sil_train],
    'sil_val': [sil_val],
    'sil_cal_train': [sil_train_cal],
    'sil_cal_val': [sil_val_cal],
    'sil_rna_train': [sil_train_rna],
    'sil_rna_val': [sil_val_rna],
    'ba': [ba],
    'ba_cal': [ba_cal],
    'ba_rna': [ba_rna],
    'prec': [prec],
    'prec_cal': [prec_cal],
    'prec_rna': [prec_rna],
    'recall': [rec],
    'recall_cal': [rec_cal],
    'recall_rna': [rec_rna],
    'f1': [f1],
    'f1_cal': [f1_cal],
    'f1_rna': [f1_rna],
    'med_entropy_tr': [med_entropy_tr],
    'med_entropy_val': [med_entropy_val],
    'med_entropy_tr_cal': [med_entropy_tr_cal],
    'med_entropy_val_cal': [med_entropy_val_cal],
    'med_entropy_tr_rna': [med_entropy_tr_rna],
    'med_entropy_val_rna': [med_entropy_val_rna],
    'kbet_train': [kbet_tr],
    'kbet_val': [kbet_val],
    'kbet_train_cal': [kbet_tr_cal],
    'kbet_val_cal': [kbet_val_cal],
    'kbet_train_rna': [kbet_tr_rna],
    'kbet_val_rna': [kbet_val_rna]
})

#%%

# Append the dataframe to a text file
results_file = f'{outdir}/results_summary_model_MultiModSupMlpVAE.txt'
header = not os.path.exists(results_file)
results_df.to_csv(results_file, mode='a', header=header, index=False, sep='\t')
superprint(f'Results saved to {results_file}')
