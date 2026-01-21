import json
import os
import os.path as osp

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# LOAD FINAL RESULTS:
datasets = ["shakespeare_char", "enwik8", "text8"]
folders = os.listdir("./")
final_results = {}
results_info = {}
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        results_dict = np.load(osp.join(folder, "all_results.npy"), allow_pickle=True).item()
        run_info = {}
        for dataset in datasets:
            run_info[dataset] = {}
            val_losses = []
            train_losses = []
            for k in results_dict.keys():
                if dataset in k and "val_info" in k:
                    run_info[dataset]["iters"] = [info["iter"] for info in results_dict[k]]
                    val_losses.append([info["val/loss"] for info in results_dict[k]])
                    train_losses.append([info["train/loss"] for info in results_dict[k]])
                mean_val_losses = np.mean(val_losses, axis=0)
                mean_train_losses = np.mean(train_losses, axis=0)
                if len(val_losses) > 0:
                    sterr_val_losses = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))
                    stderr_train_losses = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
                else:
                    sterr_val_losses = np.zeros_like(mean_val_losses)
                    stderr_train_losses = np.zeros_like(mean_train_losses)
                run_info[dataset]["val_loss"] = mean_val_losses
                run_info[dataset]["train_loss"] = mean_train_losses
                run_info[dataset]["val_loss_sterr"] = sterr_val_losses
                run_info[dataset]["train_loss_sterr"] = stderr_train_losses
        results_info[folder] = run_info

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baseline",
    "run_1": "Hidden State Similarity",
    "run_2": "Clustering Analysis", 
    "run_3": "Attention Patterns",
    "run_4": "Cross-Layer Analysis"
}


# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Training and Validation Loss Comparison
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        iters = results_info[run][dataset]["iters"]
        mean = results_info[run][dataset]["train_loss"]
        sterr = results_info[run][dataset]["train_loss_sterr"]
        plt.plot(iters, mean, label=labels[run], color=colors[i])
        plt.fill_between(iters, mean - sterr, mean + sterr, color=colors[i], alpha=0.2)

    plt.title(f"Training Loss Across Runs for {dataset} Dataset")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"train_loss_{dataset}.png")
    plt.close()

# Plot 2: Representation Similarity Analysis
plt.figure(figsize=(15, 10))
for i, dataset in enumerate(datasets):
    plt.subplot(2, 2, i+1)
    for run in runs:
        metrics = results_info[run][dataset]['cluster_metrics']
        # Skip plotting for baseline run_0 which doesn't have cluster metrics
        if run == 'run_0':
            continue
            
        # Only proceed if cluster_metrics exists and is not empty
        if results_info[run][dataset].get('cluster_metrics'):
            metrics = results_info[run][dataset]['cluster_metrics']
            if metrics and len(metrics) > 0:  # Check metrics exists and has data
                iters = [m['iter'] for m in metrics if 'iter' in m]
                
                for layer in range(6):
                    if metrics[0].get(f'layer_{layer}_cross_sim') is not None:
                        cross_sim = [m.get(f'layer_{layer}_cross_sim', 0) for m in metrics]
                        plt.plot(iters, cross_sim, label=f'{labels[run]} Layer {layer}')
    
    plt.title(f"Cross-Layer Similarity ({dataset})")
    plt.xlabel("Iteration")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig("cross_layer_similarity.png")
plt.close()

# Plot 3: Attention Head Specialization
for dataset in datasets:
    plt.figure(figsize=(12, 6))
    for run in runs:
        metrics = results_info[run][dataset]['cluster_metrics']
        iters = [m['iter'] for m in metrics]
        
        # Plot cross-layer similarity
        for layer in range(1, 6):  # Layers 1-5
            cross_sim = [m.get(f'layer_{layer}_cross_sim', 0) for m in metrics]
            plt.plot(iters, cross_sim, label=f'{labels[run]} Layer {layer}')
    
    plt.title(f"Cross-Layer Hidden State Similarity ({dataset})")
    plt.xlabel("Iteration")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"cross_layer_{dataset}.png")
    plt.close()

# Plot 3: Attention Head Specialization
plt.figure(figsize=(15, 10))
for i, dataset in enumerate(datasets):
    plt.subplot(2, 2, i+1)
    for run in runs:
        metrics = results_info[run][dataset]['cluster_metrics']
        iters = [m['iter'] for m in metrics]
        
        # Skip plotting for baseline run_0 which doesn't have cluster metrics
        if run == 'run_0':
            continue
            
        # Only proceed if cluster_metrics exists and is not empty
        if results_info[run][dataset].get('cluster_metrics'):
            metrics = results_info[run][dataset]['cluster_metrics']
            if metrics and len(metrics) > 0:  # Check metrics exists and has data
                iters = [m['iter'] for m in metrics if 'iter' in m]
                
                for layer in range(6):
                    if metrics[0].get(f'layer_{layer}_attn_sim_mean') is not None:
                        sim_scores = [m.get(f'layer_{layer}_attn_sim_mean', 0) for m in metrics]
                        plt.plot(iters, sim_scores, label=f'{labels[run]} Layer {layer}')
    
    plt.title(f"Attention Head Similarity ({dataset})")
    plt.xlabel("Iteration")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig("attention_specialization.png")
plt.close()

# Plot 4: Cluster Quality Analysis
for dataset in datasets:
    plt.figure(figsize=(12, 6))
    for run in runs:
        metrics = results_info[run][dataset]['cluster_metrics']
        iters = [m['iter'] for m in metrics]
        
        # Plot attention similarity per layer
        for layer in range(6):  # Assuming 6 layers
            sim_scores = [m[f'layer_{layer}_attn_sim_mean'] for m in metrics]
            plt.plot(iters, sim_scores, label=f'{labels[run]} Layer {layer}')
    
    plt.title(f"Attention Head Similarity ({dataset})")
    plt.xlabel("Iteration")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"attention_sim_{dataset}.png")
    plt.close()

# Plot 4: Cluster Quality Analysis
plt.figure(figsize=(15, 10))
for i, dataset in enumerate(datasets):
    plt.subplot(2, 2, i+1)
    for run in runs:
        metrics = results_info[run][dataset]['cluster_metrics']
        iters = [m['iter'] for m in metrics]
        
        # Skip plotting for baseline run_0 which doesn't have cluster metrics
        if run == 'run_0':
            continue
            
        # Only proceed if cluster_metrics exists and is not empty
        if results_info[run][dataset].get('cluster_metrics'):
            metrics = results_info[run][dataset]['cluster_metrics']
            if metrics and len(metrics) > 0:  # Check metrics exists and has data
                iters = [m['iter'] for m in metrics if 'iter' in m]
                
                for layer in range(6):
                    if metrics[0].get(f'layer_{layer}_silhouette') is not None:
                        sil_scores = [m.get(f'layer_{layer}_silhouette', 0) for m in metrics]
                        plt.plot(iters, sil_scores, label=f'{labels[run]} Layer {layer}')
    
    plt.title(f"Hidden State Cluster Quality ({dataset})")
    plt.xlabel("Iteration")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig("cluster_quality.png")
plt.close()

# Plot 5: Training Dynamics Summary
plt.figure(figsize=(15, 10))
for i, metric in enumerate(['train_loss', 'val_loss']):
    for j, dataset in enumerate(datasets):
        plt.subplot(2, 3, i*3 + j + 1)
        for run in runs:
            data = results_info[run][dataset]
            plt.plot(data['iters'], data[metric], label=labels[run])
            plt.fill_between(data['iters'], 
                           data[metric] - data[f'{metric}_sterr'],
                           data[metric] + data[f'{metric}_sterr'],
                           alpha=0.2)
        plt.title(f"{metric.replace('_', ' ').title()} ({dataset})")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
plt.tight_layout()
plt.savefig("training_dynamics.png")
plt.close()
for dataset in datasets:
    plt.figure(figsize=(12, 6))
    for run in runs:
        metrics = results_info[run][dataset]['cluster_metrics']
        iters = [m['iter'] for m in metrics]
        
        # Plot silhouette scores per layer
        for layer in range(6):  # Assuming 6 layers
            sil_scores = [m[f'layer_{layer}_silhouette'] for m in metrics]
            plt.plot(iters, sil_scores, label=f'{labels[run]} Layer {layer}')
    
    plt.title(f"Hidden State Cluster Quality ({dataset})")
    plt.xlabel("Iteration")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"clustering_{dataset}.png")
    plt.close()

# Plot 2: Line plot of validation loss for each dataset across the runs with labels
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        iters = results_info[run][dataset]["iters"]
        mean = results_info[run][dataset]["val_loss"]
        sterr = results_info[run][dataset]["val_loss_sterr"]
        plt.plot(iters, mean, label=labels[run], color=colors[i])
        plt.fill_between(iters, mean - sterr, mean + sterr, color=colors[i], alpha=0.2)

    plt.title(f"Validation Loss Across Runs for {dataset} Dataset")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"val_loss_{dataset}.png")
    plt.close()
