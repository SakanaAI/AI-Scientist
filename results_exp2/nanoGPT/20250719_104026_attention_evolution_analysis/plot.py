import json
import os
import os.path as osp

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

# LOAD FINAL RESULTS:
datasets = ["shakespeare_char", "enwik8", "text8"]
folders = os.listdir("./")
final_results = {}
results_info = {}

# Get the list of runs and generate the color palette
runs = [f for f in folders if f.startswith("run") and osp.isdir(f)]
colors = generate_color_palette(len(runs))

for folder in runs:
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
        
        if val_losses:  # Only calculate if we have data
            mean_val_losses = np.mean(val_losses, axis=0)
            mean_train_losses = np.mean(train_losses, axis=0)
            sterr_val_losses = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))
            stderr_train_losses = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
        else:
            mean_val_losses = np.array([])
            mean_train_losses = np.array([])
            sterr_val_losses = np.array([])
            stderr_train_losses = np.array([])
            
        run_info[dataset]["val_loss"] = mean_val_losses
        run_info[dataset]["train_loss"] = mean_train_losses
        run_info[dataset]["val_loss_sterr"] = sterr_val_losses
        run_info[dataset]["train_loss_sterr"] = stderr_train_losses
    results_info[folder] = run_info

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baseline",
    "run_1": "Attention Tracking",
    "run_2": "Cross-Layer Analysis", 
    "run_3": "Generation Quality"
}

# Additional plot types for attention analysis
def plot_attention_metrics():
    """Plot attention metrics (entropy, span length) over training"""
    for dataset in datasets:
        plt.figure(figsize=(12, 8))
        
        # Plot attention entropy
        plt.subplot(2, 1, 1)
        plotted_any = False
        for i, run in enumerate(runs):
            if run not in labels:  # Skip runs not in our labels dict
                continue
            try:
                # Load attention metrics from saved files
                metrics_path = osp.join(run, f"attention_metrics_{dataset}.npy")
                if not osp.exists(metrics_path):
                    continue
                    
                attn_metrics = np.load(metrics_path, allow_pickle=True).item()
                iters = attn_metrics['iters']
                entropy = attn_metrics['entropy']
                plt.plot(iters, entropy, label=labels[run], color=colors[i])
                plotted_any = True
            except Exception as e:
                print(f"Error loading attention metrics for {run} {dataset}: {e}")
                
        if not plotted_any:
            plt.close()
            continue
        plt.title(f"Attention Entropy Evolution - {dataset}")
        plt.xlabel("Iteration")
        plt.ylabel("Entropy")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        # Plot attention span
        plt.subplot(2, 1, 2)
        for i, run in enumerate(runs[1:]):  # Skip baseline
            attn_metrics = np.load(osp.join(run, f"attention_metrics_{dataset}.npy"),
                                allow_pickle=True).item()
            span = attn_metrics['span']
            plt.plot(iters, span, label=labels[run], color=colors[i+1])
        plt.title(f"Attention Span Evolution - {dataset}")
        plt.xlabel("Iteration")
        plt.ylabel("Span Length")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(f"attention_metrics_{dataset}.png")
        plt.close()

def plot_layer_similarity():
    """Plot cross-layer attention similarity"""
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        for i, run in enumerate(runs[2:]):  # Only run_2 has layer similarity
            sim_data = np.load(osp.join(run, f"layer_similarity_{dataset}.npy"),
                             allow_pickle=True).item()
            iters = sim_data['iters']
            for layer_pair, sim in sim_data['similarity'].items():
                plt.plot(iters, sim, 
                        label=f"{labels[run]} {layer_pair}",
                        color=colors[i+2], alpha=0.7)
        plt.title(f"Cross-Layer Attention Similarity - {dataset}")
        plt.xlabel("Iteration")
        plt.ylabel("Similarity Score")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"layer_similarity_{dataset}.png")
        plt.close()

# Generate all plots (only if data exists)
if any(osp.exists(osp.join(run, f"attention_metrics_{dataset}.npy")) 
       for run in runs for dataset in datasets):
    plot_attention_metrics()
    
if any(osp.exists(osp.join(run, f"layer_similarity_{dataset}.npy")) 
       for run in runs for dataset in datasets):
    plot_layer_similarity()

# Plot 1: Line plot of training loss for each dataset across the runs with labels
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
