import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp

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
    "run_1": "Multi-Style Adapter",
    "run_2": "Fine-tuned Multi-Style Adapter",
    "run_3": "Enhanced Style Consistency",
    "run_4": "Style Consistency Analysis",
}

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs))

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

# Plot 3: Bar plot of style consistency scores for each dataset across the runs
plt.figure(figsize=(12, 6))
x = np.arange(len(datasets))
width = 0.8 / len(runs)

for i, run in enumerate(runs):
    means = []
    stds = []
    for dataset in datasets:
        if 'style_consistency_scores' in final_results[run][dataset].get('means', {}):
            means.append(final_results[run][dataset]['means']['style_consistency_scores'].get('mean_consistency', 0))
            stds.append(final_results[run][dataset].get('stderrs', {}).get('style_consistency_scores', {}).get('mean_consistency', 0))
        else:
            means.append(0)
            stds.append(0)
    plt.bar(x + i*width, means, width, label=labels[run], yerr=stds, capsize=5)

plt.xlabel('Dataset')
plt.ylabel('Style Consistency Score')
plt.title('Style Consistency Scores Across Runs and Datasets')
plt.xticks(x + width*(len(runs)-1)/2, datasets)
plt.legend()
plt.tight_layout()
plt.savefig("style_consistency_scores.png")
plt.close()

# Plot 4: Bar plot of inference speed for each dataset across the runs
plt.figure(figsize=(12, 6))
x = np.arange(len(datasets))
width = 0.8 / len(runs)

for i, run in enumerate(runs):
    means = []
    stds = []
    for dataset in datasets:
        if 'avg_inference_tokens_per_second_mean' in final_results[run][dataset]['means']:
            means.append(final_results[run][dataset]['means']['avg_inference_tokens_per_second_mean'])
            stds.append(final_results[run][dataset]['stderrs'].get('avg_inference_tokens_per_second_mean', 0))
        else:
            means.append(0)
            stds.append(0)
    plt.bar(x + i*width, means, width, label=labels[run], yerr=stds, capsize=5)

plt.xlabel('Dataset')
plt.ylabel('Tokens per Second')
plt.title('Inference Speed Across Runs and Datasets')
plt.xticks(x + width*(len(runs)-1)/2, datasets)
plt.legend()
plt.tight_layout()
plt.savefig("inference_speed.png")
plt.close()
