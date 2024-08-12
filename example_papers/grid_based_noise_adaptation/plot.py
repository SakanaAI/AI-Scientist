import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp
import pickle

# LOAD FINAL RESULTS:
datasets = ["circle", "dino", "line", "moons"]
folders = os.listdir("./")
final_results = {}
train_info = {}


def smooth(x, window_len=10, window='hanning'):
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        all_results = pickle.load(open(osp.join(folder, "all_results.pkl"), "rb"))
        train_info[folder] = all_results

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')  # You can change 'tab20' to other colormaps like 'Set1', 'Set2', 'Set3', etc.
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

# CREATE LEGEND
labels = {
    "run_0": "Baseline",
    "run_1": "10x10 Grid",
    "run_2": "20x20 Grid",
    "run_3": "Multi-scale Grid",
    "run_4": "Multi-scale + L1 Reg",
    "run_5": "Adjusted L1 Reg"
}

# Only plot the runs that are both in the labels dictionary and in the final_results
runs = [run for run in final_results.keys() if run in labels]
colors = generate_color_palette(len(runs))


# CREATE PLOTS


# Get the list of runs and generate the color palette
runs = list(final_results.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Line plot of training loss for each dataset across the runs with labels
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

for j, dataset in enumerate(datasets):
    row = j // 2
    col = j % 2
    for i, run in enumerate(runs):
        mean = train_info[run][dataset]["train_losses"]
        mean = smooth(mean, window_len=25)
        axs[row, col].plot(mean, label=labels[run], color=colors[i])
        axs[row, col].set_title(dataset)
        axs[row, col].legend()
        axs[row, col].set_xlabel("Training Step")
        axs[row, col].set_ylabel("Loss")

plt.tight_layout()
plt.savefig("train_loss.png")
plt.show()

# Plot 2: Visualize generated samples
# If there is more than 1 run, these are added as extra rows.
num_runs = len(runs)
fig, axs = plt.subplots(num_runs, 4, figsize=(14, 3 * num_runs))

for i, run in enumerate(runs):
    for j, dataset in enumerate(datasets):
        images = train_info[run][dataset]["images"]
        if num_runs == 1:
            axs[j].scatter(images[:, 0], images[:, 1], alpha=0.2, color=colors[i])
            axs[j].set_title(dataset)
        else:
            axs[i, j].scatter(images[:, 0], images[:, 1], alpha=0.2, color=colors[i])
            axs[i, j].set_title(dataset)
    if num_runs == 1:
        axs[0].set_ylabel(labels[run])
    else:
        axs[i, 0].set_ylabel(labels[run])

plt.tight_layout()
plt.savefig("generated_images.png")
plt.show()

# Plot 3: Bar plot of evaluation metrics
metrics = ['eval_loss', 'kl_divergence', 'training_time', 'inference_time']
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Evaluation Metrics Across Runs", fontsize=16)

for i, metric in enumerate(metrics):
    row = i // 2
    col = i % 2
    data = [final_results[run][dataset][metric] for run in runs for dataset in datasets]
    x = np.arange(len(datasets) * len(runs))
    axs[row, col].bar(x, data, color=colors)
    axs[row, col].set_title(metric.replace('_', ' ').title())
    axs[row, col].set_xticks(x + 0.5 * (len(runs) - 1))
    axs[row, col].set_xticklabels(datasets * len(runs), rotation=45)
    axs[row, col].legend(labels.values(), loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig("evaluation_metrics.png")
plt.show()

# Plot 4: Grid variance comparison (for runs 3 and 4)
if 'run_3' in runs and 'run_4' in runs:
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Grid Variance Comparison", fontsize=16)

    for i, grid_type in enumerate(['coarse_grid_variance', 'fine_grid_variance']):
        data_run3 = [final_results['run_3'][dataset][grid_type] for dataset in datasets]
        data_run4 = [final_results['run_4'][dataset][grid_type] for dataset in datasets]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        axs[i].bar(x - width/2, data_run3, width, label='Multi-scale Grid', color=colors[3])
        axs[i].bar(x + width/2, data_run4, width, label='Multi-scale + L1 Reg', color=colors[4])
        
        axs[i].set_title(grid_type.replace('_', ' ').title())
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(datasets)
        axs[i].legend()

    plt.tight_layout()
    plt.savefig("grid_variance_comparison.png")
    plt.show()
