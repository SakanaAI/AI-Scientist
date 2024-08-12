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

# CREATE LEGEND -- PLEASE FILL IN YOUR RUN NAMES HERE
labels = {
    "run_0": "Baseline",
    "run_2": "Gradient Penalty",
    "run_3": "Fine-Tuned Hyperparameters",
    "run_4": "Quadratic Beta Schedule",
}

# Use the run key as the default label if not specified
runs = list(final_results.keys())
for run in runs:
    if run not in labels:
        labels[run] = run


# CREATE PLOTS

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')  # You can change 'tab20' to other colormaps like 'Set1', 'Set2', 'Set3', etc.
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Get the list of runs and generate the color palette
runs = list(final_results.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Line plot of training loss for each dataset across the runs with labels
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

for j, dataset in enumerate(datasets):
    row = j // 2
    col = j % 2
    for i, run in enumerate(runs):
        if run in labels:
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
num_runs = len([run for run in runs if run in labels])
fig, axs = plt.subplots(num_runs, 4, figsize=(14, 3 * num_runs))

for i, run in enumerate(runs):
    if run in labels:
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
