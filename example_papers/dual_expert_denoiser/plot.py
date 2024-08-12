import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp
import pickle
import warnings

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
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Baseline",
    "run_1": "Dual-Expert",
    "run_2": "Enhanced Gating",
    "run_3": "Increased Capacity",
    "run_4": "Diversity Loss",
    "run_5": "Adjusted Diversity",
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

# Plot 1: KL Divergence comparison across runs
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(datasets))
width = 0.15
multiplier = 0

for run, label in labels.items():
    kl_values = []
    for dataset in datasets:
        kl_value = final_results[run][dataset].get('means', {}).get('kl_divergence', 0)
        if kl_value == 0:
            warnings.warn(f"KL divergence value missing for {run} on {dataset} dataset.")
        kl_values.append(kl_value)
    offset = width * multiplier
    rects = ax.bar(x + offset, kl_values, width, label=label)
    ax.bar_label(rects, padding=3, rotation=90, fmt='%.3f')
    multiplier += 1

ax.set_ylabel('KL Divergence')
ax.set_title('KL Divergence Comparison Across Runs')
ax.set_xticks(x + width * (len(labels) - 1) / 2)
ax.set_xticklabels(datasets)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
max_kl = max([max([final_results[run][dataset].get('means', {}).get('kl_divergence', 0) for dataset in datasets]) for run in labels])
if max_kl > 0:
    ax.set_ylim(0, max_kl * 1.2)
else:
    ax.set_ylim(0, 1)  # Set a default y-axis limit if all KL divergence values are 0 or missing

plt.tight_layout()
plt.savefig("kl_divergence_comparison.png")
plt.show()

# Plot 2: Generated samples comparison (focus on 'dino' dataset)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Generated Samples for 'dino' Dataset", fontsize=16)

for i, (run, label) in enumerate(labels.items()):
    row = i // 3
    col = i % 3
    images = train_info[run]['dino']["images"]
    gating_weights = train_info[run]['dino'].get("gating_weights")
    
    scatter = axs[row, col].scatter(images[:, 0], images[:, 1], c=gating_weights, cmap='coolwarm', alpha=0.5, vmin=0, vmax=1)
    axs[row, col].set_title(label)
    fig.colorbar(scatter, ax=axs[row, col], label='Gating Weight')

plt.tight_layout()
plt.savefig("dino_generated_samples.png")
plt.show()

# Plot 3: Training loss comparison (focus on 'dino' dataset)
fig, ax = plt.subplots(figsize=(12, 6))

for run, label in labels.items():
    mean = train_info[run]['dino']["train_losses"]
    mean = smooth(mean, window_len=25)
    ax.plot(mean, label=label)

ax.set_title("Training Loss for 'dino' Dataset")
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
ax.legend()

plt.tight_layout()
plt.savefig("dino_train_loss.png")
plt.show()

# Plot 4: Gating weights histogram comparison (focus on 'dino' dataset)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Gating Weights Histogram for 'dino' Dataset", fontsize=16)

for i, (run, label) in enumerate(labels.items()):
    row = i // 3
    col = i % 3
    gating_weights = train_info[run]['dino'].get("gating_weights")
    
    if gating_weights is not None:
        axs[row, col].hist(gating_weights, bins=50, range=(0, 1))
        axs[row, col].set_title(label)
        axs[row, col].set_xlabel("Gating Weight")
        axs[row, col].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("dino_gating_weights_histogram.png")
plt.show()
