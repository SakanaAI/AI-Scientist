import json
import os
import os.path as osp
import pickle

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Load results
folders = [f for f in os.listdir('./') if f.startswith('run_') and osp.isdir(f)]
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
    try:
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        all_results = pickle.load(open(osp.join(folder, "all_results.pkl"), "rb"))
        train_info[folder] = all_results
    except (FileNotFoundError, IOError):
        print(f"Skipping folder {folder} due to missing results files")

# Create labels dictionary
labels = {
    "run_0": "Baseline AMP",
}

# Use run key as default label if not specified
runs = list(final_results.keys())
for run in runs:
    if run not in labels:
        labels[run] = run

# Create color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20') 
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

runs = list(final_results.keys())
colors = generate_color_palette(len(runs))

# Create figure with 1x3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Training losses
for i, run in enumerate(runs):
    for motion, results in train_info[run].items():
        if 'train_losses' in results and results['train_losses']:
            losses = results['train_losses']
            mean = smooth(losses, window_len=25)
            ax1.plot(mean, label=f"{labels[run]} - {motion}", color=colors[i])
            
ax1.set_title("Training Losses Across Motions")
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Loss")
ax1.legend()

# Plot 2: Discriminator rewards
for i, run in enumerate(runs):
    rewards = []
    motion_names = []
    for motion, results in train_info[run].items():
        if 'means' in results and 'disc_reward_mean' in results['means']:
            rewards.append(results['means']['disc_reward_mean'])
            motion_names.append(motion)
    
    ax2.bar(motion_names, rewards, label=labels[run], color=colors[i])

ax2.set_title("Discriminator Rewards by Motion")
ax2.set_ylabel("Mean Discriminator Reward")
ax2.tick_params(axis='x', rotation=45)
ax2.legend()

# Plot 3: Pose errors
for i, run in enumerate(runs):
    errors = []
    motion_names = []
    for motion, results in train_info[run].items():
        if 'final_pose_error' in results:
            errors.append(results['final_pose_error'])
            motion_names.append(motion)
    
    ax3.bar(motion_names, errors, label=labels[run], color=colors[i])

ax3.set_title("Final Pose Errors by Motion")
ax3.set_ylabel("Mean Pose Error")
ax3.tick_params(axis='x', rotation=45)
ax3.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig("amp_training_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

# Print final metrics
print("\nFinal Metrics:")
for run in runs:
    print(f"\n{labels.get(run, run)}:")
    for motion, results in final_results[run].items():
        print(f"\n{motion}:")
        print(f"Training Time: {results['training_time']:.2f}s")
        print(f"Samples Collected: {results['samples_collected']}")
        print(f"Final Disc Reward: {results['means']['disc_reward_mean']:.4f} ± {results['stderrs']['disc_reward_std']:.4f}")
        if 'final_pose_error' in results:
            print(f"Final Pose Error: {results['final_pose_error']:.4f}")