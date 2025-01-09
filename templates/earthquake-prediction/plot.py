import json
import os
import os.path as osp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Load final results
datasets = ["earthquake"]
folders = os.listdir("./")
final_results = {}
results_info = {}


def plot_losses_with_phases(results_dict, dataset, save_path):
    """Plot both training and validation losses with phase transitions marked."""
    plt.figure(figsize=(10, 6))

    # Get validation info which has both train and val losses at validation points
    val_info = results_dict[f"{dataset}_val_info"]
    val_iters = [info["iter"] for info in val_info]
    train_at_val = [info["train/loss"] for info in val_info]
    val_losses = [info["val/loss"] for info in val_info]

    # Plot validation point losses
    plt.plot(val_iters, train_at_val,
             label='Training Loss (at validation iteration)', color='blue', alpha=0.5)
    plt.plot(val_iters, val_losses,
             label='Validation Loss', color='red', alpha=0.8)

    # Find phase transitions from phase information
    phases = [info.get("phase", "full1") for info in val_info]
    phase_changes = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            phase_changes.append(val_iters[i])

    # Add phase transition markers
    for i, change_point in enumerate(phase_changes):
        plt.axvline(x=change_point, color='gray',
                    linestyle='--' if i == 0 else ':',
                    label=f'Phase Change {i + 1}')

    plt.title(f"Training and Validation Loss with Phase Transitions")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        results_dict = np.load(osp.join(folder, "all_results.npy"),
                               allow_pickle=True).item()

        run_info = {}
        for dataset in datasets:
            run_info[dataset] = {}
            val_losses = []
            train_losses = []

            for k in results_dict.keys():
                if dataset in k and "val_info" in k:
                    run_info[dataset]["iters"] = [info["iter"]
                                                  for info in results_dict[k]]
                    val_losses.append([info["val/loss"]
                                       for info in results_dict[k]])
                    train_losses.append([info["train/loss"]
                                         for info in results_dict[k]])

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

# Create legend
labels = {
    "run_0": "Baseline",
}


# Generate color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


runs = list(labels.keys())
colors = generate_color_palette(len(runs))

# Plot training and validation losses
for dataset in datasets:
    for i, run in enumerate(runs):
        plot_losses_with_phases(
            results_dict=results_dict,
            dataset='earthquake',
            save_path=f"train_val_loss_with_phases_{dataset}.png"
        )
