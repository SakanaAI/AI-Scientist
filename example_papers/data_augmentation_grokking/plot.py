import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp

# LOAD FINAL RESULTS:
datasets = ["x_div_y", "x_minus_y", "x_plus_y", "permutation"]
folders = os.listdir("./")
final_results = {}
results_info = {}
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        results_dict = np.load(
            osp.join(folder, "all_results.npy"), allow_pickle=True
        ).item()
        print(results_dict.keys())
        run_info = {}
        for dataset in datasets:
            run_info[dataset] = {}
            val_losses = []
            train_losses = []
            val_accs = []
            train_accs = []
            for k in results_dict.keys():
                if dataset in k and "val_info" in k:
                    run_info[dataset]["step"] = [
                        info["step"] for info in results_dict[k]
                    ]
                    val_losses.append([info["val_loss"] for info in results_dict[k]])
                    val_accs.append([info["val_accuracy"] for info in results_dict[k]])
                if dataset in k and "train_info" in k:
                    train_losses.append(
                        [info["train_loss"] for info in results_dict[k]]
                    )
                    train_accs.append(
                        [info["train_accuracy"] for info in results_dict[k]]
                    )
                mean_val_losses = np.mean(val_losses, axis=0)
                mean_train_losses = np.mean(train_losses, axis=0)
                mean_val_accs = np.mean(val_accs, axis=0)
                mean_train_accs = np.mean(train_accs, axis=0)
                if len(val_losses) > 0:
                    sterr_val_losses = np.std(val_losses, axis=0) / np.sqrt(
                        len(val_losses)
                    )
                    stderr_train_losses = np.std(train_losses, axis=0) / np.sqrt(
                        len(train_losses)
                    )
                    sterr_val_accs = np.std(val_accs, axis=0) / np.sqrt(len(val_accs))
                    stderr_train_accs = np.std(train_accs, axis=0) / np.sqrt(
                        len(train_accs)
                    )
                else:
                    sterr_val_losses = np.zeros_like(mean_val_losses)
                    stderr_train_losses = np.zeros_like(mean_train_losses)
                    sterr_val_accs = np.zeros_like(mean_val_accs)
                    stderr_train_accs = np.zeros_like(mean_train_accs)
                run_info[dataset]["val_loss"] = mean_val_losses
                run_info[dataset]["train_loss"] = mean_train_losses
                run_info[dataset]["val_loss_sterr"] = sterr_val_losses
                run_info[dataset]["train_loss_sterr"] = stderr_train_losses
                run_info[dataset]["val_acc"] = mean_val_accs
                run_info[dataset]["train_acc"] = mean_train_accs
                run_info[dataset]["val_acc_sterr"] = sterr_val_accs
                run_info[dataset]["train_acc_sterr"] = stderr_train_accs
        results_info[folder] = run_info

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baseline",
    "run_1": "Operand Reversal",
    "run_2": "Negation",
    "run_3": "Combined (15%)",
    "run_4": "Combined (30%)",
    "run_5": "Combined (20%)",
}


# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap("tab20")
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Line plot of validation accuracy for each dataset across the runs
for dataset in datasets:
    plt.figure(figsize=(12, 6))
    for i, run in enumerate(runs):
        iters = results_info[run][dataset]["step"]
        mean = results_info[run][dataset]["val_acc"]
        sterr = results_info[run][dataset]["val_acc_sterr"]
        plt.plot(iters, mean, label=labels[run], color=colors[i])
        plt.fill_between(iters, mean - sterr, mean + sterr, color=colors[i], alpha=0.2)

    plt.title(f"Validation Accuracy Across Runs for {dataset} Dataset")
    plt.xlabel("Update Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"val_acc_{dataset}.png")
    plt.close()

# Plot 2: Bar plot of steps to 99% validation accuracy for each dataset and run
step_val_acc_99 = {dataset: [] for dataset in datasets}
for dataset in datasets:
    for run in runs:
        step = final_results[run][dataset]["means"].get("step_val_acc_99_mean")
        step_val_acc_99[dataset].append(step if step is not None else 7500)

plt.figure(figsize=(12, 6))
x = np.arange(len(datasets))
width = 0.15
for i, run in enumerate(runs):
    plt.bar(x + i*width, [step_val_acc_99[dataset][i] for dataset in datasets], width, label=labels[run])

plt.xlabel('Datasets')
plt.ylabel('Steps to 99% Validation Accuracy')
plt.title('Steps to 99% Validation Accuracy Across Runs and Datasets')
plt.xticks(x + width * 2.5, datasets)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('steps_to_99_acc.png')
plt.close()

# Plot 3: Bar plot of final validation accuracy for each dataset and run
final_val_acc = {dataset: [] for dataset in datasets}
for dataset in datasets:
    for run in runs:
        acc = final_results[run][dataset]["means"]["final_val_acc_mean"]
        final_val_acc[dataset].append(acc)

plt.figure(figsize=(12, 6))
x = np.arange(len(datasets))
width = 0.15
for i, run in enumerate(runs):
    plt.bar(x + i*width, [final_val_acc[dataset][i] for dataset in datasets], width, label=labels[run])

plt.xlabel('Datasets')
plt.ylabel('Final Validation Accuracy')
plt.title('Final Validation Accuracy Across Runs and Datasets')
plt.xticks(x + width * 2.5, datasets)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('final_val_acc.png')
plt.close()
