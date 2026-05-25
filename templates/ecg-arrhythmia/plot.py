import json
import os
import os.path as osp

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


datasets = ["ecg_arrhythmia"]
folders = os.listdir("./")
results_info = {}


def plot_curves(results_dict, dataset, save_loss, save_f1):
    val_info = results_dict[f"{dataset}_val_info"]
    epochs = [info["iter"] for info in val_info]
    train_losses = [info["train/loss"] for info in val_info]
    val_losses = [info["val/loss"] for info in val_info]
    val_f1 = [info["val/macro_f1"] for info in val_info]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss", color="#4C78A8")
    plt.plot(epochs, val_losses, label="Val Loss", color="#F58518")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ECG Arrhythmia Training Curves")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_loss)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_f1, label="Val Macro F1", color="#54A24B")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("ECG Arrhythmia Validation Macro F1")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_f1)
    plt.close()


def plot_gate_by_class(results_dict, dataset, save_gate):
    test_metrics = results_dict.get(f"{dataset}_test_metrics", {})
    class_names = results_dict.get(f"{dataset}_class_names", [])
    per_class_gate = test_metrics.get("per_class_gate")
    if not per_class_gate:
        return

    labels = class_names if class_names else [f"Class {i}" for i in range(len(per_class_gate))]
    values = np.array(per_class_gate, dtype=float)
    valid = ~np.isnan(values)
    labels = [label for label, keep in zip(labels, valid) if keep]
    values = values[valid]
    if len(values) == 0:
        return

    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = values[order]

    plt.figure(figsize=(8, 5))
    bars = plt.barh(labels, values, color="#72B7B2")
    plt.axvline(np.nanmean(values), color="#E45756", linestyle="--", linewidth=1.5, label="Mean gate")
    for bar, value in zip(bars, values):
        plt.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    plt.xlim(0, 1)
    plt.xlabel("Mean gate value")
    plt.ylabel("ECG class")
    plt.title("Lead-aware gate by class")
    plt.grid(True, axis="x", alpha=0.2)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_gate)
    plt.close()


for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results = json.load(f)
        results_dict = np.load(osp.join(folder, "all_results.npy"), allow_pickle=True).item()
        results_info[folder] = results_dict


labels = {
    "run_0": "Baseline",
}


def generate_color_palette(n):
    cmap = plt.get_cmap("tab20")
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


runs = list(labels.keys())
_ = generate_color_palette(len(runs))

for dataset in datasets:
    for run in runs:
        if run in results_info:
            plot_curves(
                results_dict=results_info[run],
                dataset=dataset,
                save_loss=f"{dataset}_loss.png",
                save_f1=f"{dataset}_macro_f1.png",
            )
            plot_gate_by_class(
                results_dict=results_info[run],
                dataset=dataset,
                save_gate=f"{dataset}_gate_by_class.png",
            )
