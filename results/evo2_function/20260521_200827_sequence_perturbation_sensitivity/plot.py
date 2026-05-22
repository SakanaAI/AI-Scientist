import json
import os
import os.path as osp

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

folders = [f for f in os.listdir("./") if f.startswith("run") and osp.isdir(f)]
final_results = {}
results_info = {}

for folder in folders:
    fp = osp.join(folder, "final_info.json")
    if not osp.exists(fp):
        continue
    with open(fp) as f:
        final_results[folder] = json.load(f)

    rp = osp.join(folder, "all_results.npy")
    if not osp.exists(rp):
        continue
    results_dict = np.load(rp, allow_pickle=True).item()
    seeds = sorted({k.split("_")[0] for k in results_dict if "train_log" in k})
    train_logs = [results_dict[f"{s}_train_log"] for s in seeds]
    val_logs = [results_dict[f"{s}_val_log"] for s in seeds]

    epochs = [e["epoch"] for e in train_logs[0]]
    results_info[folder] = {
        "epochs": epochs,
        "train_loss": np.mean([[e["train_loss"] for e in tl] for tl in train_logs], axis=0),
        "train_loss_sterr": np.std([[e["train_loss"] for e in tl] for tl in train_logs], axis=0) / np.sqrt(len(seeds)),
        "val_auroc": np.mean([[e["auroc"] for e in vl] for vl in val_logs], axis=0),
        "val_auroc_sterr": np.std([[e["auroc"] for e in vl] for vl in val_logs], axis=0) / np.sqrt(len(seeds)),
        "val_accuracy": np.mean([[e["accuracy"] for e in vl] for vl in val_logs], axis=0),
        "val_accuracy_sterr": np.std([[e["accuracy"] for e in vl] for vl in val_logs], axis=0) / np.sqrt(len(seeds)),
        "val_f1": np.mean([[e["f1"] for e in vl] for vl in val_logs], axis=0),
        "val_f1_sterr": np.std([[e["f1"] for e in vl] for vl in val_logs], axis=0) / np.sqrt(len(seeds)),
    }

labels = {"run_0": "Evo2-7B (layer 28, mean-pool)"}


def color_palette(n):
    cmap = plt.get_cmap("tab20")
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


runs = list(labels.keys())
colors = color_palette(len(runs))


def plot_metric(metric, ylabel, title, fname):
    plt.figure(figsize=(8, 5))
    for i, run in enumerate(runs):
        if run not in results_info:
            continue
        info = results_info[run]
        plt.plot(info["epochs"], info[metric], label=labels[run], color=colors[i])
        plt.fill_between(
            info["epochs"],
            info[metric] - info[f"{metric}_sterr"],
            info[metric] + info[f"{metric}_sterr"],
            color=colors[i], alpha=0.2,
        )
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


plot_metric("train_loss", "Binary Cross-Entropy Loss", "Training Loss vs Epoch", "train_loss.png")
plot_metric("val_auroc", "AUROC", "Validation AUROC vs Epoch", "val_auroc.png")
plot_metric("val_accuracy", "Accuracy", "Validation Accuracy vs Epoch", "val_accuracy.png")
plot_metric("val_f1", "F1 Score", "Validation F1 vs Epoch", "val_f1.png")

# Bar chart: final metrics comparison across runs
if final_results:
    metrics = ["final_val_accuracy", "final_val_auroc", "final_val_f1"]
    metric_labels = ["Accuracy", "AUROC", "F1"]
    run_names = [r for r in runs if r in final_results]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        means = [final_results[r]["means"][metric] for r in run_names]
        errs = [final_results[r]["stderrs"][metric] for r in run_names]
        ax.bar([labels[r] for r in run_names], means, yerr=errs,
               color=colors[:len(run_names)], capsize=5)
        ax.set_title(mlabel)
        ax.set_ylim(0, 1)
        ax.set_ylabel(mlabel)
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Final Classification Metrics: DNA Repair vs Tumor Suppressor")
    plt.tight_layout()
    plt.savefig("final_metrics_bar.png")
    plt.close()
