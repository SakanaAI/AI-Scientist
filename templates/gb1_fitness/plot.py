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
        "val_mse": np.mean([[e["mse"] for e in vl] for vl in val_logs], axis=0),
        "val_mse_sterr": np.std([[e["mse"] for e in vl] for vl in val_logs], axis=0) / np.sqrt(len(seeds)),
        "val_spearman": np.mean([[e["spearman"] for e in vl] for vl in val_logs], axis=0),
        "val_spearman_sterr": np.std([[e["spearman"] for e in vl] for vl in val_logs], axis=0) / np.sqrt(len(seeds)),
    }

labels = {"run_0": "Baseline (ESM-2 + MLP)"}


def color_palette(n):
    cmap = plt.get_cmap("tab20")
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


runs = list(labels.keys())
colors = color_palette(len(runs))

# Plot 1: Training loss
plt.figure(figsize=(8, 5))
for i, run in enumerate(runs):
    if run not in results_info:
        continue
    info = results_info[run]
    plt.plot(info["epochs"], info["train_loss"], label=labels[run], color=colors[i])
    plt.fill_between(info["epochs"],
                     info["train_loss"] - info["train_loss_sterr"],
                     info["train_loss"] + info["train_loss_sterr"],
                     color=colors[i], alpha=0.2)
plt.title("Training Loss (MSE) vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Train Loss (MSE)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("train_loss.png")
plt.close()

# Plot 2: Validation MSE
plt.figure(figsize=(8, 5))
for i, run in enumerate(runs):
    if run not in results_info:
        continue
    info = results_info[run]
    plt.plot(info["epochs"], info["val_mse"], label=labels[run], color=colors[i])
    plt.fill_between(info["epochs"],
                     info["val_mse"] - info["val_mse_sterr"],
                     info["val_mse"] + info["val_mse_sterr"],
                     color=colors[i], alpha=0.2)
plt.title("Validation MSE vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Val MSE")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("val_mse.png")
plt.close()

# Plot 3: Validation Spearman correlation
plt.figure(figsize=(8, 5))
for i, run in enumerate(runs):
    if run not in results_info:
        continue
    info = results_info[run]
    plt.plot(info["epochs"], info["val_spearman"], label=labels[run], color=colors[i])
    plt.fill_between(info["epochs"],
                     info["val_spearman"] - info["val_spearman_sterr"],
                     info["val_spearman"] + info["val_spearman_sterr"],
                     color=colors[i], alpha=0.2)
plt.title("Validation Spearman Correlation vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Spearman ρ")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("val_spearman.png")
plt.close()

# Plot 4: Bar chart of final Spearman per run
if final_results:
    run_names = [r for r in runs if r in final_results]
    means = [final_results[r]["means"]["final_val_spearman"] for r in run_names]
    errs = [final_results[r]["stderrs"]["final_val_spearman"] for r in run_names]
    plt.figure(figsize=(6, 4))
    plt.bar([labels[r] for r in run_names], means, yerr=errs,
            color=colors[:len(run_names)], capsize=5)
    plt.title("Final Validation Spearman Correlation")
    plt.ylabel("Spearman ρ")
    plt.tight_layout()
    plt.savefig("final_spearman_bar.png")
    plt.close()
