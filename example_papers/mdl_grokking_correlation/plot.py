import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp
from scipy.signal import savgol_filter

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

            # Add MDL info
            mdl_data = [info for k, info in results_dict.items() if dataset in k and "mdl_info" in k]
            if mdl_data:
                run_info[dataset]["mdl_step"] = [item["step"] for item in mdl_data[0]]
                run_info[dataset]["mdl"] = [item["mdl"] for item in mdl_data[0]]

        results_info[folder] = run_info

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baseline",
    "run_1": "MDL Tracking",
    "run_2": "MDL Analysis",
    "run_3": "Extended Analysis",
    "run_4": "Comprehensive Analysis",
}


# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap("tab20")
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Get the list of runs and generate the color palette
runs = list(labels.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Line plot of training loss for each dataset across the runs with labels
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        iters = results_info[run][dataset]["step"]
        mean = results_info[run][dataset]["train_loss"]
        sterr = results_info[run][dataset]["train_loss_sterr"]
        plt.plot(iters, mean, label=labels[run], color=colors[i])
        plt.fill_between(iters, mean - sterr, mean + sterr, color=colors[i], alpha=0.2)

    plt.title(f"Training Loss Across Runs for {dataset} Dataset")
    plt.xlabel("Update Steps")
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
        iters = results_info[run][dataset]["step"]
        mean = results_info[run][dataset]["val_loss"]
        sterr = results_info[run][dataset]["val_loss_sterr"]
        plt.plot(iters, mean, label=labels[run], color=colors[i])
        plt.fill_between(iters, mean - sterr, mean + sterr, color=colors[i], alpha=0.2)

    plt.title(f"Validation Loss Across Runs for {dataset} Dataset")
    plt.xlabel("Update Steps")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"val_loss_{dataset}.png")
    plt.close()


# Plot 3: Line plot of training acc for each dataset across the runs with labels
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        iters = results_info[run][dataset]["step"]
        mean = results_info[run][dataset]["train_acc"]
        sterr = results_info[run][dataset]["train_acc_sterr"]
        plt.plot(iters, mean, label=labels[run], color=colors[i])
        plt.fill_between(iters, mean - sterr, mean + sterr, color=colors[i], alpha=0.2)

    plt.title(f"Training Accuracy Across Runs for {dataset} Dataset")
    plt.xlabel("Update Steps")
    plt.ylabel("Training Acc")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"train_acc_{dataset}.png")
    plt.close()

# Plot 2: Line plot of validation acc for each dataset across the runs with labels
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        iters = results_info[run][dataset]["step"]
        mean = results_info[run][dataset]["val_acc"]
        sterr = results_info[run][dataset]["val_acc_sterr"]
        plt.plot(iters, mean, label=labels[run], color=colors[i])
        plt.fill_between(iters, mean - sterr, mean + sterr, color=colors[i], alpha=0.2)

    plt.title(f"Validation Loss Across Runs for {dataset} Dataset")
    plt.xlabel("Update Steps")
    plt.ylabel("Validation Acc")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"val_acc_{dataset}.png")
    plt.close()

# Plot 5: MDL estimates alongside validation accuracy
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        if run != "run_0":  # Skip baseline run
            iters = results_info[run][dataset]["step"]
            val_acc = results_info[run][dataset]["val_acc"]
            mdl_step = results_info[run][dataset]["mdl_step"]
            mdl = results_info[run][dataset]["mdl"]

            # Normalize MDL values
            mdl_normalized = (mdl - np.min(mdl)) / (np.max(mdl) - np.min(mdl))

            # Apply Savitzky-Golay filter to smooth MDL curve
            mdl_smooth = savgol_filter(mdl_normalized, window_length=5, polyorder=2)

            plt.plot(iters, val_acc, label=f"{labels[run]} - Val Acc", color=colors[i])
            plt.plot(mdl_step, mdl_smooth, label=f"{labels[run]} - MDL", linestyle='--', color=colors[i])

    plt.title(f"Validation Accuracy and MDL for {dataset} Dataset")
    plt.xlabel("Update Steps")
    plt.ylabel("Validation Accuracy / Normalized MDL")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"val_acc_mdl_{dataset}.png")
    plt.close()

# Calculate MDL transition point and correlation
mdl_analysis = {}
for dataset in datasets:
    mdl_analysis[dataset] = {}
    for run in runs:
        if run != "run_0":  # Skip baseline run
            mdl = results_info[run][dataset]["mdl"]
            mdl_step = results_info[run][dataset]["mdl_step"]
            val_acc = results_info[run][dataset]["val_acc"]
            train_acc = results_info[run][dataset]["train_acc"]

            # Calculate MDL transition point (steepest decrease)
            mdl_diff = np.diff(mdl)
            mdl_transition_idx = np.argmin(mdl_diff)
            mdl_transition_point = mdl_step[mdl_transition_idx]

            # Find grokking point (95% validation accuracy)
            grokking_point = next((step for step, acc in zip(results_info[run][dataset]["step"], val_acc) if acc >= 0.95), None)

            # Calculate correlation between MDL reduction and validation accuracy improvement
            mdl_normalized = (mdl - np.min(mdl)) / (np.max(mdl) - np.min(mdl))
            val_acc_interp = np.interp(mdl_step, results_info[run][dataset]["step"], val_acc)
            correlation = np.corrcoef(mdl_normalized, val_acc_interp)[0, 1]

            # Calculate generalization gap
            train_acc_interp = np.interp(mdl_step, results_info[run][dataset]["step"], train_acc)
            gen_gap = train_acc_interp - val_acc_interp

            mdl_analysis[dataset][run] = {
                "mdl_transition_point": mdl_transition_point,
                "grokking_point": grokking_point,
                "correlation": correlation,
                "mdl": mdl,
                "mdl_step": mdl_step,
                "val_acc": val_acc_interp,
                "gen_gap": gen_gap
            }

# Plot MDL transition point vs Grokking point
plt.figure(figsize=(10, 6))
for dataset in datasets:
    for run in runs:
        if run != "run_0":
            mdl_tp = mdl_analysis[dataset][run]["mdl_transition_point"]
            grok_p = mdl_analysis[dataset][run]["grokking_point"]
            plt.scatter(mdl_tp, grok_p, label=f"{dataset} - {run}")

plt.plot([0, max(plt.xlim())], [0, max(plt.xlim())], 'k--', alpha=0.5)
plt.xlabel("MDL Transition Point")
plt.ylabel("Grokking Point")
plt.title("MDL Transition Point vs Grokking Point")
plt.legend()
plt.tight_layout()
plt.savefig("mdl_transition_vs_grokking.png")
plt.close()

# Plot correlation between MDL reduction and val acc improvement
plt.figure(figsize=(10, 6))
for dataset in datasets:
    correlations = [mdl_analysis[dataset][run]["correlation"] for run in runs if run != "run_0"]
    plt.bar(dataset, np.mean(correlations), yerr=np.std(correlations), capsize=5)

plt.xlabel("Dataset")
plt.ylabel("Correlation")
plt.title("Correlation between MDL Reduction and Val Acc Improvement")
plt.tight_layout()
plt.savefig("mdl_val_acc_correlation.png")
plt.close()

# Plot MDL evolution and generalization gap
for dataset in datasets:
    plt.figure(figsize=(12, 8))
    for run in runs:
        if run != "run_0":
            mdl_step = mdl_analysis[dataset][run]["mdl_step"]
            mdl = mdl_analysis[dataset][run]["mdl"]
            gen_gap = mdl_analysis[dataset][run]["gen_gap"]
            
            plt.subplot(2, 1, 1)
            plt.plot(mdl_step, mdl, label=f"{run} - MDL")
            plt.title(f"MDL Evolution and Generalization Gap - {dataset}")
            plt.ylabel("MDL")
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(mdl_step, gen_gap, label=f"{run} - Gen Gap")
            plt.xlabel("Steps")
            plt.ylabel("Generalization Gap")
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"mdl_gen_gap_{dataset}.png")
    plt.close()

# Calculate and plot MDL transition rate
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for run in runs:
        if run != "run_0":
            mdl_step = mdl_analysis[dataset][run]["mdl_step"]
            mdl = mdl_analysis[dataset][run]["mdl"]
            mdl_rate = np.gradient(mdl, mdl_step)
            plt.plot(mdl_step, mdl_rate, label=f"{run} - MDL Rate")
    plt.title(f"MDL Transition Rate - {dataset}")
    plt.xlabel("Steps")
    plt.ylabel("MDL Rate of Change")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"mdl_transition_rate_{dataset}.png")
    plt.close()

# Scatter plot of MDL transition points vs grokking points
plt.figure(figsize=(10, 6))
for dataset in datasets:
    for run in runs:
        if run != "run_0":
            mdl_tp = mdl_analysis[dataset][run]["mdl_transition_point"]
            grok_p = mdl_analysis[dataset][run]["grokking_point"]
            if mdl_tp is not None and grok_p is not None:
                plt.scatter(mdl_tp, grok_p, label=f"{dataset} - {run}")
if plt.gca().get_xlim()[1] > 0 and plt.gca().get_ylim()[1] > 0:
    plt.plot([0, max(plt.xlim())], [0, max(plt.ylim())], 'k--', alpha=0.5)
plt.xlabel("MDL Transition Point")
plt.ylabel("Grokking Point")
plt.title("MDL Transition Point vs Grokking Point")
plt.legend()
plt.tight_layout()
plt.savefig("mdl_transition_vs_grokking_scatter.png")
plt.close()

# Print analysis results
for dataset in datasets:
    print(f"Dataset: {dataset}")
    for run in runs:
        if run != "run_0":
            analysis = mdl_analysis[dataset][run]
            print(f"  Run: {run}")
            print(f"    MDL Transition Point: {analysis['mdl_transition_point']}")
            print(f"    Grokking Point: {analysis['grokking_point']}")
            print(f"    Correlation: {analysis['correlation']:.4f}")
    print()

# Calculate and print average MDL transition point and grokking point for each dataset
for dataset in datasets:
    mdl_tps = []
    grok_ps = []
    correlations = []
    for run in runs:
        if run != "run_0":
            mdl_tps.append(mdl_analysis[dataset][run]["mdl_transition_point"])
            grok_ps.append(mdl_analysis[dataset][run]["grokking_point"])
            correlations.append(mdl_analysis[dataset][run]["correlation"])
    avg_mdl_tp = np.mean(mdl_tps) if mdl_tps else None
    avg_grok_p = np.mean(grok_ps) if grok_ps else None
    avg_correlation = np.mean(correlations) if correlations else None
    print(f"Dataset: {dataset}")
    print(f"  Average MDL Transition Point: {avg_mdl_tp:.2f if avg_mdl_tp is not None else 'N/A'}")
    print(f"  Average Grokking Point: {avg_grok_p:.2f if avg_grok_p is not None else 'N/A'}")
    if avg_mdl_tp is not None and avg_grok_p is not None:
        print(f"  Difference: {abs(avg_mdl_tp - avg_grok_p):.2f}")
    else:
        print("  Difference: N/A")
    print(f"  Average Correlation: {avg_correlation:.4f if avg_correlation is not None else 'N/A'}")

    # Add these lines for debugging
    print(f"  MDL Transition Points: {mdl_tps}")
    print(f"  Grokking Points: {grok_ps}")
    print(f"  Correlations: {correlations}")
    print()

# Plot MDL Transition Rate vs Grokking Speed
try:
    plt.figure(figsize=(12, 8))
    for dataset in datasets:
        for run in runs:
            if run != "run_0":
                analysis = mdl_analysis[dataset][run]
                mdl_transition_rate = np.min(np.gradient(analysis['mdl'], analysis['mdl_step']))
                if analysis['grokking_point'] is not None and analysis['mdl_transition_point'] is not None:
                    if analysis['grokking_point'] != analysis['mdl_transition_point']:
                        grokking_speed = 1 / (analysis['grokking_point'] - analysis['mdl_transition_point'])
                    else:
                        grokking_speed = np.inf
                    plt.scatter(mdl_transition_rate, grokking_speed, label=f"{dataset} - {labels[run]}", alpha=0.7)

    plt.xlabel("MDL Transition Rate")
    plt.ylabel("Grokking Speed")
    plt.title("MDL Transition Rate vs Grokking Speed")
    plt.legend()
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("mdl_transition_rate_vs_grokking_speed.png")
    plt.close()
except Exception as e:
    print(f"Error plotting MDL Transition Rate vs Grokking Speed: {e}")

# Plot MDL evolution and validation accuracy for all datasets
for dataset in datasets:
    plt.figure(figsize=(15, 10))
    for run in runs:
        if run != "run_0":
            analysis = mdl_analysis[dataset][run]
            mdl_step = analysis['mdl_step']
            mdl = analysis['mdl']
            val_acc = analysis['val_acc']
            
            plt.plot(mdl_step, mdl, label=f'{labels[run]} - MDL')
            plt.plot(mdl_step, val_acc, label=f'{labels[run]} - Val Acc')
            plt.axvline(x=analysis['mdl_transition_point'], color='r', linestyle='--', label='MDL Transition')
            plt.axvline(x=analysis['grokking_point'], color='g', linestyle='--', label='Grokking Point')
    
    plt.title(f"MDL Evolution and Validation Accuracy - {dataset}")
    plt.xlabel("Steps")
    plt.ylabel("MDL / Validation Accuracy")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"mdl_val_acc_evolution_{dataset}.png")
    plt.close()

# Plot correlation between MDL reduction and validation accuracy improvement
plt.figure(figsize=(10, 6))
for dataset in datasets:
    correlations = []
    for run in runs:
        if run != "run_0":
            correlations.append(mdl_analysis[dataset][run]["correlation"])
    plt.bar(dataset, np.mean(correlations), yerr=np.std(correlations), capsize=5)

plt.xlabel("Dataset")
plt.ylabel("Correlation")
plt.title("Correlation between MDL Reduction and Validation Accuracy Improvement")
plt.tight_layout()
plt.savefig("mdl_val_acc_correlation.png")
plt.close()

# Print analysis results
print("\nAnalysis Results:")
for dataset in datasets:
    print(f"\nDataset: {dataset}")
    for run in runs:
        if run != "run_0":
            analysis = mdl_analysis[dataset][run]
            print(f"  Run: {labels[run]}")
            print(f"    MDL Transition Point: {analysis['mdl_transition_point']}")
            print(f"    Grokking Point: {analysis['grokking_point']}")
            print(f"    Correlation: {analysis['correlation']:.4f}")

# Calculate and print average MDL transition point and grokking point for each dataset
print("\nAverage MDL Transition Point and Grokking Point:")
for dataset in datasets:
    mdl_tps = []
    grok_ps = []
    correlations = []
    for run in runs:
        if run != "run_0":
            mdl_tp = mdl_analysis[dataset][run]["mdl_transition_point"]
            grok_p = mdl_analysis[dataset][run]["grokking_point"]
            correlation = mdl_analysis[dataset][run]["correlation"]
            if mdl_tp is not None:
                mdl_tps.append(mdl_tp)
            if grok_p is not None:
                grok_ps.append(grok_p)
            if correlation is not None:
                correlations.append(correlation)
    
    avg_mdl_tp = np.mean(mdl_tps) if mdl_tps else None
    avg_grok_p = np.mean(grok_ps) if grok_ps else None
    avg_correlation = np.mean(correlations) if correlations else None
    
    print(f"\nDataset: {dataset}")
    print(f"  Average MDL Transition Point: {avg_mdl_tp:.2f if avg_mdl_tp is not None else 'N/A'}")
    print(f"  Average Grokking Point: {avg_grok_p:.2f if avg_grok_p is not None else 'N/A'}")
    if avg_mdl_tp is not None and avg_grok_p is not None:
        print(f"  Difference: {abs(avg_mdl_tp - avg_grok_p):.2f}")
    else:
        print("  Difference: N/A")
    print(f"  Average Correlation: {avg_correlation:.4f if avg_correlation is not None else 'N/A'}")

    # Add these lines for debugging
    print(f"  MDL Transition Points: {mdl_tps}")
    print(f"  Grokking Points: {grok_ps}")
    print(f"  Correlations: {correlations}")
