import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_results(results_path):
    if not os.path.exists(results_path):
        return None
    with open(results_path, "r") as f:
        return json.load(f)

def plot_summary():
    # Define runs to include in the summary plot
    run_dirs = [f"run_{i}" for i in range(0, 11)]
    labels = {
        "run_0": "SVD & NN (0 iter)",
        "run_1": "SVD & NN (1 iter)",
        "run_2": "SVD & NN (2 iter)",
        "run_3": "SVD & NN (3 iter)",
        "run_4": "SVD & NN (4 iter)",
        "run_5": "SVD & NN (5 iter)",
        "run_6": "SVD & NN (6 iter)",
        "run_7": "SVD & NN (7 iter)",
        "run_8": "SVD & NN (8 iter)",
        "run_9": "SVD & NN (9 iter)",
        "run_10": "SVD & NN (10 iter)",
    }

    init_means = []
    init_stderrs = []
    corr_means = []
    corr_stderrs = []
    run_labels = []

    for run_dir in run_dirs:
        run_label = labels.get(run_dir)
        if not run_label:
            continue

        results_path = os.path.join(run_dir, "final_info.json")
        if not os.path.exists(results_path):
            continue

        results = load_results(results_path)
        if results is None:
            continue

        orbit_corr = results.get("orbit_correction", {})
        means = orbit_corr.get("means", {})
        stderrs = orbit_corr.get("stderrs", {})

        init_means.append(means.get("init_loss_mean", 0))
        init_stderrs.append(stderrs.get("init_loss_stderr", 0))
        corr_means.append(means.get("corr_loss_mean", 0))
        corr_stderrs.append(stderrs.get("corr_loss_stderr", 0))
        run_labels.append(run_label)

    if not run_labels:
        return

    # Set up plot
    plt.figure(figsize=(14, 8))
    x = np.arange(len(run_labels))
    width = 0.35

    # Plot bars with error bars
    plt.bar(x - width/2, init_means, width, yerr=init_stderrs, label='Initial Loss', capsize=5)
    plt.bar(x + width/2, corr_means, width, yerr=corr_stderrs, label='Corrected Loss', capsize=5)

    # Formatting
    plt.yscale('log')
    plt.ylabel('Loss $m^2$ (log scale)')
    plt.title('Orbit Correction Performance by Run')
    plt.xticks(x, run_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig("summary_plot.png")
    plt.close()

if __name__ == "__main__":
    plot_summary()
