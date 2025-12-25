import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_convergence(out_dir):
    """
    Loads optimization histories and plots the convergence of energy vs. iteration.
    """
    # Load history arrays from the output directory
    try:
        gd_history = np.load(os.path.join(out_dir, "gd_history.npy"))
        qng_history = np.load(os.path.join(out_dir, "qng_history.npy"))
    except FileNotFoundError:
        print("Error: History files not found. Please run experiment.py first.")
        return

    # --- Plot Convergence ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    plt.plot(gd_history, label="Gradient Descent (GD)", marker='o', linestyle='--', markersize=4)
    plt.plot(qng_history, label="Quantum Natural Gradient (QNG)", marker='x', linestyle='-')
    
    # Add a line for the true ground state energy
    plt.axhline(y=-1.0, color='r', linestyle=':', label="True Ground State Energy (-1.0)")

    # --- Formatting ---
    plt.title("VQE Optimization Convergence", fontsize=16)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Energy Expectation Value", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(out_dir, "vqe_convergence.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, help="Directory containing experiment results.")
    args = parser.parse_args()
    plot_convergence(args.out_dir)