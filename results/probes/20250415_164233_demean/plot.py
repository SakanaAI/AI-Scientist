import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

def plot_results(run_dirs):
    """
    Plot the results from multiple runs for comparison.
    
    Args:
        run_dirs (list): List of run directory names to compare
    """
    plt.figure(figsize=(14, 8))
    
    # Set up colors for different runs
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Plot each run
    for i, run_dir in enumerate(run_dirs):
        # Load the final info
        with open(os.path.join(run_dir, "final_info.json"), "r") as f:
            results = json.load(f)
        
        # Extract data for plotting
        layers = []
        betas = []
        accuracies = []
        
        for key, value in results.items():
            # Parse the key to extract layer and beta
            parts = key.split('_')
            layer = float(parts[1])
            beta = float(parts[3])
            
            layers.append(layer)
            betas.append(beta)
            accuracies.append(value['means'])
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame({
            'Layer': layers,
            'Beta': betas,
            'Accuracy': accuracies
        })
        
        # Plot for each beta value
        for beta in sorted(df['Beta'].unique()):
            beta_df = df[df['Beta'] == beta]
            plt.plot(beta_df['Layer'], beta_df['Accuracy'], 
                     marker='o', linestyle='-', 
                     color=colors[i % len(colors)],
                     alpha=0.7 if i > 0 else 1.0,  # Make baseline slightly transparent
                     label=f"{os.path.basename(run_dir)} (β={beta})")
    
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Accuracy Across Runs')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # List of run directories to compare
    run_dirs = ["run_0", "run_1"]  # Add more as needed
    plot_results(run_dirs)
