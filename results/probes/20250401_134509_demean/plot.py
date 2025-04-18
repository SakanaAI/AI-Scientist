import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

def load_results(base_dir='.'):
    """
    Load results from all run directories and combine them into a single DataFrame.
    
    Args:
        base_dir: Base directory containing run_* directories
        
    Returns:
        DataFrame with all results and a 'run' column indicating the source
    """
    all_results = []
    
    # Find all run directories
    run_dirs = glob.glob(os.path.join(base_dir, 'run_*'))
    
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        results_file = os.path.join(run_dir, 'results.csv')
        
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            df['run'] = run_name
            all_results.append(df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

def plot_accuracy_comparison(results_df):
    """
    Plot accuracy comparison across different runs, layers, and beta values.
    """
    if results_df.empty:
        print("No results found to plot.")
        return
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy by run and layer (averaged over splits)
    pivot_layer = results_df.groupby(['run', 'layer'])['accuracy'].mean().reset_index()
    pivot_layer = pivot_layer.pivot(index='layer', columns='run', values='accuracy')
    
    sns.heatmap(pivot_layer, annot=True, fmt='.3f', cmap='viridis', ax=axes[0])
    axes[0].set_title('Average Accuracy by Layer and Run')
    axes[0].set_ylabel('Layer')
    
    # Plot 2: Accuracy by run and beta (averaged over splits, layers, and mean_scale if present)
    pivot_beta = results_df.groupby(['run', 'beta'])['accuracy'].mean().reset_index()
    pivot_beta = pivot_beta.pivot(index='beta', columns='run', values='accuracy')
    
    sns.heatmap(pivot_beta, annot=True, fmt='.3f', cmap='viridis', ax=axes[1])
    axes[1].set_title('Average Accuracy by Beta and Run')
    axes[1].set_ylabel('Beta')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.close()
    
    # Create line plot for accuracy by layer for each run
    plt.figure(figsize=(10, 6))
    
    for run in results_df['run'].unique():
        run_data = results_df[results_df['run'] == run]
        layer_means = run_data.groupby('layer')['accuracy'].mean()
        plt.plot(layer_means.index, layer_means.values, marker='o', label=run)
    
    plt.xlabel('Layer')
    plt.ylabel('Average Accuracy')
    plt.title('Accuracy by Layer for Different Runs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('accuracy_by_layer.png')
    plt.close()
    
    # Create line plot for accuracy by beta for each run
    plt.figure(figsize=(10, 6))
    
    for run in results_df['run'].unique():
        run_data = results_df[results_df['run'] == run]
        beta_means = run_data.groupby('beta')['accuracy'].mean()
        plt.plot(beta_means.index, beta_means.values, marker='o', label=run)
    
    plt.xlabel('Beta')
    plt.ylabel('Average Accuracy')
    plt.title('Accuracy by Beta for Different Runs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('accuracy_by_beta.png')
    plt.close()

def plot_mean_scale_comparison(results_df):
    """
    Plot accuracy comparison across different mean_scale values if they exist in the data.
    """
    if results_df.empty or 'mean_scale' not in results_df.columns:
        print("No mean_scale data found to plot.")
        return
    
    # Create line plot for accuracy by mean_scale for each run
    plt.figure(figsize=(10, 6))
    
    for run in results_df['run'].unique():
        run_data = results_df[results_df['run'] == run]
        if 'mean_scale' in run_data.columns:
            mean_scale_means = run_data.groupby('mean_scale')['accuracy'].mean()
            plt.plot(mean_scale_means.index, mean_scale_means.values, marker='o', label=run)
    
    plt.xlabel('Mean Scale')
    plt.ylabel('Average Accuracy')
    plt.title('Accuracy by Mean Scale for Different Runs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('accuracy_by_mean_scale.png')
    plt.close()
    
    # Create heatmap for mean_scale vs beta
    if 'mean_scale' in results_df.columns and 'beta' in results_df.columns:
        plt.figure(figsize=(12, 8))
        for run in results_df['run'].unique():
            run_data = results_df[results_df['run'] == run]
            if len(run_data['mean_scale'].unique()) > 1 and len(run_data['beta'].unique()) > 1:
                pivot = run_data.groupby(['mean_scale', 'beta'])['accuracy'].mean().reset_index()
                pivot_table = pivot.pivot(index='mean_scale', columns='beta', values='accuracy')
                
                plt.figure(figsize=(10, 6))
                sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis')
                plt.title(f'Accuracy by Mean Scale and Beta for {run}')
                plt.xlabel('Beta')
                plt.ylabel('Mean Scale')
                plt.tight_layout()
                plt.savefig(f'accuracy_heatmap_{run}.png')
                plt.close()

if __name__ == "__main__":
    # Load results from all runs
    results = load_results()
    
    # Plot comparisons
    plot_accuracy_comparison(results)
    
    # Plot mean_scale comparisons if data exists
    plot_mean_scale_comparison(results)
    
    print("Plots generated: accuracy_comparison.png, accuracy_by_layer.png, accuracy_by_beta.png")
    print("Additional plots may have been generated if mean_scale data was found.")
