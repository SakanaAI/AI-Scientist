import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

def plot_results(results_dir):
    """
    Plot the results from the experiment.
    
    Args:
        results_dir (str): Directory containing the results
    """
    # Load results
    results_file = os.path.join(results_dir, "results.csv")
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
        
    results = pd.read_csv(results_file)
    
    # Check if this is a layer-specific scales experiment
    if 'layer_specific_scale' in results.columns:
        # Calculate mean accuracy per layer and beta
        mean_results = results.groupby(['layer', 'beta'])['accuracy'].mean().reset_index()
        
        # Create plot for each beta value
        unique_betas = mean_results['beta'].unique()
        for beta in unique_betas:
            beta_data = mean_results[mean_results['beta'] == beta]
            
            plt.figure(figsize=(12, 6))
            # Get the layer-specific scale for each layer
            beta_data['scale_label'] = beta_data['layer'].apply(
                lambda x: f"Layer {x}\nScale {results[results['layer'] == x]['layer_specific_scale'].iloc[0]:.2f}"
            )
            
            sns.barplot(x='scale_label', y='accuracy', data=beta_data)
            plt.title(f'Mean Accuracy with Layer-Specific Scales (Beta={beta}) - {os.path.basename(results_dir)}')
            plt.xlabel('Layer and Scale')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(results_dir, f'layer_specific_accuracy_plot_beta_{beta}.png'))
        
        # Find best configuration
        best_idx = results['accuracy'].idxmax()
        best_row = results.loc[best_idx]
        print(f"\nSummary for {results_dir} (Layer-Specific Scales Experiment):")
        print(f"Overall mean accuracy: {results['accuracy'].mean():.4f}")
        print(f"Best accuracy: {results['accuracy'].max():.4f}")
        print(f"Best configuration: Layer {best_row['layer']}, " +
              f"Beta {best_row['beta']}, Scale {best_row['layer_specific_scale']:.2f}")
              
    # Check if this is a cross-layer experiment
    elif 'steering_layer' in results.columns and 'demean_layer' in results.columns:
        # Calculate mean accuracy per steering_layer, demean_layer, and beta
        mean_results = results.groupby(['steering_layer', 'demean_layer', 'beta'])['accuracy'].mean().reset_index()
        
        # Create heatmap for each beta value
        unique_betas = mean_results['beta'].unique()
        for beta in unique_betas:
            beta_data = mean_results[mean_results['beta'] == beta]
            
            plt.figure(figsize=(10, 8))
            heatmap_data = beta_data.pivot(index='steering_layer', columns='demean_layer', values='accuracy')
            sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=0, vmax=1.0)
            plt.title(f'Cross-Layer Accuracy Heatmap (Beta={beta}) - {os.path.basename(results_dir)}')
            plt.xlabel('Demean Layer')
            plt.ylabel('Steering Layer')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'cross_layer_heatmap_beta_{beta}.png'))
        
        # Find best configuration
        best_idx = results['accuracy'].idxmax()
        best_row = results.loc[best_idx]
        print(f"\nSummary for {results_dir} (Cross-Layer Experiment):")
        print(f"Overall mean accuracy: {results['accuracy'].mean():.4f}")
        print(f"Best accuracy: {results['accuracy'].max():.4f}")
        print(f"Best configuration: Steering Layer {best_row['steering_layer']}, " +
              f"Demean Layer {best_row['demean_layer']}, Beta {best_row['beta']}")
        
    # Check if demean_scale column exists (Run 2)
    elif 'demean_scale' in results.columns:
        # Calculate mean accuracy per layer, beta, and demean_scale
        mean_results = results.groupby(['layer', 'beta', 'demean_scale'])['accuracy'].mean().reset_index()
        
        # Create plot for each beta value
        unique_betas = mean_results['beta'].unique()
        for beta in unique_betas:
            beta_data = mean_results[mean_results['beta'] == beta]
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='layer', y='accuracy', hue='demean_scale', data=beta_data)
            plt.title(f'Mean Accuracy by Layer and Demean Scale (Beta={beta}) - {os.path.basename(results_dir)}')
            plt.xlabel('Layer')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(results_dir, f'accuracy_plot_beta_{beta}.png'))
            
        # Create heatmap of layer vs demean_scale for best beta
        best_beta = results.loc[results['accuracy'].idxmax()]['beta']
        best_beta_data = mean_results[mean_results['beta'] == best_beta]
        
        plt.figure(figsize=(10, 8))
        heatmap_data = best_beta_data.pivot(index='layer', columns='demean_scale', values='accuracy')
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=0, vmax=1.0)
        plt.title(f'Accuracy Heatmap: Layer vs Demean Scale (Beta={best_beta}) - {os.path.basename(results_dir)}')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'accuracy_heatmap.png'))
    else:
        # Original plotting code for backward compatibility
        # Calculate mean accuracy per layer and beta
        mean_results = results.groupby(['layer', 'beta'])['accuracy'].mean().reset_index()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='layer', y='accuracy', hue='beta', data=mean_results)
        plt.title(f'Mean Accuracy by Layer and Beta - {os.path.basename(results_dir)}')
        plt.xlabel('Layer')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(results_dir, 'accuracy_plot.png'))
    print(f"Plot saved to {os.path.join(results_dir, 'accuracy_plot.png')}")
    
    # Print summary statistics
    print(f"\nSummary for {results_dir}:")
    print(f"Overall mean accuracy: {results['accuracy'].mean():.4f}")
    print(f"Best accuracy: {results['accuracy'].max():.4f}")
    best_row = results.loc[results['accuracy'].idxmax()]
    
    if 'steering_layer' in results.columns and 'demean_layer' in results.columns:
        print(f"Best configuration: Steering Layer {best_row['steering_layer']}, " +
              f"Demean Layer {best_row['demean_layer']}, Beta {best_row['beta']}")
    elif 'demean_scale' in results.columns:
        print(f"Best configuration: Layer {best_row['layer']}, Beta {best_row['beta']}, Demean Scale {best_row['demean_scale']}")
    else:
        print(f"Best configuration: Layer {best_row['layer']}, Beta {best_row['beta']}")
    
if __name__ == "__main__":
    # Plot results for all run directories
    run_dirs = sorted(glob.glob("run_*"))
    
    if not run_dirs:
        print("No run directories found")
    else:
        for run_dir in run_dirs:
            plot_results(run_dir)
