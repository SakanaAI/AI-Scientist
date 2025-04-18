import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def plot_results(baseline_results, run_results, run_name, output_dir):
    """
    Plot comparison between baseline and current run results.
    
    Args:
        baseline_results: Dictionary of baseline results
        run_results: Dictionary of current run results
        run_name: Name of the current run
        output_dir: Directory to save plots
    """
    # Extract data
    layers = sorted(set([int(float(k.split('_')[1])) for k in baseline_results.keys()]))
    betas = sorted(set([int(float(k.split('_')[3])) for k in baseline_results.keys()]))
    
    # Check if we have demean factors in the results
    has_demean = any('demean' in k for k in run_results.keys())
    
    # Create dataframe for plotting
    data = []
    for layer in layers:
        for beta in betas:
            baseline_key = f"layer_{layer}.0_beta_{beta}.0"
            if baseline_key in baseline_results:
                data.append({
                    'Layer': layer,
                    'Beta': beta,
                    'Accuracy': baseline_results[baseline_key],
                    'Method': 'Baseline',
                    'Demean': 0.0  # Baseline has no demeaning
                })
            
            # Handle run results with or without demean factor
            if has_demean:
                # Find all keys for this layer and beta with different demean factors
                for key in run_results.keys():
                    if f"layer_{layer}" in key and f"beta_{beta}" in key:
                        # Extract demean factor from key
                        demean_parts = key.split('demean_')
                        if len(demean_parts) > 1:
                            demean_factor = float(demean_parts[1])
                            data.append({
                                'Layer': layer,
                                'Beta': beta,
                                'Accuracy': run_results[key]['means'],
                                'Method': f"{run_name} (Demean={demean_factor})",
                                'Demean': demean_factor
                            })
            else:
                # Handle results without demean factor (like previous runs)
                key = f"layer_{layer}.0_beta_{beta}.0"
                if key in run_results:
                    data.append({
                        'Layer': layer,
                        'Beta': beta,
                        'Accuracy': run_results[key]['means'],
                        'Method': run_name,
                        'Demean': 1.0  # Assume full demeaning for previous runs
                    })
    
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Plot by layer
    for i, layer in enumerate(layers):
        plt.subplot(2, 2, i+1)
        layer_data = df[df['Layer'] == layer]
        
        if has_demean:
            # Use different plot style when we have multiple demean factors
            sns.lineplot(x='Beta', y='Accuracy', hue='Method', style='Method', 
                        markers=True, dashes=False, data=layer_data)
        else:
            sns.barplot(x='Beta', y='Accuracy', hue='Method', data=layer_data)
            
        plt.title(f'Layer {layer}')
        plt.xlabel('Beta')
        plt.ylabel('Accuracy')
        plt.legend(title='Method', loc='best', fontsize='small')
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_by_layer.png'))
    
    # Plot by beta
    plt.figure(figsize=(12, 8))
    for i, beta in enumerate(betas):
        plt.subplot(1, 3, i+1)
        beta_data = df[df['Beta'] == beta]
        
        if has_demean:
            # Use different plot style when we have multiple demean factors
            sns.lineplot(x='Layer', y='Accuracy', hue='Method', style='Method',
                        markers=True, dashes=False, data=beta_data)
        else:
            sns.barplot(x='Layer', y='Accuracy', hue='Method', data=beta_data)
            
        plt.title(f'Beta {beta}')
        plt.xlabel('Layer')
        plt.ylabel('Accuracy')
        plt.legend(title='Method', loc='best', fontsize='small')
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_by_beta.png'))
    
    # If we have demean factors, create an additional plot showing the effect of demeaning
    if has_demean:
        plt.figure(figsize=(15, 10))
        
        # Plot accuracy vs demean factor for each layer and beta
        for i, layer in enumerate(layers):
            plt.subplot(2, 2, i+1)
            layer_data = df[df['Layer'] == layer]
            
            # Group by demean factor and beta
            pivot_data = layer_data.pivot(index='Demean', columns='Beta', values='Accuracy')
            
            # Plot lines for each beta value
            for beta in betas:
                if beta in pivot_data.columns:
                    plt.plot(pivot_data.index, pivot_data[beta], marker='o', label=f'Beta={beta}')
            
            plt.title(f'Layer {layer}: Effect of Demeaning')
            plt.xlabel('Demean Factor')
            plt.ylabel('Accuracy')
            plt.legend(title='Beta')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'demean_effect.png'))

if __name__ == "__main__":
    # Load baseline results
    baseline_results = {
        'layer_9.0_beta_1.0': 0.0070885481852315, 
        'layer_9.0_beta_3.0': 0.06547090112640801, 
        'layer_9.0_beta_5.0': 0.06672298706716727, 
        'layer_10.0_beta_1.0': 0.027939090529828934, 
        'layer_10.0_beta_3.0': 0.10925740508969539, 
        'layer_10.0_beta_5.0': 0.10842563621193158, 
        'layer_11.0_beta_1.0': 0.042952127659574464, 
        'layer_11.0_beta_3.0': 0.11717980809345009, 
        'layer_11.0_beta_5.0': 0.10216781393408424, 
        'layer_12.0_beta_1.0': 0.02252085940759277, 
        'layer_12.0_beta_3.0': 0.10008083020442216, 
        'layer_12.0_beta_5.0': 0.051295890696704205
    }
    
    # Find all run directories
    run_dirs = [d for d in os.listdir('.') if d.startswith('run_') and os.path.isdir(d)]
    
    for run_dir in run_dirs:
        if run_dir == 'run_0':  # Skip baseline
            continue
            
        # Load run results
        try:
            with open(os.path.join(run_dir, 'final_info.json'), 'r') as f:
                run_results = json.load(f)
                
            # Plot results
            plot_results(baseline_results, run_results, run_dir, run_dir)
            print(f"Generated plots for {run_dir}")
        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
