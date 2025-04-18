import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
from typing import Optional

def plot_results(results_dir: str, baseline_accuracy: float = 0.1):
    """
    Plot the results from the demeaned probe experiments.
    
    Args:
        results_dir (str): Directory containing the results CSV files
        baseline_accuracy (float): Baseline accuracy to show as reference
    """
    # Find all results.csv files in subdirectories
    result_files = glob.glob(f"{results_dir}/results.csv")
    if not result_files:
        result_files = glob.glob(f"{results_dir}/*/results.csv")
    
    if not result_files:
        print(f"No results found in {results_dir}")
        return
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, len(result_files), figsize=(15, 5), sharey=True)
    if len(result_files) == 1:
        axes = [axes]
    
    # Process each results file
    for i, result_file in enumerate(sorted(result_files)):
        run_name = os.path.basename(os.path.dirname(result_file))
        
        # Load results
        df = pd.read_csv(result_file)
        
        # Check if we have 'layer' or 'layers' column
        layer_col = 'layer' if 'layer' in df.columns else 'layers'
        
        # Calculate average accuracy per layer and beta
        avg_results = df.groupby([layer_col, 'beta'])['accuracy'].mean().reset_index()
        
        # Reshape for heatmap
        pivot_df = avg_results.pivot(index='beta', columns=layer_col, values='accuracy')
        
        # Plot heatmap
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[i], 
                    vmin=baseline_accuracy, vmax=max(0.5, pivot_df.max().max()))
        axes[i].set_title(f"Run: {run_name}")
        axes[i].set_xlabel("Layer(s)")
        
        if i == 0:
            axes[i].set_ylabel("Beta (Scaling Factor)")
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/accuracy_heatmap.png")
    print(f"Saved heatmap to {results_dir}/accuracy_heatmap.png")
    plt.close()
    
    # Create a line plot comparing the best configurations across runs
    plt.figure(figsize=(10, 6))
    
    # Add baseline
    plt.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline (10%)')
    
    for result_file in sorted(result_files):
        run_name = os.path.basename(os.path.dirname(result_file))
        df = pd.read_csv(result_file)
        
        # Get average accuracy per beta
        avg_by_beta = df.groupby('beta')['accuracy'].mean()
        
        plt.plot(avg_by_beta.index, avg_by_beta.values, marker='o', label=run_name)
    
    plt.xlabel('Beta (Scaling Factor)')
    plt.ylabel('Average Accuracy')
    plt.title('Accuracy vs Beta Across Runs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{results_dir}/beta_comparison.png")
    print(f"Saved beta comparison to {results_dir}/beta_comparison.png")
    plt.close()
    
    # Create a new plot showing layer combinations performance
    if 'layers' in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Group by layers and get average accuracy
        layer_performance = df.groupby('layers')['accuracy'].mean().sort_values(ascending=False)
        
        # Plot as horizontal bar chart
        ax = layer_performance.plot(kind='barh', color='skyblue')
        plt.axvline(x=baseline_accuracy, color='r', linestyle='--', label='Baseline (10%)')
        
        plt.xlabel('Average Accuracy')
        plt.ylabel('Layer Combination')
        plt.title('Performance by Layer Combination')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/layer_performance.png")
        print(f"Saved layer performance chart to {results_dir}/layer_performance.png")
        plt.close()

def analyze_predictions(results_dir: str, output_file: Optional[str] = None):
    """
    Analyze prediction files to identify patterns in correct/incorrect predictions.
    
    Args:
        results_dir (str): Directory containing prediction CSV files
        output_file (str, optional): File to save analysis results
    """
    prediction_files = glob.glob(f"{results_dir}/predictions_*.csv")
    
    if not prediction_files:
        print(f"No prediction files found in {results_dir}")
        return
    
    analysis_results = []
    
    for pred_file in prediction_files:
        # Extract configuration from filename
        filename = os.path.basename(pred_file)
        parts = filename.replace('.csv', '').split('_')
        if len(parts) >= 4:
            split = parts[1]
            layers = parts[2]
            beta = parts[3]
        else:
            continue
            
        # Load predictions
        preds_df = pd.read_csv(pred_file)
        
        # Calculate accuracy
        correct = [p.startswith(t) for p, t in zip(preds_df['prediction'], preds_df['target'])]
        preds_df['correct'] = correct
        accuracy = sum(correct) / len(correct)
        
        # Analyze error patterns
        error_df = preds_df[~preds_df['correct']]
        common_errors = {}
        if len(error_df) > 0:
            for _, row in error_df.iterrows():
                target = row['target']
                pred = row['prediction']
                if target in common_errors:
                    common_errors[target].append(pred)
                else:
                    common_errors[target] = [pred]
        
        # Summarize results
        result = {
            'split': split,
            'layers': layers,
            'beta': beta,
            'accuracy': accuracy,
            'total_samples': len(preds_df),
            'correct_samples': sum(correct),
            'error_samples': len(error_df),
            'common_errors': common_errors
        }
        
        analysis_results.append(result)
    
    # Save or print analysis
    if output_file:
        with open(output_file, 'w') as f:
            for result in analysis_results:
                f.write(f"Configuration: Split {result['split']}, Layers {result['layers']}, Beta {result['beta']}\n")
                f.write(f"Accuracy: {result['accuracy']:.4f} ({result['correct_samples']}/{result['total_samples']})\n")
                f.write("Common errors:\n")
                for target, preds in result['common_errors'].items():
                    f.write(f"  Target: '{target}', Predictions: {preds[:5]}\n")
                f.write("\n")
        print(f"Analysis saved to {output_file}")
    else:
        # Find best configuration
        best_result = max(analysis_results, key=lambda x: x['accuracy'])
        print(f"Best configuration: Split {best_result['split']}, Layers {best_result['layers']}, Beta {best_result['beta']}")
        print(f"Best accuracy: {best_result['accuracy']:.4f} ({best_result['correct_samples']}/{best_result['total_samples']})")

if __name__ == "__main__":
    # Plot results and analyze predictions
    results_dir = "results"
    plot_results(results_dir)
    analyze_predictions(results_dir, f"{results_dir}/prediction_analysis.txt")
