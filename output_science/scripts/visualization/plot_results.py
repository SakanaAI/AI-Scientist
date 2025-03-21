#!/usr/bin/env python
"""
Results Visualization Script for AI Scientist

This script generates plots and visualizations from experiment results.
It supports different visualization types based on the experiment type.
"""

import argparse
import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize AI Scientist experiment results")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["nanoGPT", "nanoGPT_lite", "2d_diffusion", "grokking"],
        help="Experiment type to visualize",
    )
    parser.add_argument(
        "--idea-id",
        type=str,
        default=None,
        help="Specific idea ID to visualize. If not provided, visualizes all ideas.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing experiment results. Defaults to output_science/experiments/{experiment}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations. Defaults to output_science/papers/assets/{idea_id}",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format for visualizations",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline results",
    )
    return parser.parse_args()

def visualize_nanogpt_results(idea_dir, output_dir, idea_id, file_format, compare_baseline):
    """Visualize nanoGPT experiment results"""
    print(f"Visualizing nanoGPT results for idea {idea_id}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results_file = os.path.join(idea_dir, "results.json")
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Extract training data
    if "experiment_data" not in results or not results["experiment_data"]:
        print("Error: No experiment data found in results")
        return
    
    # Plot training loss
    try:
        experiment_data = results["experiment_data"]
        iterations = []
        train_losses = []
        val_losses = []
        
        for entry in experiment_data:
            if "iteration" in entry and "train_loss" in entry:
                iterations.append(entry["iteration"])
                train_losses.append(entry["train_loss"])
            if "iteration" in entry and "val_loss" in entry:
                val_losses.append(entry["val_loss"])
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_losses, label='Train Loss')
        if val_losses:
            plt.plot(iterations, val_losses, label='Validation Loss')
        
        if compare_baseline:
            # Add baseline comparison if available
            baseline_file = os.path.join(idea_dir, "baseline.json")
            if os.path.exists(baseline_file):
                with open(baseline_file, "r") as f:
                    baseline = json.load(f)
                if "train_loss" in baseline:
                    plt.axhline(y=baseline["train_loss"], color='r', linestyle='--', label='Baseline Train Loss')
                if "val_loss" in baseline:
                    plt.axhline(y=baseline["val_loss"], color='g', linestyle='--', label='Baseline Val Loss')
        
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Training Progress for Idea {idea_id}')
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        output_file = os.path.join(output_dir, f"training_loss.{file_format}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training loss plot saved to {output_file}")
    
    except Exception as e:
        print(f"Error plotting nanoGPT results: {e}")

def visualize_2d_diffusion_results(idea_dir, output_dir, idea_id, file_format, compare_baseline):
    """Visualize 2D diffusion experiment results"""
    print(f"Visualizing 2D diffusion results for idea {idea_id}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results_file = os.path.join(idea_dir, "results.json")
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Check for samples and metrics
    if "experiment_data" not in results or not results["experiment_data"]:
        print("Error: No experiment data found in results")
        return
    
    # Plot 2D distributions
    try:
        experiment_data = results["experiment_data"]
        
        # Find samples in the results
        samples_data = None
        metrics_data = None
        
        for entry in experiment_data:
            if "samples" in entry:
                samples_data = entry["samples"]
            if "metrics" in entry:
                metrics_data = entry["metrics"]
        
        if samples_data:
            # Plot samples for each dataset
            for dataset_name, samples in samples_data.items():
                if isinstance(samples, list) and len(samples) > 0:
                    # Convert to numpy array if it's a list of points
                    if isinstance(samples[0], list):
                        points = np.array(samples)
                        
                        plt.figure(figsize=(8, 8))
                        plt.scatter(points[:, 0], points[:, 1], alpha=0.5, s=10)
                        plt.title(f'Generated Samples - {dataset_name}')
                        plt.xlabel('x')
                        plt.ylabel('y')
                        plt.grid(True)
                        
                        # Save the figure
                        output_file = os.path.join(output_dir, f"{dataset_name}_samples.{file_format}")
                        plt.savefig(output_file, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Sample plot for {dataset_name} saved to {output_file}")
        
        # Plot metrics if available
        if metrics_data:
            for metric_name, metric_values in metrics_data.items():
                if isinstance(metric_values, dict):
                    # Plot metrics by dataset
                    datasets = list(metric_values.keys())
                    values = list(metric_values.values())
                    
                    plt.figure(figsize=(10, 6))
                    plt.bar(datasets, values)
                    plt.title(f'{metric_name} by Dataset')
                    plt.xlabel('Dataset')
                    plt.ylabel(metric_name)
                    plt.xticks(rotation=45)
                    
                    # Save the figure
                    output_file = os.path.join(output_dir, f"{metric_name}_comparison.{file_format}")
                    plt.savefig(output_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Metric plot for {metric_name} saved to {output_file}")
    
    except Exception as e:
        print(f"Error plotting 2D diffusion results: {e}")

def visualize_grokking_results(idea_dir, output_dir, idea_id, file_format, compare_baseline):
    """Visualize grokking experiment results"""
    print(f"Visualizing grokking results for idea {idea_id}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results_file = os.path.join(idea_dir, "results.json")
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Check for training metrics
    if "experiment_data" not in results or not results["experiment_data"]:
        print("Error: No experiment data found in results")
        return
    
    # Plot grokking curves
    try:
        experiment_data = results["experiment_data"]
        
        # Extract training and test accuracy
        epochs = []
        train_acc = []
        test_acc = []
        
        for entry in experiment_data:
            if "epoch" in entry and "train_acc" in entry and "test_acc" in entry:
                epochs.append(entry["epoch"])
                train_acc.append(entry["train_acc"])
                test_acc.append(entry["test_acc"])
        
        if epochs:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, train_acc, label='Train Accuracy')
            plt.plot(epochs, test_acc, label='Test Accuracy')
            
            if compare_baseline:
                # Add baseline comparison if available
                baseline_file = os.path.join(idea_dir, "baseline.json")
                if os.path.exists(baseline_file):
                    with open(baseline_file, "r") as f:
                        baseline = json.load(f)
                    if "train_acc" in baseline:
                        plt.axhline(y=baseline["train_acc"], color='r', linestyle='--', label='Baseline Train Acc')
                    if "test_acc" in baseline:
                        plt.axhline(y=baseline["test_acc"], color='g', linestyle='--', label='Baseline Test Acc')
            
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'Grokking Curve for Idea {idea_id}')
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            output_file = os.path.join(output_dir, f"grokking_curve.{file_format}")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Grokking curve saved to {output_file}")
            
            # Plot generalization gap
            if train_acc and test_acc:
                gen_gap = [train - test for train, test in zip(train_acc, test_acc)]
                
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, gen_gap)
                plt.xlabel('Epoch')
                plt.ylabel('Generalization Gap')
                plt.title(f'Generalization Gap for Idea {idea_id}')
                plt.grid(True)
                
                # Save the figure
                output_file = os.path.join(output_dir, f"generalization_gap.{file_format}")
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Generalization gap plot saved to {output_file}")
    
    except Exception as e:
        print(f"Error plotting grokking results: {e}")

def main():
    args = parse_arguments()
    
    # Set up directories
    output_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # output_science/
    
    if args.results_dir is None:
        args.results_dir = os.path.join(output_base, "experiments", args.experiment)
    
    # Process specific idea or all ideas
    if args.idea_id:
        idea_ids = [args.idea_id]
    else:
        # Find all idea directories
        idea_dirs = glob.glob(os.path.join(args.results_dir, "*"))
        idea_ids = [os.path.basename(d) for d in idea_dirs if os.path.isdir(d)]
    
    for idea_id in idea_ids:
        idea_dir = os.path.join(args.results_dir, idea_id)
        
        if not os.path.isdir(idea_dir):
            print(f"Warning: Idea directory not found at {idea_dir}")
            continue
        
        if args.output_dir is None:
            output_dir = os.path.join(output_base, "papers", "assets", idea_id)
        else:
            output_dir = args.output_dir
        
        # Visualize based on experiment type
        if args.experiment in ["nanoGPT", "nanoGPT_lite"]:
            visualize_nanogpt_results(idea_dir, output_dir, idea_id, args.format, args.compare_baseline)
        elif args.experiment == "2d_diffusion":
            visualize_2d_diffusion_results(idea_dir, output_dir, idea_id, args.format, args.compare_baseline)
        elif args.experiment == "grokking":
            visualize_grokking_results(idea_dir, output_dir, idea_id, args.format, args.compare_baseline)
        else:
            print(f"Unsupported experiment type: {args.experiment}")
            sys.exit(1)
    
    print("Visualization complete")

if __name__ == "__main__":
    main() 