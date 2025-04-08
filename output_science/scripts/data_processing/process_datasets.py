#!/usr/bin/env python
"""
Data Processing Script for AI Scientist

This script processes raw datasets for use in experiments.
It handles different data sources based on the experiment type.
"""

import argparse
import os
import sys
import shutil
import json

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process datasets for AI Scientist experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["nanoGPT", "nanoGPT_lite", "2d_diffusion", "grokking"],
        help="Experiment type to process data for",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed data. Defaults to output_science/data/processed/{experiment}",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Directory containing raw data. Defaults to output_science/data/raw/{experiment}",
    )
    return parser.parse_args()

def process_nanogpt_data(raw_dir, output_dir):
    """Process data for nanoGPT experiments"""
    print(f"Processing nanoGPT data from {raw_dir} to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each dataset
    for dataset in ["enwik8", "shakespeare_char", "text8"]:
        dataset_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Call the original prepare script
        template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "templates", "nanoGPT")
        prepare_script = os.path.join(template_dir, "data", dataset, "prepare.py")
        
        if os.path.exists(prepare_script):
            # Run the prepare script with custom output directory
            cmd = f"python {prepare_script} --output_dir {dataset_dir}"
            print(f"Running: {cmd}")
            os.system(cmd)
        else:
            print(f"Warning: Prepare script not found at {prepare_script}")
    
    print("nanoGPT data processing complete")

def process_2d_diffusion_data(raw_dir, output_dir):
    """Process data for 2D diffusion experiments"""
    print(f"Processing 2D diffusion data from {raw_dir} to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 2D diffusion doesn't have external data to process
    # It generates synthetic data within the experiments
    # Just create a config file for experiments
    config = {
        "datasets": ["8gaussians", "25gaussians", "swissroll", "circles", "moons"],
        "batch_size": 64,
        "sample_size": 10000,
        "data_path": output_dir
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("2D diffusion data configuration complete")

def process_grokking_data(raw_dir, output_dir):
    """Process data for grokking experiments"""
    print(f"Processing grokking data from {raw_dir} to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Grokking doesn't have external data to process
    # It generates synthetic mathematics problems within the experiments
    # Just create a config file for experiments
    config = {
        "modulos": [97, 113, 127],
        "operations": ["addition", "multiplication"],
        "train_fraction": 0.5,
        "data_path": output_dir
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("Grokking data configuration complete")

def main():
    args = parse_arguments()
    
    # Set up directories
    output_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # output_science/
    
    if args.output_dir is None:
        args.output_dir = os.path.join(output_base, "data", "processed", args.experiment)
    
    if args.raw_dir is None:
        args.raw_dir = os.path.join(output_base, "data", "raw", args.experiment)
    
    # Create raw directory if it doesn't exist
    os.makedirs(args.raw_dir, exist_ok=True)
    
    # Process data based on experiment type
    if args.experiment in ["nanoGPT", "nanoGPT_lite"]:
        process_nanogpt_data(args.raw_dir, args.output_dir)
    elif args.experiment == "2d_diffusion":
        process_2d_diffusion_data(args.raw_dir, args.output_dir)
    elif args.experiment == "grokking":
        process_grokking_data(args.raw_dir, args.output_dir)
    else:
        print(f"Unsupported experiment type: {args.experiment}")
        sys.exit(1)
    
    print(f"Data processing complete. Processed data saved to {args.output_dir}")

if __name__ == "__main__":
    main() 