#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
import torch
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.llm import create_client, AVAILABLE_LLMS
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_review import perform_review, perform_improvement
from ai_scientist.perform_writeup import perform_writeup, generate_latex

NUM_REFLECTIONS = 3

def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments with custom output structure")
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        choices=AVAILABLE_LLMS,
        help="LLM to use for this experiment.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=5,
        help="Number of ideas to generate.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run ideas in parallel using multiple GPUs.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use, e.g., '0,1,2'.",
    )
    parser.add_argument(
        "--skip-experiment",
        action="store_true",
        help="Skip running experiments, just do idea generation and writeup.",
    )
    parser.add_argument(
        "--skip-writeup",
        action="store_true",
        help="Skip paper writeup, just do idea generation and experiment.",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Generate a review for the paper.",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Generate an improvement for the paper.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="s2",
        choices=["s2", "openalex"],
        help="Search engine to use for paper search. Either 's2' (Semantic Scholar) or 'openalex' (OpenAlex).",
    )
    return parser.parse_args()

def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))

def do_idea(
    base_dir,
    results_dir,
    idea,
    model,
    client,
    client_model,
    writeup,
    improvement,
    log_file=False,
):
    # Create directories
    idea_dir = os.path.join(base_dir, idea["id"])
    os.makedirs(idea_dir, exist_ok=True)
    
    # Save idea details
    idea_file = os.path.join(idea_dir, "idea.json")
    with open(idea_file, "w") as f:
        json.dump(idea, f, indent=2)
    
    # Log file handling
    if log_file:
        sys.stdout = open(os.path.join(idea_dir, "log.txt"), "w")
        sys.stderr = sys.stdout
    
    try:
        # Run experiments if not skipped
        if args.skip_experiment:
            print("Skipping experiment")
        else:
            print("Running experiments")
            print_time()
            results = perform_experiments(
                client, idea, args.experiment, model, NUM_REFLECTIONS
            )
            # Save results
            results_file = os.path.join(idea_dir, "results.json")
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
        
        # Generate paper if not skipped
        if args.skip_writeup:
            print("Skipping writeup")
        else:
            print("Running writeup")
            print_time()
            writeup_result = perform_writeup(client, idea, args.experiment, model)
            # Save writeup
            writeup_file = os.path.join(idea_dir, "writeup.json")
            with open(writeup_file, "w") as f:
                json.dump(writeup_result, f, indent=2)
            
            # Generate LaTeX
            latex_dir = os.path.join(results_dir, "papers", "final", idea["id"])
            os.makedirs(latex_dir, exist_ok=True)
            generate_latex(writeup_result, args.experiment, latex_dir)
        
        # Generate review if requested
        if args.review:
            print("Running review")
            print_time()
            review_dir = os.path.join(results_dir, "reviews", idea["id"])
            os.makedirs(review_dir, exist_ok=True)
            review = perform_review(client, idea, args.experiment, model)
            review_file = os.path.join(review_dir, "review.json")
            with open(review_file, "w") as f:
                json.dump(review, f, indent=2)
        
        # Generate improvement if requested
        if args.improvement:
            print("Running improvement")
            print_time()
            improvement_dir = os.path.join(results_dir, "papers", "drafts", idea["id"])
            os.makedirs(improvement_dir, exist_ok=True)
            improved = perform_improvement(client, idea, args.experiment, model)
            improvement_file = os.path.join(improvement_dir, "improvement.json")
            with open(improvement_file, "w") as f:
                json.dump(improved, f, indent=2)
    
    finally:
        # Reset stdout if logging to file
        if log_file:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

def main():
    # Set up output directories
    output_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # output_science/
    experiment_dir = os.path.join(output_base, "experiments", args.experiment)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up logging
    log_dir = os.path.join(output_base, "logs", "idea_generation")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create LLM client
    client, client_model = create_client(args.model)
    
    # Generate or load ideas
    ideas_file = os.path.join(experiment_dir, "ideas.json")
    
    if args.skip_idea_generation and os.path.exists(ideas_file):
        print("Loading existing ideas from", ideas_file)
        with open(ideas_file, "r") as f:
            ideas = json.load(f)
    else:
        print("Generating ideas")
        print_time()
        ideas = generate_ideas(client, args.experiment, args.model, args.num_ideas)
        with open(ideas_file, "w") as f:
            json.dump(ideas, f, indent=2)
    
    # Check novelty if requested
    if not args.skip_novelty_check:
        print("Checking novelty")
        print_time()
        for i, idea in enumerate(ideas):
            idea["is_novel"] = check_idea_novelty(client, idea, args.model, args.engine)
            # Save updated ideas with novelty check
            with open(ideas_file, "w") as f:
                json.dump(ideas, f, indent=2)
    
    # Process ideas
    if args.parallel:
        # Get available GPUs
        gpu_ids = get_available_gpus(args.gpus)
        print(f"Running in parallel on GPUs: {gpu_ids}")
        
        # TODO: Implement parallel processing using multiprocessing
        print("Parallel processing not yet implemented in this script")
        # For now, run sequentially
        for idea in ideas:
            if "is_novel" in idea and not idea["is_novel"]:
                print(f"Skipping non-novel idea {idea['id']}")
                continue
            do_idea(
                experiment_dir,
                output_base,
                idea,
                args.model,
                client,
                client_model,
                not args.skip_writeup,
                args.improvement,
                True,
            )
    else:
        for idea in ideas:
            if "is_novel" in idea and not idea["is_novel"]:
                print(f"Skipping non-novel idea {idea['id']}")
                continue
            do_idea(
                experiment_dir,
                output_base,
                idea,
                args.model,
                client,
                client_model,
                not args.skip_writeup,
                args.improvement,
                False,
            )

if __name__ == "__main__":
    args = parse_arguments()
    main() 