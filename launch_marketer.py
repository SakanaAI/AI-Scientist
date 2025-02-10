import argparse
import json
import os
import os.path as osp
import shutil
import sys
from datetime import datetime

import pandas as pd
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from ai_scientist.generate_ideas import generate_ideas
from ai_scientist.llm import AVAILABLE_LLMS, create_client
from ai_scientist.perform_experiments import perform_experiments
from templates.personal_coupon.perform_user_behavior import perform_user_behavior

NUM_REFLECTIONS = 3


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI marketer experiments")
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20240620",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="coupon_simulation",
        help="Experiment to run AI Marketer on.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of PDCA cycles to perform.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=50,
        help="Number of ideas to generate",
    )
    return parser.parse_args()


def do_idea(
    base_dir,
    results_dir,
    idea,
    model,
    client,
    client_model,
    user_df,
    restaurant_df,
    log_file=False,
):
    """Perform experiments and run Aider for modification."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)

    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)

    exp_file = osp.join(folder_name, "experiment.py")
    review_file = osp.join(folder_name, "perform_user_behavior.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")

    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write("## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write("Description: Baseline results.\n")

    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log

    # Initialize Aider Coder
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]
        read_only_fnames = [review_file]
        io = InputOutput(
            yes=True,
            chat_history_file=f"{folder_name}/{idea_name}_aider.txt",
        )
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
            read_only_fnames=read_only_fnames,
        )
        print_time()
        # ---------------------------
        # --- Perform experiments ---
        # ---------------------------
        print("*Starting Experiments*")
        try:
            # TODO: Exploring methods to present suggestions to users—such as indicating
            #       which table should contain certain data or
            #       that having particular data would enable a specific outcome.
            success = perform_experiments(
                idea,
                folder_name,
                coder,
                baseline_results,
                marketing_mode=True,
            )
        except Exception as e:
            print(f"Error during experiments: {e}")
            print(f"Experiments failed for idea {idea_name}")
            return False

        if not success:
            print(f"Experiments failed for idea {idea_name}")
            return False

        print_time()
        print("*Experiments Completed*")

        # ---------------------------
        # --- Perform Review --------
        # ---------------------------
        print("*Starting Review*")
        try:
            coupon_df = pd.read_csv(osp.join(folder_name, "coupon.csv"))
            review = perform_user_behavior(
                coupon_df,
                user_df,
                restaurant_df,
                client,
                client_model,
                num_reflections=5,
                temperature=0.1,
                test_mode=True,
            )
            # Store the review in separate review.txt file
            with open(osp.join(folder_name, "review_improved.txt"), "w") as f:
                f.write(json.dumps(review))
        except Exception as e:
            print(f"Error during review: {e}")
            print(f"Review failed for idea {idea_name}")
            return False
    except Exception as e:
        print(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False
    finally:
        print("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()

    print(f"Running experiment for idea: {idea_name}")

    # Execute the experiment
    experiment_file = osp.join(folder_name, "experiment.py")
    os.system(f"python {experiment_file} --out_dir {folder_name}")

    print(f"Experiment {idea_name} completed.")

    return True


if __name__ == "__main__":
    args = parse_arguments()

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)

    # Load data
    user_df = pd.read_csv("data/personal_coupon/user.csv")
    restaurant_df = pd.read_csv("data/personal_coupon/restaurant.csv")

    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.num_ideas,
        num_reflections=NUM_REFLECTIONS,
    )

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    # TODO: replace novel idea filter with something more sophisticated
    # novel_ideas = [idea for idea in ideas if idea["novel"]]
    novel_ideas = ideas

    for idea in novel_ideas:
        print(f"Processing idea: {idea['Name']}")
        try:
            success = do_idea(
                base_dir,
                results_dir,
                idea,
                args.model,
                client,
                client_model,
                user_df,
                restaurant_df,
            )
            print(f"Completed idea: {idea['Name']}, success: {success}")
        except Exception as e:
            print(f"Failed to evaluate idea{idea['Name']}: {str(e)}")
            import traceback

            print(traceback.format_exc())

    print("All ideas processed.")
