import argparse
import json
import multiprocessing
import openai
import os
import os.path as osp
import shutil
import sys
import time
import torch
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from datetime import datetime

from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.llm import create_client, AVAILABLE_LLMS
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement
from ai_scientist.perform_writeup import perform_writeup, generate_latex

NUM_REFLECTIONS = 3


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
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
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20240620",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist (default for all phases).",
    )
    parser.add_argument(
        "--idea-model",
        type=str,
        default=None,
        choices=AVAILABLE_LLMS,
        help="Model to use for idea generation.",
    )
    parser.add_argument(
        "--experiment-model",
        type=str,
        default=None,
        choices=AVAILABLE_LLMS,
        help="Model to use for experiments.",
    )
    parser.add_argument(
        "--fast-model",
        type=str,
        default=None,
        choices=AVAILABLE_LLMS,
        help="Fast model for Level 1 implementation.",
    )
    parser.add_argument(
        "--fix-model",
        type=str,
        default=None,
        choices=AVAILABLE_LLMS,
        help="Model for Level 2 linear fixes.",
    )
    parser.add_argument(
        "--architect-model",
        type=str,
        default=None,
        choices=AVAILABLE_LLMS,
        help="Architect model for Level 4 foundation checks.",
    )
    parser.add_argument(
        "--writeup-model",
        type=str,
        default=None,
        choices=AVAILABLE_LLMS,
        help="Model to use for writeup.",
    )
    parser.add_argument(
        "--review-model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for review.",
    )
    parser.add_argument(
        "--writeup",
        type=str,
        default="latex",
        choices=["latex"],
        help="What format to use for writeup",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes to run. 0 for sequential execution.",
    )
    parser.add_argument(
        "--improvement",
        action="store_true",
        help="Improve based on reviews.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=50,
        help="Number of ideas to generate",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="semanticscholar",
        choices=["semanticscholar", "openalex"],
        help="Scholar engine to use.",
    )
    return parser.parse_args()


def get_available_gpus(gpu_ids=None):
    if gpu_ids is not None:
        return [int(gpu_id) for gpu_id in gpu_ids.split(",")]
    return list(range(torch.cuda.device_count()))


def check_latex_dependencies():
    """
    Check if required LaTeX dependencies are installed on the system.
    Returns True if all dependencies are found, False otherwise.
    """
    import shutil
    import sys

    required_dependencies = ['pdflatex', 'chktex']
    missing_deps = []

    for dep in required_dependencies:
        if shutil.which(dep) is None:
            missing_deps.append(dep)
    
    if missing_deps:
        print("Error: Required LaTeX dependencies not found:", file=sys.stderr)
        return False
    
    return True
    
def worker(
        queue,
        base_dir,
        results_dir,
        models,
        writeup,
        improvement,
        gpu_id,
        engine,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Worker {gpu_id} started.")
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            base_dir,
            results_dir,
            idea,
            models,
            writeup,
            improvement,
            log_file=True,
            engine=engine,
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    print(f"Worker {gpu_id} finished.")


def do_idea(
        base_dir,
        results_dir,
        idea,
        models,
        writeup,
        improvement,
        log_file=False,
        engine="semanticscholar",
):
    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    # Check if baseline_results is a dictionary before extracting means
    if isinstance(baseline_results, dict):
        baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
    # Create clients inside the process
    clients = {
        "idea": create_client(models["idea"]),
        "experiment": create_client(models["experiment"]),
        "writeup": create_client(models["writeup"]),
        "review": create_client(models["review"]),
        "fast": create_client(models["fast"]),
        "fix": create_client(models["fix"]),
        "architect": create_client(models["architect"]),
    }
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )

        def create_coder(model_name):
            if model_name == "deepseek-coder-v2-0724":
                m = Model("deepseek/deepseek-coder")
            elif model_name == "deepseek-reasoner":
                m = Model("deepseek/deepseek-reasoner")
            elif model_name == "llama3.1-405b":
                m = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
            else:
                m = Model(model_name)
            return Coder.create(
                main_model=m,
                fnames=fnames,
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )

        coders = {
            "experiment": create_coder(models["experiment"]),
            "fast": create_coder(models["fast"]),
            "fix": create_coder(models["fix"]),
            "architect": create_coder(models["architect"]),
        }

        print_time()
        print(f"*Starting Experiments*")
        try:
            success = perform_experiments(idea, folder_name, coders, baseline_results)
        except Exception as e:
            print(f"Error during experiments: {e}")
            print(f"Experiments failed for idea {idea_name}")
            return False

        if not success:
            print(f"Experiments failed for idea {idea_name}")
            return False

        print_time()
        print(f"*Starting Writeup*")
        ## PERFORM WRITEUP
        if writeup == "latex":
            writeup_file = osp.join(folder_name, "latex", "template.tex")
            fnames = [exp_file, writeup_file, notes]
            writeup_model = models["writeup"]
            if writeup_model == "deepseek-coder-v2-0724":
                main_model = Model("deepseek/deepseek-coder")
            elif writeup_model == "deepseek-reasoner":
                main_model = Model("deepseek/deepseek-reasoner")
            elif writeup_model == "llama3.1-405b":
                main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
            else:
                main_model = Model(writeup_model)
            coder = Coder.create(
                main_model=main_model,
                fnames=fnames,
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )
            try:
                writeup_client, writeup_client_model = clients["writeup"]
                perform_writeup(idea, folder_name, coder, writeup_client, writeup_client_model, engine=engine)
            except Exception as e:
                print(f"Failed to perform writeup: {e}")
                return False
            print("Done writeup")
        else:
            raise ValueError(f"Writeup format {writeup} not supported.")

        print_time()
        print(f"*Starting Review*")
        ## REVIEW PAPER
        if writeup == "latex":
            try:
                paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
                review_client, review_client_model = clients["review"]
                review = perform_review(
                    paper_text,
                    model=review_client_model,
                    client=review_client,
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review.txt"), "w") as f:
                    f.write(json.dumps(review, indent=4))
            except Exception as e:
                print(f"Failed to perform review: {e}")
                return False

        ## IMPROVE WRITEUP
        if writeup == "latex" and improvement:
            print_time()
            print(f"*Starting Improvement*")
            try:
                perform_improvement(review, coder)
                generate_latex(
                    coder, folder_name, f"{folder_name}/{idea['Name']}_improved.pdf"
                )
                paper_text = load_paper(f"{folder_name}/{idea['Name']}_improved.pdf")
                review_client, review_client_model = clients["review"]
                review = perform_review(
                    paper_text,
                    model=review_client_model,
                    client=review_client,
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review_improved.txt"), "w") as f:
                    f.write(json.dumps(review))
            except Exception as e:
                print(f"Failed to perform improvement: {e}")
                return False
        return True
    except Exception as e:
        print(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False
    finally:
        print("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


if __name__ == "__main__":
    args = parse_arguments()

    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus(args.gpus)
    if args.parallel > len(available_gpus):
        print(
            f"Warning: Requested {args.parallel} parallel processes, but only {len(available_gpus)} GPUs available. Adjusting to {len(available_gpus)}."
        )
        args.parallel = len(available_gpus)

    print(f"Using GPUs: {available_gpus}")

    # Check LaTeX dependencies before proceeding
    if args.writeup == "latex" and not check_latex_dependencies():
        sys.exit(1)

    # Determine models to use for each phase
    models = {
        "idea": args.idea_model or args.model,
        "experiment": args.experiment_model or args.model,
        "writeup": args.writeup_model or args.model,
        "review": args.review_model,
        "fast": args.fast_model or args.experiment_model or args.model,
        "fix": args.fix_model or args.experiment_model or args.model,
        "architect": args.architect_model or args.experiment_model or args.model,
    }

    # Create clients for each phase
    clients = {
        "idea": create_client(models["idea"]),
        "experiment": create_client(models["experiment"]),
        "writeup": create_client(models["writeup"]),
        "review": create_client(models["review"]),
        "fast": create_client(models["fast"]),
        "fix": create_client(models["fix"]),
        "architect": create_client(models["architect"]),
    }

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)

    idea_client, idea_client_model = clients["idea"]
    ideas = generate_ideas(
        base_dir,
        client=idea_client,
        model=idea_client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.num_ideas,
        num_reflections=NUM_REFLECTIONS,
    )
    if not args.skip_novelty_check:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=idea_client,
            model=idea_client_model,
            engine=args.engine,
        )

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    novel_ideas = [idea for idea in ideas if idea["novel"]]
    # novel_ideas = list(reversed(novel_ideas))

    if args.parallel > 0:
        print(f"Running {args.parallel} parallel processes")
        queue = multiprocessing.Queue()
        for idea in novel_ideas:
            queue.put(idea)

        processes = []
        for i in range(args.parallel):
            gpu_id = available_gpus[i % len(available_gpus)]
            p = multiprocessing.Process(
                target=worker,
                args=(
                    queue,
                    base_dir,
                    results_dir,
                    models,
                    args.writeup,
                    args.improvement,
                    gpu_id,
                    args.engine,
                ),
            )
            p.start()
            time.sleep(150)
            processes.append(p)

        # Signal workers to exit
        for _ in range(args.parallel):
            queue.put(None)

        for p in processes:
            p.join()

        print("All parallel processes completed.")
    else:
        for idea in novel_ideas:
            print(f"Processing idea: {idea['Name']}")
            try:
                success = do_idea(
                    base_dir,
                    results_dir,
                    idea,
                    models,
                    args.writeup,
                    args.improvement,
                    engine=args.engine,
                )
                print(f"Completed idea: {idea['Name']}, Success: {success}")
            except Exception as e:
                print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")
                import traceback
                print(traceback.format_exc())
    print("All ideas evaluated.")
