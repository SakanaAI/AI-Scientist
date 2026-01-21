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

from ai_scientist.generate_ideas_after_paper_review import generate_ideas, check_idea_novelty
from ai_scientist.llm import create_client, AVAILABLE_LLMS
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_review import perform_review, load_paper, perform_improvement
from ai_scientist.perform_writeup import perform_writeup, generate_latex
from utils_tool import save_json_data_to_file, load_json_from_file

NUM_REFLECTIONS = 3


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--skip-run-experiment",
        action="store_true",
        help="Skip experiment",
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and load existing ideas",
    )
    parser.add_argument(
        "--run-idea-dedup",
        action="store_true",
        help="run idea dedup",
    )
    parser.add_argument(
        "--exist-idea-file",
        type=str,
        help="Skip idea generation and use this exist ideas for experiment.",
    )

    parser.add_argument(
        "--skip-novelty-check",
        action="store_true",
        help="Skip novelty check and use existing ideas",
    )
    parser.add_argument(
        "--skip-write-paper",
        action="store_true",
        help="Skip writeup generation",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug only run 1 idea",
    )
    parser.add_argument(
        "--topk_for_experiment",
        type=int,
        default=3,
        help="Number of parallel processes to run. 0 for sequential execution.",
    )
    parser.add_argument(
        "--target-exp-idea-file",
        type=str,
        help="target idea file for Experiment.",
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
        help="Model to use for AI Scientist.",
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
    parser.add_argument(
        "--use-literature",
        action="store_true",
        help="Use literature review.",
    )
    parser.add_argument(
        "--lit-review-size",
        type=int,
        default=5,
        help="Number of results to use for literature review.",
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
        model,
        client,
        client_model,
        writeup,
        improvement,
        gpu_id,
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
            model,
            client,
            client_model,
            writeup,
            improvement,
            write_paper=True,  # Default to True for worker processes
            engine="semanticscholar",
            log_file=True,
        )
        print(f"Completed idea: {idea['Name']}, Success: {success}")
    print(f"Worker {gpu_id} finished.")


def do_idea(
        base_dir,
        results_dir,
        idea,
        model,
        client,
        client_model,
        writeup,
        improvement,
        write_paper=True,  # Default to True
        engine="semanticscholar",  # Add engine parameter with default
        log_file=False,
        ):
    print("do_idea | idea info:", idea)
    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    # save current idea at destination_dir
    save_json_data_to_file(
        idea,
        osp.join(destination_dir, "current_idea.json"),
    )
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
    try:
        print_time()
        print(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )
        main_model = Model('deepseek/deepseek-chat')
        # if model == 'deepseek-chat':
            # main_model = Model('deepseek/deepseek-chat')
        # if model == "deepseek-coder-v2-0724":
        #     main_model = Model("deepseek/deepseek-coder")
        # elif model == "deepseek-reasoner":
        #     main_model = Model("deepseek/deepseek-reasoner")
        # elif model == "llama3.1-405b":
        #     main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        # else:
        #     main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        print_time()
        print(f"*Starting Experiments*")
        try:
            success = perform_experiments(idea, folder_name, coder, baseline_results)
        except Exception as e:
            print(f"Error during experiments: {e}")
            print(f"Experiments failed for idea {idea_name}")
            return False

        if not success:
            print(f"Experiments failed for idea {idea_name}")
            return False

        print_time()
        if write_paper:
            print(f"*Starting Writeup*")
            ## PERFORM WRITEUP
            if writeup == "latex":
                writeup_file = osp.join(folder_name, "latex", "template.tex")
                # exp_file: experiments.py
                # notes:记录的实验结果，以及plot的结果
                fnames = [exp_file, writeup_file, notes]
                main_model = Model('deepseek/deepseek-chat')
                # if model == "deepseek-coder-v2-0724":
                #     main_model = Model("deepseek/deepseek-coder")
                # elif model == "deepseek-reasoner":
                #     main_model = Model("deepseek/deepseek-reasoner")
                # elif model == "llama3.1-405b":
                #     main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
                # else:
                #     main_model = Model(model)
                coder = Coder.create(
                    main_model=main_model,
                    fnames=fnames,
                    io=io,
                    stream=False,
                    use_git=False,
                    edit_format="diff",
                )
                try:
                    perform_writeup(idea, folder_name, coder, client, client_model, engine=engine)
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
                    review = perform_review(
                        paper_text,
                        model="gpt-4o-2024-05-13",
                        client=openai.OpenAI(),
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
                    review = perform_review(
                        paper_text,
                        model="gpt-4o-2024-05-13",
                        client=openai.OpenAI(),
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

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    review_dir = osp.join('data', "exp2_review_data", args.experiment)
    assert os.path.exists(osp.join(review_dir, "paper.txt")), f"review_dir not exists paper: {review_dir}"
    assert os.path.exists(osp.join(review_dir, "gpt4.1-runs_ratings.csv")), f"review_dir not exists gpt4.1-runs_ratings.csv: {review_dir}"

    results_dir = osp.join("results_exp2", args.experiment)

    print("args.use_literature:", args.use_literature)
    print("args.lit_review_size:", args.lit_review_size)

    if args.target_exp_idea_file:
        ideas = load_json_from_file(args.target_exp_idea_file)
        print(f"target_exp_idea_file load {len(ideas)} ideas")
    else:
        ideas = generate_ideas(
            base_dir,
            review_dir,
            client=client,
            model=client_model,
            skip_generation=args.skip_idea_generation,
            exist_idea_file=args.exist_idea_file,
            max_num_generations=args.num_ideas,
            num_reflections=NUM_REFLECTIONS,
            use_literature=args.use_literature,
            lit_review_size=args.lit_review_size,
        )
    if not args.skip_novelty_check:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
            engine=args.engine,
        )

    novel_ideas = [idea for idea in ideas if idea["novel"]]
    # novel_ideas = list(reversed(novel_ideas))

    if args.run_idea_dedup:
        from dedup_tool import get_dedup_results_from_initial_proposal_files
        final_proposals, final_ideas, repeat_idea_map = get_dedup_results_from_initial_proposal_files(
            novel_ideas,
            similarity_threshold=0.8,
            EMBEDING_SERVER_URL="http://127.0.0.1:10041/compute_embedding"
        )
        print(f"orignal ideas size:{len(novel_ideas)}, dedup ideas size:{len(final_ideas)}")
        novel_ideas = final_proposals

    if args.debug:
        novel_ideas = novel_ideas[:1]

    print(f"Running {len(novel_ideas)} novel ideas")
    # rank and select topk idea according to Interestingness and Feasibility(th>=0.8) score
    novel_ideas =[_ for _ in novel_ideas if _["Feasibility"] >= 0.8]
    novel_ideas.sort(key=lambda x: x["Interestingness"] + x["Feasibility"], reverse=True)
    novel_ideas = novel_ideas[:args.topk_for_experiment]

    print(f"Running {len(novel_ideas)} novel ideas")

    if not args.skip_run_experiment:
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
                        args.model,
                        client,
                        client_model,
                        args.writeup,
                        args.improvement,
                        gpu_id,
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
                        args.model,
                        client,
                        client_model,
                        args.writeup,
                        args.improvement,
                        not args.skip_write_paper,  # write_paper should be True when skip_write_paper is False
                        engine=args.engine,
                    )
                    print(f"Completed idea: {idea['Name']}, Success: {success}")
                except Exception as e:
                    print(f"Failed to evaluate idea {idea['Name']}: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
    print("All ideas evaluated.")
