import dataclasses
import datetime
import json
import logging
import pickle
import sys
import time
from functools import partial
from pathlib import Path

import hydra
import treequest as tq
from dotenv import load_dotenv
from omegaconf import DictConfig
from tqdm import tqdm

from ab_mcts_arc2.llm.llm_builder import call_llm
from ab_mcts_arc2.llm_generation_interface import GenerationRequest, GenerationResult
from ab_mcts_arc2.prompts.base import PromptTemplate
from ab_mcts_arc2.prompts.prompt_configs import PromptConfig
from ab_mcts_arc2.tasks.arc.task import ARCProblem

sys.path.append(str(Path(__file__)))
from prompt import BaselinePrompt
from utils import NodeState, is_power_of_two

sys.setrecursionlimit(
    20000
)  # Example: Increase limit to 20000.  Choose a sensible value.


logger = logging.getLogger(__name__)

# Global variables to track total cost
total_cost = 0.0
cost_by_model: dict[str, float] = {}

# Global variables to track execution time
total_time = 0.0
time_by_model: dict[str, float] = {}
node_times: list[float] = []


def generate_fn(
    state: NodeState | None,
    task: ARCProblem,
    prompt_template: PromptTemplate,
    model_name: str,
    model_temp: float,
    llm_log_dir: Path,
) -> tuple[NodeState, float]:
    global total_cost, cost_by_model, time_by_model, node_times

    start_time = time.time()

    # From root
    if state is None:
        messages = [{"role": "user", "content": prompt_template.initial_prompt()}]
    else:
        feedback_prompt = prompt_template.feedback_prompt(
            "transform",
            eval_results=state.eval_results,
            generation_result=state.generation_result,
        )
        messages = [
            {"role": "user", "content": feedback_prompt},
        ]

    generation, cost = call_llm(model_name, model_temp, messages)

    # Update cost info
    total_cost += cost
    if model_name not in cost_by_model:
        cost_by_model[model_name] = 0.0
    cost_by_model[model_name] += cost

    result = GenerationResult(
        request=GenerationRequest(messages=messages), generation=generation
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
        :-3
    ]  # up to milliseconds

    log_txt = llm_log_dir / f"log_{timestamp}_{model_name}.txt"
    log_txt.write_text(
        json.dumps(
            {"model": model_name, "cost": cost, "result": dataclasses.asdict(result)},
            indent=4,
        )
    )  # save cost and result

    eval_results = task.generate_eval_results(llm_answer=result, kind="transform")
    if eval_results is None:
        score = 0.0
    else:
        score = sum([eval_result.get_score() for eval_result in eval_results]) / len(
            eval_results
        )

    # Calculate execution time for this node
    execution_time = time.time() - start_time
    node_times.append(execution_time)

    # Update time by model
    if model_name not in time_by_model:
        time_by_model[model_name] = 0.0
    time_by_model[model_name] += execution_time

    return NodeState(
        generation_result=result, eval_results=eval_results, model_name=model_name
    ), score


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    global total_cost, cost_by_model, total_time, time_by_model, node_times

    # Reset global cost and time trackers
    total_cost = 0.0
    cost_by_model = {}
    total_time = 0.0
    time_by_model = {}
    node_times = []

    start_time = time.time()

    # Task
    task_id = str(cfg["task_id"])
    if task_id == "inf":
        # task_id: 13e47133 is recognized as inf
        import os

        task_id = os.getenv("TASK_ID")
    arc_problem_path = Path(f"ARC-AGI/data/evaluation/{task_id}.json")
    if not arc_problem_path.exists():
        print(f"Task {task_id} not found")
        sys.exit(1)
    task = ARCProblem.load_file(arc_problem_path)

    # prompt
    prompt_template = BaselinePrompt(prompt_config=PromptConfig(), problem=task)

    # Models
    models_config = cfg["models"]
    if isinstance(models_config, dict):
        models_config = [models_config]

    # Algo
    algo_config = cfg["algo"]
    algo_cls = getattr(tq, algo_config["class_name"])
    algo: tq.Algorithm = algo_cls(**algo_config["params"])

    print(algo)

    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    for subdir in ["llm_logs", "costs", "checkpoints"]:
        if not (save_dir / subdir).exists():
            (save_dir / subdir).mkdir()

    llm_log_dir = save_dir / "llm_logs"

    generate_fns = {
        model_config["name"]: partial(
            generate_fn,
            task=task,
            model_name=model_config["name"],
            model_temp=model_config["temperature"],
            llm_log_dir=llm_log_dir,
            prompt_template=prompt_template,
        )
        for model_config in models_config
    }

    if not cfg["checkpoint_path"]:
        search_tree = algo.init_tree()
        print("Initialized state")
    else:
        with open(cfg["checkpoint_path"], "rb") as f:
            search_tree = pickle.load(f)
        print(f"Loaded checkpoint from {cfg['checkpoint_path']}")
        # get cost so far
        if (save_dir / "cost_summary.json").exists():
            with open(save_dir / "cost_summary.json", "r") as f:
                cost_summary = json.load(f)
                total_cost = cost_summary["total_cost"]
                cost_by_model = cost_summary["cost_by_model"]

        # get time so far if available
        time_summary_path = save_dir / "time_summary.json"
        if time_summary_path.exists():
            with open(time_summary_path, "r") as f:
                time_summary = json.load(f)
                total_time = time_summary.get("total_time", 0.0)
                time_by_model = time_summary.get("time_by_model", {})
                node_times = time_summary.get("node_times", [])

    max_num_nodes = cfg["max_num_nodes"]
    initial_num_nodes = len(algo.get_state_score_pairs(search_tree))
    for i in tqdm(range(max_num_nodes - initial_num_nodes)):
        node_start_time = time.time()
        search_tree = algo.step(search_tree, generate_fns)
        n_answers = len(algo.get_state_score_pairs(search_tree))

        # Update total time
        if i >= len(node_times):  # only add if not loaded from checkpoint
            node_execution_time = time.time() - node_start_time
            node_times.append(node_execution_time)

        if n_answers % 10 == 0 or is_power_of_two(n_answers):
            with open(
                save_dir / "checkpoints" / f"checkpoint_n_answers_{n_answers}.pkl", "wb"
            ) as f:
                pickle.dump(search_tree, f)
            with open(save_dir / "checkpoints" / "checkpoint_latest.pkl", "wb") as f:
                pickle.dump(search_tree, f)

            # Update total time
            total_time = time.time() - start_time

            # Log accumulated cost and time every 10 steps
            logger.info(f"Current total cost: ${total_cost:.6f}")
            logger.info(f"Current total time: {total_time:.2f} seconds")
            for model, model_cost in cost_by_model.items():
                logger.info(f"  {model} cost: ${model_cost:.6f}")
            for model, model_time in time_by_model.items():
                logger.info(f"  {model} time: {model_time:.2f} seconds")

            # Save cost summary to a JSON file
            cost_summary = {"total_cost": total_cost, "cost_by_model": cost_by_model}
            with open(
                save_dir / "costs" / f"cost_summary_n_answers_{n_answers}.json", "w"
            ) as f:
                json.dump(cost_summary, f, indent=2)
            with open(save_dir / "costs" / "cost_summary.json", "w") as f:
                json.dump(cost_summary, f, indent=2)

            # Save time summary to a JSON file
            time_summary = {
                "total_time": total_time,
                "total_time_minutes": total_time / 60,
                "total_time_hours": total_time / 3600,
                "time_by_model": time_by_model,
                "time_by_model_minutes": {
                    model: time / 60 for model, time in time_by_model.items()
                },
                "node_times": node_times,
                "avg_node_time": sum(node_times) / len(node_times) if node_times else 0,
                "avg_node_time_minutes": (
                    sum(node_times) / len(node_times) if node_times else 0
                )
                / 60,
            }
            with open(
                save_dir / "costs" / f"time_summary_n_answers_{n_answers}.json", "w"
            ) as f:
                json.dump(time_summary, f, indent=2)
            with open(save_dir / "costs" / "time_summary.json", "w") as f:
                json.dump(time_summary, f, indent=2)

    # Update final total time
    total_time = time.time() - start_time

    # Log the final total cost and time
    logger.info("===== Final Cost Summary =====")
    logger.info(f"Total LLM cost: ${total_cost:.6f}")
    for model, model_cost in cost_by_model.items():
        logger.info(f"  {model}: ${model_cost:.6f}")

    logger.info("===== Final Time Summary =====")
    logger.info(
        f"Total execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes, {total_time / 3600:.2f} hours)"
    )
    logger.info(
        f"Average node time: {sum(node_times) / len(node_times) if node_times else 0:.2f} seconds ({(sum(node_times) / len(node_times) if node_times else 0) / 60:.2f} minutes)"
    )
    for model, model_time in time_by_model.items():
        logger.info(
            f"  {model}: {model_time:.2f} seconds ({model_time / 60:.2f} minutes)"
        )

    # Save cost summary to a JSON file
    cost_summary = {"total_cost": total_cost, "cost_by_model": cost_by_model}
    with open(save_dir / "cost_summary.json", "w") as f:
        json.dump(cost_summary, f, indent=2)

    # Save time summary to a JSON file
    time_summary = {
        "total_time": total_time,
        "total_time_minutes": total_time / 60,
        "total_time_hours": total_time / 3600,
        "time_by_model": time_by_model,
        "time_by_model_minutes": {
            model: time / 60 for model, time in time_by_model.items()
        },
        "node_times": node_times,
        "avg_node_time": sum(node_times) / len(node_times) if node_times else 0,
        "avg_node_time_minutes": (
            sum(node_times) / len(node_times) if node_times else 0
        )
        / 60,
    }
    with open(save_dir / "time_summary.json", "w") as f:
        json.dump(time_summary, f, indent=2)


if __name__ == "__main__":
    load_dotenv(Path(__file__).parents[1].resolve())
    main()
