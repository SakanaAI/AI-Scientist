import abc
import argparse
import copy
import dataclasses
import datetime
import json
import logging
import os
import pickle
import sys
import time
import warnings
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from logging import getLogger
from math import exp, log, sqrt
from pathlib import Path
from random import shuffle
from typing import Generic, TypeVar, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import NodeState, is_power_of_two

from ab_mcts_arc2.data_types import Action
from ab_mcts_arc2.eval_result import EvalResultWithAns
from ab_mcts_arc2.llm.llm_builder import call_llm
from ab_mcts_arc2.llm_generation_interface import GenerationRequest, GenerationResult
from ab_mcts_arc2.prompts.arc.grid_repr import list_format
from ab_mcts_arc2.prompts.base import PromptTemplate
from ab_mcts_arc2.prompts.prompt_configs import PromptConfig
from ab_mcts_arc2.tasks.arc.task import ARCProblem
from treequest.algos.tree import Node, Tree
from treequest.types import GenerateFnType, StateScoreType

logger = getLogger(__name__)


sys.path.append("./experiments/arc2")
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__)))


# stopwords from nltk.corpus.stopwords.words('english')
STOPWORDS = {'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'don', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves'}

def tokenize(code: str) -> set[str]:
    """Splits on non-alphanumerics and drops stopwords."""
    tokens = set(re.split(r'\W+', code.lower()))
    return {token for token in tokens if token and token not in STOPWORDS}

def jaccard(s1: set[str], s2: set[str]) -> float:
    """Computes Jaccard similarity between two sets."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def code_similarity(c1: str, c2: str) -> float:
    """Computes Jaccard similarity between the token sets of two code snippets."""
    return jaccard(tokenize(c1), tokenize(c2))

def grid_similarity(g1: list[list[int]], g2: list[list[int]]) -> float:
    """Computes the fraction of equal cells between two grids. Returns 1 when shapes differ."""
    try:
        g1_np = np.array(g1)
        g2_np = np.array(g2)
    except Exception:
        return 1.0

    if g1_np.shape != g2_np.shape:
        return 1.0
    if g1_np.size == 0 and g2_np.size == 0:
        return 1.0
    if g1_np.size == 0 or g2_np.size == 0:
        return 0.0
    
    return np.sum(g1_np == g2_np) / g1_np.size

def novelty_bonus(item: Any, history: list[Any], mode: str) -> float:
    """Computes novelty bonus for an item against a history."""
    if not history:
        return 1.0
    
    if mode == 'code':
        similarities = [code_similarity(item, h) for h in history]
    elif mode == 'output':
        similarities = [grid_similarity(item, h) for h in history]
    else:
        return 1.0

    if not similarities:
        return 1.0

    max_similarity = max(similarities)
    return 1.0 - max_similarity


##########################
## Algorithm Base Class ##
##########################

# Type variables for node state and algorithm state
NodeStateT = TypeVar("NodeStateT")
AlgoStateT = TypeVar("AlgoStateT")


class Algorithm(Generic[NodeStateT, AlgoStateT], abc.ABC):
    """
    Algorithm base class for tree search.

    The Algorithm object itself should be stateless, other than the algorithm configuration which should be specified at object instantiation time.
    The state should be maintained and saved by the caller of `step` function.
    """

    @abc.abstractmethod
    def step(
        self,
        state: AlgoStateT,
        generate_fn: Mapping[str, GenerateFnType[NodeStateT]],
        inplace: bool = False,
    ) -> AlgoStateT:
        """
        Generate one additional node and add that to a given state.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def init_tree(self) -> AlgoStateT:
        """
        Initialize the AlgoState, e.g. creating the root-only tree etc.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_state_score_pairs(self, state: AlgoStateT) -> list[StateScoreType[NodeStateT]]:
        """
        Get all the state-score pairs of the tree.
        """
        raise NotImplementedError()


# Type variable for state
StateT = TypeVar("StateT")


###################
## Standard MCTS ##
###################


def softmax(values: list[float]) -> list[float]:
    """
    Compute softmax values for a list of scores.

    Args:
        values: List of scores

    Returns:
        List of softmax probabilities
    """
    # Shift values for numerical stability (prevent overflow)
    shifted = [x - max(values) for x in values]
    exp_values = [exp(x) for x in shifted]
    sum_exp = sum(exp_values)
    return [x / sum_exp for x in exp_values]


@dataclass
class MCTSState(Generic[StateT]):
    """State for Monte Carlo Tree Search algorithm."""

    tree: Tree[StateT]
    visit_counts: dict[int, int] = field(default_factory=dict)
    value_sums: dict[int, float] = field(default_factory=dict)
    priors: dict[int, float] = field(default_factory=dict)
    next_nodes: list[tuple[Node[StateT], str]] = field(default_factory=list)


class StandardMCTS(Algorithm[StateT, MCTSState[StateT]]):
    """
    Standard Monte Carlo Tree Search (MCTS) algorithm with UCT scoring.

    This implementation uses the Upper Confidence Bound for Trees (UCT)
    formula to balance exploration and exploitation.
    """

    def __init__(self, *, samples_per_action: int = 2, exploration_weight: float = sqrt(2)):
        """
        Initialize the MCTS algorithm.

        Args:
            samples_per_action: Number of samples to generate for each action
            exploration_weight: Weight for the exploration term in UCT formula
        """
        self.samples_per_action = samples_per_action
        self.exploration_weight = exploration_weight

    def step(
        self,
        state: MCTSState,
        generate_fn: Mapping[str, GenerateFnType[StateT]],
        inplace: bool = False,
    ) -> MCTSState:
        """
        Perform one step of the MCTS algorithm.

        Args:
            state: Current algorithm state
            generate_fn: Mapping of action names to generation functions

        Returns:
            Updated algorithm state
        """
        if not inplace:
            state = copy.deepcopy(state)

        # If no nodes are queued for expansion, select nodes to expand
        if not state.next_nodes:
            # Selection: Find the most promising node to expand
            node = self._select(state)

            # Create pairs of (node, action) for all actions
            pairs_to_add = []
            actions = list(generate_fn.keys())

            # For each sample, add all actions
            for _ in range(self.samples_per_action):
                for action in actions:
                    pairs_to_add.append((node, action))

            # Shuffle the pairs to add randomness to expansion order
            shuffle(pairs_to_add)
            state.next_nodes.extend(pairs_to_add)

        # Get the next node and action to expand
        node, action = state.next_nodes.pop(0)

        # Simulation: Generate a new state using the selected action
        new_state, new_score = generate_fn[action](node.state)

        # Add the new node to the tree
        new_node = state.tree.add_node((new_state, new_score), node)

        # Update statistics for the new node
        node_id = new_node.expand_idx
        state.visit_counts[node_id] = 1
        state.value_sums[node_id] = new_score

        # Backpropagation: Update statistics for all nodes in the path
        self._backpropagate(state, new_node, new_score)

        # Update priors if this node has siblings
        parent = new_node.parent
        if parent and len(parent.children) > 1:
            self._update_priors(state, parent)

        return state

    def _update_priors(self, state: MCTSState, parent: Node) -> None:
        """
        Update prior probabilities for all children of a node using softmax.

        Args:
            state: Current algorithm state
            parent: Parent node whose children's priors will be updated
        """
        children = parent.children
        scores = [child.score for child in children]
        priors = softmax(scores)

        for child, prior in zip(children, priors):
            state.priors[child.expand_idx] = prior

    def _select(self, state: MCTSState) -> Node:
        """
        Select a node to expand using UCT.

        Starts from the root and selects child nodes with highest UCT score
        until reaching a leaf node or a node with unexplored actions.

        Args:
            state: Current algorithm state

        Returns:
            Selected node
        """
        node = state.tree.root

        # If the tree is empty, return the root
        if not node.children:
            return node

        # Traverse down the tree selecting best child according to UCT
        while node.children:
            # We're selecting based on the UCT score, which balances exploration and exploitation.
            best_child = max(node.children, key=lambda child: self._uct_score(state, child, node))
            node = best_child

        return node

    def _uct_score(self, state: MCTSState, node: Node, parent: Node) -> float:
        """
        Calculate the UCT score for a node.

        UCT = prior * average_value + exploration_weight * sqrt(log(parent_visits) / node_visits)

        Args:
            state: Current algorithm state
            node: Node to calculate score for
            parent: Parent node

        Returns:
            UCT score
        """
        # Get visit counts
        parent_visits = state.visit_counts.get(parent.expand_idx, 1)
        node_visits = state.visit_counts.get(node.expand_idx, 1)

        # Get value sum
        value_sum = state.value_sums.get(node.expand_idx, 0)

        # Calculate exploitation term
        exploitation = value_sum / node_visits

        # Get prior (default to 1.0 if not set)
        prior = state.priors.get(node.expand_idx, 1.0)

        # Calculate exploration term (weighted by prior)
        exploration = self.exploration_weight * prior * sqrt(log(parent_visits) / node_visits)

        return exploitation + exploration

    def _backpropagate(self, state: MCTSState, node: Node, score: float) -> None:
        """
        Update statistics for all nodes in the path from node to root.

        Args:
            state: Current algorithm state
            node: Leaf node to start backpropagation from
            score: Score to backpropagate
        """
        current: Node | None = node
        while current is not None:
            node_id = current.expand_idx
            state.visit_counts[node_id] = state.visit_counts.get(node_id, 0) + 1
            state.value_sums[node_id] = state.value_sums.get(node_id, 0) + score
            current = current.parent

    def init_tree(self) -> MCTSState:
        """
        Initialize the algorithm state with an empty tree.

        Returns:
            Initial algorithm state
        """
        tree: Tree = Tree.with_root_node()
        return MCTSState(tree=tree)

    def get_state_score_pairs(self, state: MCTSState) -> list[StateScoreType[StateT]]:
        """
        Get all the state-score pairs from the tree.

        Args:
            state: Current algorithm state

        Returns:
            List of (state, score) pairs
        """
        return state.tree.get_state_score_pairs()


#####################
## Baseline Prompt ##
#####################


class BaselinePrompt(PromptTemplate):
    version = "baseline"

    def __init__(self, prompt_config: PromptConfig, problem: ARCProblem):
        self.problem = problem

    def initial_prompt(self) -> str:
        prompt = initial_prompt()
        prompt += problem_prompt(self.problem)
        return prompt

    def feedback_prompt(
        self,
        action: Action,
        eval_results: list[EvalResultWithAns],
        generation_result: GenerationResult,
    ) -> str:
        try:
            code = generation_result.parse_python_code()
        except Exception:
            code = ""
        match action:
            case "transform":
                return transform_feedback_prompt(
                    problem=self.problem,
                    eval_results=eval_results,
                    pycode=code,
                )
            case _:
                raise NotImplementedError(f"feedback_prompt not implemented for action {action}")

    def add_next_action_instruction(self, action: Action, next_prompt: GenerationRequest) -> GenerationRequest:
        last_user_msg = next_prompt.messages[-1]
        assert last_user_msg.role == "user"
        # Only use the last user message
        next_prompt.messages = next_prompt.messages[-1:]

        return next_prompt


def problem_prompt(problem: ARCProblem) -> str:
    prompt = ""
    for i, demo in enumerate(problem.demos):
        prompt += f"""
# Example {i + 1}

## Input
{list_format(demo["input"])}

## Output
{list_format(demo["output"])}

"""
    for i, test in enumerate(problem.tests):
        prompt += f"""
# Additional Input {i + 1}
{list_format(test["input"])}

"""
    return prompt


def initial_prompt() -> str:
    task_explanation = """
You will be given some number of paired example inputs and outputs. The outputs were produced by applying a transformation rule to the inputs. In addition to the paired example inputs and outputs, there is also one additional input without a known output. Your task is to determine the transformation rule and implement it in code.

The inputs and outputs are each "grids". A grid is a rectangular matrix of integers between 0 and 9 (inclusive). These grids will be shown to you as grids of numbers (ASCII). Each number corresponds to a color. The correspondence is as follows: black: 0, blue: 1, red: 2, green: 3, yellow: 4, grey: 5, pink: 6, orange: 7, purple: 8, brown: 9.

The transformation only needs to be unambiguous and applicable to the example inputs and the additional input. It doesn't need to work for all possible inputs.
"""

    reasoning_instruction = """
You'll need to carefully reason in order to determine the transformation rule. Start your response by carefully reasoning in <reasoning></reasoning> tags. Then, implement the transformation in code.

After your reasoning write code in triple backticks (```python and then ```). You should write a function called `transform` which takes a single argument, the input grid as `list[list[int]]`, and returns the transformed grid (also as `list[list[int]]`). You should make sure that you implement a version of the transformation which works in general (it shouldn't just work for the additional input).
"""

    other_instruction = """
Don't write tests in your python code, just output the `transform` function. (It will be tested later.)

You can also ask question to verify your observation on the inputs/outputs patterns in the form of python function which takes two arguments, the input and expected output grid both as `list[list[int]]` and returns the boolean flag (True or False). We will help you by running your Python function on examples and let you know whether your question is True or False.

You follow a particular reasoning style. You break down complex problems into smaller parts and reason through them step by step, arriving at sub-conclusions before stating an overall conclusion. This reduces the extent to which you need to do large leaps of reasoning.

You reason in substantial detail for as is necessary to determine the transformation rule.

You are creative and accomplished at solving puzzles. When you write `transform`, do not hardcode the solution for each example. We will run your transform function on additional inputs later and check if your logic is generic in addition to check the correctness.
"""

    return task_explanation + reasoning_instruction + other_instruction


def transform_feedback_prompt(problem: ARCProblem, eval_results: list[EvalResultWithAns], pycode: str | None) -> str:
    # Since there’s no task information without the initial prompt code, it is required for single-turn scenarios.
    # TODO: fix False assuming that o1 is not used.
    prompt = initial_prompt()
    if pycode == "" or eval_results is None:
        prompt += "### Answer: Your answer doesn't include any code."
        prompt += "\n\n"
        prompt += "# Again, here we show the input and output grids for the problem."
        prompt += "\n\n"
        prompt += problem_prompt(problem)
        return prompt

    prompt += f"\nYour previous code:\n```\n{pycode}\n```\n\n"
    prompt += "Here are the results based on the code above.\n"

    num_correct = 0
    for i, eval_result in enumerate(eval_results):
        output = eval_result.answer
        is_correct = eval_result.get_score() == 1.0
        prompt += f"# Example {i}\n\n"
        if is_correct is True:
            prompt += "Result: Correct\n\n"
            num_correct += 1
        else:
            prompt += f"""
Result: Wrong

Your Output:
{list_format(output)}
Expected Output:
{list_format(problem.demos[i]["output"])}

"""

    if num_correct == len(eval_results):
        prompt += "# Summary\n\nYour solution is correct for all the problems!\n\n"
    else:
        prompt += f"# Summary\n\nYour solution is correct for {num_correct} problems among {len(eval_results)}!\n\n"

    # We also show transform function's result on additional inputs
    if pycode is None:
        prompt += "Your `transform` function was malformed, so please fix it accordingly.\n\n"
    else:
        prompt += "Also, here are the outputs of your `transform` function on additional inputs. Please check if your `transform` worked on additional inputs as intended, and correct your mistake in your next turns.\n\n"
        outputs = problem.run_transform_on_tests(pycode)
        for i, eval_result in enumerate(outputs):
            output = eval_result.answer
            prompt += f"# Transformed output on Additional Input {i}\n\n"
            if output is None:
                prompt += f"Your `transform` function is invalid for Additional Input {i}\n\n"
            else:
                prompt += f"""
{list_format(output)}

"""

    return prompt


def next_task_prompt(kind: Action, is_first_turn: bool) -> str:
    first_line = "Given the above result, reflect what was correct and/or wrong with your understanding and correct it accordingly inside <reflection></reflection> block, and w" if not is_first_turn else "W"

    if kind == "transform":
        return (
            f"{first_line}" + "rite your reasoning and details, and then write a new transform Python function which takes input grid as an argument inside code block surrounded by ```python and ```.\n"
            "Also, be careful to find pattern from example input and output and try to generalize it to additional inputs. "
            "DO NOT hardcode output into your `transform` function and return it for each example. Please remember that your task is to identify general transform pattern from examples.\n"
        )
    else:
        raise NotImplementedError()


sys.setrecursionlimit(20000)  # Example: Increase limit to 20000.  Choose a sensible value.


logger = logging.getLogger(__name__)


class TaskMetrics:
    """Class to track metrics for a single task."""

    def __init__(self):
        self.total_cost = 0.0
        self.cost_by_model: dict[str, float] = {}
        self.total_time = 0.0
        self.time_by_model: dict[str, float] = {}
        self.node_times: list[float] = []

    def add_cost(self, model_name: str, cost: float):
        self.total_cost += cost
        if model_name not in self.cost_by_model:
            self.cost_by_model[model_name] = 0.0
        self.cost_by_model[model_name] += cost

    def add_time(self, model_name: str, execution_time: float):
        self.node_times.append(execution_time)
        if model_name not in self.time_by_model:
            self.time_by_model[model_name] = 0.0
        self.time_by_model[model_name] += execution_time

    def reset(self):
        self.total_cost = 0.0
        self.cost_by_model = {}
        self.total_time = 0.0
        self.time_by_model = {}
        self.node_times = []


def generate_fn(
    state: NodeState | None,
    task: ARCProblem,
    prompt_template: PromptTemplate,
    model_name: str,
    model_temp: float,
    llm_log_dir: Path,
    metrics: TaskMetrics,
    diversity_mode: str,
    diversity_alpha: float,
    tokenized_code_history: list[set[str]],
    output_history: list[Any],
) -> tuple[NodeState, float]:
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
    metrics.add_cost(model_name, cost)

    result = GenerationResult(request=GenerationRequest(messages=messages), generation=generation)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # up to milliseconds

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
        score = sum([eval_result.get_score() for eval_result in eval_results]) / len(eval_results)

    if diversity_mode != "off":
        bonus = 0.0
        if diversity_mode == "code":
            try:
                code = result.parse_python_code()
                if code:
                    tokenized_code = tokenize(code)
                    if not tokenized_code_history:
                        bonus = 1.0
                    else:
                        similarities = [jaccard(tokenized_code, h) for h in tokenized_code_history]
                        max_similarity = max(similarities)
                        bonus = 1.0 - max_similarity
                    tokenized_code_history.append(tokenized_code)
            except Exception:
                pass  # No code, no bonus, no history update
        elif diversity_mode == "output":
            if eval_results and eval_results[0].answer is not None:
                # Use first demo's output grid
                output_grid = eval_results[0].answer
                bonus = novelty_bonus(output_grid, output_history, "output")
                output_history.append(output_grid)

        blended_score = (1 - diversity_alpha) * score + diversity_alpha * bonus
    else:
        blended_score = score

    # Calculate execution time for this node
    execution_time = time.time() - start_time

    # Update time by model
    metrics.add_time(model_name, execution_time)

    return NodeState(generation_result=result, eval_results=eval_results, model_name=model_name), blended_score


def get_private_score(task: ARCProblem, node_state: NodeState | None) -> float:
    if node_state is not None:
        eval_results, _score = task.evaluate_on_test(llm_answer=node_state.generation_result)
        if len(eval_results) == 0:
            private_score = 0
        else:
            private_score = sum([eval_result.get_score() for eval_result in eval_results]) / len(eval_results)
    else:
        private_score = 0
    return private_score


def process_node(node, task):
    """Helper function to process a single node and calculate scores."""
    if node.expand_idx < 0:
        return None

    node_idx = node.expand_idx
    public_score = node.score

    # calc private score
    node_state = node.state
    private_score = get_private_score(task, node_state)

    return node_idx, public_score, private_score


def get_coverage(df, is_lower_better=False):
    array_df = df.copy().values
    for i in range(len(array_df) - 1):
        if is_lower_better:
            array_df[i + 1] = np.minimum(array_df[i], array_df[i + 1])
        else:
            array_df[i + 1] = np.maximum(array_df[i], array_df[i + 1])
    return pd.DataFrame(array_df, index=df.index, columns=df.columns)


def get_perfect_coverage(df_test, df_reward, is_lower_better=False):
    array_df = df_test.copy().values * df_reward.copy().values
    for i in range(len(array_df) - 1):
        if is_lower_better:
            array_df[i + 1] = np.minimum(array_df[i], array_df[i + 1])
        else:
            array_df[i + 1] = np.maximum(array_df[i], array_df[i + 1])
    return pd.DataFrame(array_df, index=df_test.index, columns=df_test.columns)


def get_test_score_by_reward_topk(
    df_test,
    df_reward,
    top_k: int = 1,
    is_early_prioritize: bool = False,
    is_lower_better: bool = False,
):
    """
    Selects the best test scores from the top_k candidates with highest reward (val),
    returning a time-series DataFrame (≈ pass@k).
    """
    arr_test = df_test.values.copy()
    arr_reward = df_reward.values.copy()
    arr_result = arr_test.copy()

    n_rows, n_cols = arr_test.shape
    k = max(1, top_k)

    for i in range(1, n_rows):
        window_reward = arr_reward[: i + 1]
        idx = np.repeat(np.arange(i + 1)[:, None], n_cols, axis=1)

        if is_early_prioritize:
            if is_lower_better:
                sort_idx = np.lexsort((idx, window_reward), axis=0)
            else:
                sort_idx = np.lexsort((idx, -window_reward), axis=0)
        else:
            if is_lower_better:
                sort_idx = np.lexsort((-idx, window_reward), axis=0)
            else:
                sort_idx = np.lexsort((-idx, -window_reward), axis=0)

        topk_idx = sort_idx[:k].T

        row_out = []
        for test_col, idx_k in zip(arr_test[: i + 1].T, topk_idx, strict=True):
            sel_scores = test_col[idx_k]
            best = np.nanmin(sel_scores) if is_lower_better else np.nanmax(sel_scores)
            row_out.append(best)

        arr_result[i] = row_out

    return pd.DataFrame(arr_result, index=df_test.index, columns=df_test.columns)


def _process_task(
    task_id: str,
    algo: Algorithm,
    model_name: str,
    max_num_nodes: int,
    save_dir: Path,
    n_jobs: int,
    diversity_mode: str,
    diversity_alpha: float,
) -> None:
    # Create task-specific metrics tracker
    metrics = TaskMetrics()

    start_time = time.time()

    # Task
    arc_problem_path = Path(f"ARC-AGI/data/evaluation/{task_id}.json")
    if not arc_problem_path.exists():
        print(f"Task {task_id} not found")
        sys.exit(1)
    task = ARCProblem.load_file(arc_problem_path)

    # prompt
    prompt_template = BaselinePrompt(prompt_config=PromptConfig(), problem=task)

    # histories for diversity bonus
    tokenized_code_history: list[set[str]] = []
    output_history: list[Any] = []

    for subdir in ["llm_logs", "costs", "checkpoints"]:
        if not (save_dir / subdir).exists():
            (save_dir / subdir).mkdir(parents=True)

    llm_log_dir = save_dir / "llm_logs"

    generate_fns = {
        model_name: partial(
            generate_fn,
            task=task,
            model_name=model_name,
            model_temp=0.6,
            llm_log_dir=llm_log_dir,
            prompt_template=prompt_template,
            metrics=metrics,
            diversity_mode=diversity_mode,
            diversity_alpha=diversity_alpha,
            tokenized_code_history=tokenized_code_history,
            output_history=output_history,
        )
    }

    checkpoint_path = save_dir / "checkpoints" / "checkpoint_latest.pkl"

    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            search_tree = pickle.load(f)
        print(f"Loaded checkpoint from {checkpoint_path}")
        # get cost so far
        if (save_dir / "cost_summary.json").exists():
            with open(save_dir / "cost_summary.json") as f:
                cost_summary = json.load(f)
                metrics.total_cost = cost_summary["total_cost"]
                metrics.cost_by_model = cost_summary["cost_by_model"]

        # get time so far if available
        time_summary_path = save_dir / "time_summary.json"
        if time_summary_path.exists():
            with open(time_summary_path) as f:
                time_summary = json.load(f)
                metrics.total_time = time_summary.get("total_time", 0.0)
                metrics.time_by_model = time_summary.get("time_by_model", {})
                metrics.node_times = time_summary.get("node_times", [])
    else:
        search_tree = algo.init_tree()
        print("Initialized state")

    initial_num_nodes = len(algo.get_state_score_pairs(search_tree))
    for i in tqdm(range(max_num_nodes - initial_num_nodes)):
        node_start_time = time.time()
        search_tree = algo.step(search_tree, generate_fns)
        n_answers = len(algo.get_state_score_pairs(search_tree))

        # Update total time
        if i >= len(metrics.node_times):  # only add if not loaded from checkpoint
            node_execution_time = time.time() - node_start_time
            metrics.node_times.append(node_execution_time)

        if n_answers % 10 == 0 or is_power_of_two(n_answers):
            with open(save_dir / "checkpoints" / f"checkpoint_n_answers_{n_answers}.pkl", "wb") as f:
                pickle.dump(search_tree, f)
            with open(save_dir / "checkpoints" / "checkpoint_latest.pkl", "wb") as f:
                pickle.dump(search_tree, f)

            # Update total time
            metrics.total_time = time.time() - start_time

            # Log accumulated cost and time every 10 steps
            logger.info(f"Current total cost: ${metrics.total_cost:.6f}")
            logger.info(f"Current total time: {metrics.total_time:.2f} seconds")
            for model, model_cost in metrics.cost_by_model.items():
                logger.info(f"  {model} cost: ${model_cost:.6f}")
            for model, model_time in metrics.time_by_model.items():
                logger.info(f"  {model} time: {model_time:.2f} seconds")

            # Save cost summary to a JSON file
            cost_summary = {
                "total_cost": metrics.total_cost,
                "cost_by_model": metrics.cost_by_model,
            }
            with open(save_dir / "costs" / f"cost_summary_n_answers_{n_answers}.json", "w") as f:
                json.dump(cost_summary, f, indent=2)
            with open(save_dir / "costs" / "cost_summary.json", "w") as f:
                json.dump(cost_summary, f, indent=2)

            # Save time summary to a JSON file
            time_summary = {
                "total_time": metrics.total_time,
                "total_time_minutes": metrics.total_time / 60,
                "total_time_hours": metrics.total_time / 3600,
                "time_by_model": metrics.time_by_model,
                "time_by_model_minutes": {model: time / 60 for model, time in metrics.time_by_model.items()},
                "node_times": metrics.node_times,
                "avg_node_time": sum(metrics.node_times) / len(metrics.node_times) if metrics.node_times else 0,
                "avg_node_time_minutes": (sum(metrics.node_times) / len(metrics.node_times) if metrics.node_times else 0) / 60,
            }
            with open(save_dir / "costs" / f"time_summary_n_answers_{n_answers}.json", "w") as f:
                json.dump(time_summary, f, indent=2)
            with open(save_dir / "costs" / "time_summary.json", "w") as f:
                json.dump(time_summary, f, indent=2)

    # Update final total time
    metrics.total_time = time.time() - start_time

    # Log the final total cost and time
    logger.info("===== Final Cost Summary =====")
    logger.info(f"Total LLM cost: ${metrics.total_cost:.6f}")
    for model, model_cost in metrics.cost_by_model.items():
        logger.info(f"  {model}: ${model_cost:.6f}")

    logger.info("===== Final Time Summary =====")
    logger.info(f"Total execution time: {metrics.total_time:.2f} seconds ({metrics.total_time / 60:.2f} minutes, {metrics.total_time / 3600:.2f} hours)")
    logger.info(f"Average node time: {sum(metrics.node_times) / len(metrics.node_times) if metrics.node_times else 0:.2f} seconds ({(sum(metrics.node_times) / len(metrics.node_times) if metrics.node_times else 0) / 60:.2f} minutes)")
    for model, model_time in metrics.time_by_model.items():
        logger.info(f"  {model}: {model_time:.2f} seconds ({model_time / 60:.2f} minutes)")

    # Save cost summary to a JSON file
    cost_summary = {
        "total_cost": metrics.total_cost,
        "cost_by_model": metrics.cost_by_model,
    }
    with open(save_dir / "cost_summary.json", "w") as f:
        json.dump(cost_summary, f, indent=2)

    # Save time summary to a JSON file
    time_summary = {
        "total_time": metrics.total_time,
        "total_time_minutes": metrics.total_time / 60,
        "total_time_hours": metrics.total_time / 3600,
        "time_by_model": metrics.time_by_model,
        "time_by_model_minutes": {model: time / 60 for model, time in metrics.time_by_model.items()},
        "node_times": metrics.node_times,
        "avg_node_time": sum(metrics.node_times) / len(metrics.node_times) if metrics.node_times else 0,
        "avg_node_time_minutes": (sum(metrics.node_times) / len(metrics.node_times) if metrics.node_times else 0) / 60,
    }
    with open(save_dir / "time_summary.json", "w") as f:
        json.dump(time_summary, f, indent=2)

    # Eval
    valid_nodes = [node for node in search_tree.tree.get_nodes() if node.expand_idx >= 0]

    # Parallel processing of nodes
    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(process_node)(node, task) for node in tqdm(valid_nodes, desc=f"Processing {task_id}"))

    # Filter out None results (from nodes with expand_idx < 0, though already filtered)
    results = [r for r in results if r is not None]

    # Sort results by node_idx (the first element of each tuple in results)
    # This ensures that node_idx_list, public_scores, and private_scores are ordered by node_idx
    results.sort(key=lambda x: x[0])

    # Unpack sorted results
    node_idx_list = []
    public_scores = []
    private_scores = []
    for result in results:
        node_idx, public_score, private_score = result  # No need to check for None here, already filtered
        node_idx_list.append(node_idx)
        public_scores.append(public_score)
        private_scores.append(private_score)
    proc_ret = {
        "node_idx_list": node_idx_list,
        "public_scores": public_scores,
        "private_scores": private_scores,
    }
    with open(
        f"{save_dir}/checkpoints/checkpoint_n_answers_{max_num_nodes}_proc_result.json",
        "w",
    ) as f:
        json.dump(proc_ret, f)


# Set your algorithm here
algorithm: Algorithm = StandardMCTS()

#######################
#  DO NOT EDIT BELOW ##
#######################

MAX_NUM_NODES = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    parser.add_argument("--diversity_mode", type=str, default="output", choices=["off", "code", "output"], help="Diversity bonus mode")
    parser.add_argument("--diversity_alpha", type=float, default=0.5, help="Weight for diversity bonus")
    args = parser.parse_args()

    task_list_path = "experiments/arc2/arc_agi_eval_short.txt"

    with open(task_list_path) as f:
        task_list = f.readlines()
    task_list = [t.strip() for t in task_list]

    # Define function for parallel processing
    def process_task(task_id):
        _process_task(
            task_id=task_id,
            algo=algorithm,
            model_name="openrouter_deepseek_deepseek-chat-v3.1",
            max_num_nodes=MAX_NUM_NODES,
            save_dir=Path(args.out_dir) / "tasks" / task_id,
            n_jobs=4,
            diversity_mode=args.diversity_mode,
            diversity_alpha=args.diversity_alpha,
        )

    # Parallelize task processing with threading to avoid mmap pickling issues
    n_parallel = 20
    Parallel(n_jobs=n_parallel, backend="threading")(delayed(process_task)(task_id) for task_id in tqdm(task_list, desc="Processing tasks in parallel"))

    node_idx_dict = {}
    public_scores_dict = {}
    private_scores_dict = {}

    for task_id in task_list:
        save_path = f"./{args.out_dir}/tasks/{task_id}"
        proc_ret_path = Path(f"{save_path}/checkpoints/checkpoint_n_answers_{MAX_NUM_NODES}_proc_result.json")

        if not proc_ret_path.exists():
            raise FileNotFoundError(f"Result path {proc_ret_path} does not exist")

        with open(proc_ret_path) as f:
            proc_ret = json.load(f)

        node_idx_dict[task_id] = proc_ret["node_idx_list"]
        public_scores_dict[task_id] = proc_ret["public_scores"]
        private_scores_dict[task_id] = proc_ret["private_scores"]

    df_reward = pd.DataFrame(public_scores_dict)
    df_test = pd.DataFrame(private_scores_dict)

    df_reward.to_csv(os.path.join(args.out_dir, "df_reward.csv"), index=False)
    df_test.to_csv(os.path.join(args.out_dir, "df_test.csv"), index=False)

    print(f"Saved to {save_path}")
    print(f"Quick results: {df_test.max(0).sum()} / {df_test.shape[1]}")
    print(f"Quick results: {df_reward.max(0).sum()} / {df_reward.shape[1]}")

    # Calculate metrics for final_info
    test_max_scores = df_test.max(axis=0)  # Best score for each task
    reward_max_scores = df_reward.max(axis=0)  # Best reward for each task

    final_info = {
        "arc_agi": {
            "num_tasks": len(task_list),
            "max_num_nodes": MAX_NUM_NODES,
            "test_solved": int(df_test.max(0).sum()),
            "test_total": df_test.shape[1],
            "reward_solved": int(df_reward.max(0).sum()),
            "reward_total": df_reward.shape[1],
            "test_scores_by_task": test_max_scores.to_dict(),
            "reward_scores_by_task": reward_max_scores.to_dict(),
            "algorithm": str(algorithm.__class__.__name__),
            "means": {
                "test_accuracy": float(df_test.max(0).sum() / df_test.shape[1]),
                "reward_accuracy": float(df_reward.max(0).sum() / df_reward.shape[1]),
            },
        }
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f, indent=2)

    print(f"\nFinal results saved to {os.path.join(args.out_dir, 'final_info.json')}")
    os._exit(0)
