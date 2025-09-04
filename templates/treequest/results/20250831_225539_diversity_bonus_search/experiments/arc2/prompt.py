from typing import List, Optional

from ab_mcts_arc2.data_types import Action
from ab_mcts_arc2.eval_result import EvalResultWithAns
from ab_mcts_arc2.llm_generation_interface import GenerationRequest, GenerationResult
from ab_mcts_arc2.prompts.arc.grid_repr import list_format
from ab_mcts_arc2.prompts.base import PromptTemplate
from ab_mcts_arc2.prompts.prompt_configs import PromptConfig
from ab_mcts_arc2.tasks.arc.task import ARCProblem


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
        eval_results: Optional[List[EvalResultWithAns]],
        generation_result: GenerationResult,
    ) -> str:
        try:
            code = generation_result.parse_python_code()
        except:
            code = ""
        match action:
            case "transform":
                return transform_feedback_prompt(
                    problem=self.problem,
                    eval_results=eval_results,
                    pycode=code,
                )
            case _:
                raise NotImplementedError(
                    f"feedback_prompt not implemented for action {action}"
                )

    def add_next_action_instruction(
        self, action: Action, next_prompt: GenerationRequest
    ) -> GenerationRequest:
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


def transform_feedback_prompt(
    problem: ARCProblem, eval_results: List[EvalResultWithAns], pycode: Optional[str]
) -> str:
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
        prompt += (
            "Your `transform` function was malformed, so please fix it accordingly.\n\n"
        )
    else:
        prompt += "Also, here are the outputs of your `transform` function on additional inputs. Please check if your `transform` worked on additional inputs as intended, and correct your mistake in your next turns.\n\n"
        outputs = problem.run_transform_on_tests(pycode)
        for i, eval_result in enumerate(outputs):
            output = eval_result.answer
            prompt += f"# Transformed output on Additional Input {i}\n\n"
            if output is None:
                prompt += (
                    f"Your `transform` function is invalid for Additional Input {i}\n\n"
                )
            else:
                prompt += f"""
{list_format(output)}

"""

    return prompt


def next_task_prompt(kind: Action, is_first_turn: bool) -> str:
    first_line = (
        "Given the above result, reflect what was correct and/or wrong with your understanding and correct it accordingly inside <reflection></reflection> block, and w"
        if not is_first_turn
        else "W"
    )

    if kind == "transform":
        return (
            f"{first_line}"
            + "rite your reasoning and details, and then write a new transform Python function which takes input grid as an argument inside code block surrounded by ```python and ```.\n"
            "Also, be careful to find pattern from example input and output and try to generalize it to additional inputs. "
            "DO NOT hardcode output into your `transform` function and return it for each example. Please remember that your task is to identify general transform pattern from examples.\n"
        )
    else:
        raise NotImplementedError()
