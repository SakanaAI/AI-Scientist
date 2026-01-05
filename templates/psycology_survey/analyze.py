"""Helper functions that analysis survey results.

An analysis is a python function that takes SurveyResult as an input,
and either mutates the input or stores an artifact in final_infos.

Example Processor: PointScaleProcessor 

How it is used: Specify the following in survey.json:
{
    "question_id": "2",
    "point_scale": [1,2,3,4,5],
    "desc": "Change jp happiness desirability respone to 5 point scale"
}

What it does: It mutates input SurveyResylt by turning answers into point scales.
"""
from typing import Literal, Union
import json
from collections import defaultdict

from survey import SurveyResult
from llm import LLMInput, get_llm_response

import pydantic
from sklearn.manifold import MDS
import numpy as np

ANALYSIS_SYSTEM_INSTRUCTION = """
You are a Phd student that is responible for analyzing survey results.
Try to represent as much opinions as possible in surveys.
"""

class PointScaleProcessor(pydantic.BaseModel):
    question_id: str
    point_scale: list[int]
    desc: str

    def process(self, survey_result: SurveyResult, artifacts: dict):
        """Mutates literal answers to point scales"""
        for respondent_id, resp in survey_result.responses.items():
            for question, answer in resp.items():
                if question.id != self.question_id:
                    continue
                if len(list(answer.values())) != len(self.point_scale):
                    raise Exception("Point scale doesn't match the shape of survey result")
                
                original_choices = question.response_format.sub_type.choices
                for k, v in answer.items():
                    point_scale_val = self.point_scale[original_choices.index(v)]
                    survey_result.responses[respondent_id][question][k] = point_scale_val

class MergeProcessor(pydantic.BaseModel):
    """Pair responses for analysis purpose.

    Example:
    question: "What is your favorite dessert?" A: "cake"
    merge_with: "On a scale of 1 ~ 5, how much do you like this dessert?" A: 3

    merged answer: ("cake", 3)
    """
    question_id: str
    merge_with: str
    desc: str

    def process(self, survey_result: SurveyResult, artifacts: dict):
        for respondent_id, resp in survey_result.responses.items():
            # Question was not tarted for this respondent 
            if self.question_id not in set({q.id for q in resp.keys()}):
                continue
            other_resp = None
            for question, answer in resp.items():
                if question.id == self.merge_with:
                    other_resp = answer
            if not other_resp:
                raise Exception("Merge response failed.")    
            for question, answer in resp.items():
                if question.id != self.question_id:
                    continue                    
                for k, v in answer.items():
                    merged = [v, other_resp[k]]
                    survey_result.responses[respondent_id][question][k] = merged


class SimilarityMatrix(pydantic.BaseModel):
    matrix: list[list[float]]
    responses: list[str]


class SimilarityMatrixProcessor(pydantic.BaseModel):
    """Given a list of statements, use LLM to create a similarity matrix.

    The similarity matrix is stored in final_infos, and will be used 
    for downstream analysis (ex: MDS analysis).
    """
    question_id: str
    artifact_id: str
    processor: Literal["free_form_to_similarity_matrix"]
    desc: str

    def process(self, survey_result: SurveyResult, final_infos: dict):
        all_answers = []
        answer_has_multiple_items = False
        for respondent_id, resp in survey_result.responses.items():
            for question, answer in resp.items():
                if question.id != self.question_id:
                    continue
                for k, v in answer.items():
                    if isinstance(v, list) or isinstance(v, tuple):
                        # For paired answers, assume first item is free form.
                        all_answers.append(v[0])
                    else:
                        all_answers.append(v)

        answers_str = (",").join(all_answers)
        prompt = f"""
        Given a list of n statements, return a similarity matrix of n x n.
        The similarity matrix should be symmetrical, and a greater value
        at matrix[i][j] means that statements i and j are more similar.
        All values should be between 0.0 and 1.0 inclusive. Before returning
        the answer, ensure that the final matrix is symmetric.

        example input: ["I like apple", "I like pear", "I hate food"]
        example output: [[1.0,0.8,0.1], [0.8,1.0,0.1], [0.1,0.1,1.0]]

        input: {answers_str}
        """
        llm_input = LLMInput(
            prompt=prompt,
            max_output_tokens = 50000,
            temperature = 0,
            response_type = 'application/json',
            system_instruction = ANALYSIS_SYSTEM_INSTRUCTION
        )
        resp = get_llm_response(llm_input)
        data = json.loads(resp)
        final_infos[self.artifact_id] = \
            SimilarityMatrix(matrix=json.loads(resp), responses=all_answers).model_dump()

class Processors(pydantic.BaseModel):
    processor: Union[PointScaleProcessor, MergeProcessor, SimilarityMatrixProcessor]
