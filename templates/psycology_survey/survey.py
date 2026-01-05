"""Helper functions to run a virtual survey."""
from __future__ import annotations
from typing import Union, Optional, Literal, Any
from pathlib import Path
import json
import logging
import argparse
import os
from collections import defaultdict, OrderedDict

import pydantic

SURVEYER_SYSTEM_INSTRUCTION = """
Your are an expert surveyer trying to understand social and societal treands. 

You will be provided with:
1.  **Persona Description:** A detailed description of the persona.
2.  **Question:** A societal question that the described person is to answer.

Your goal is to answer the question from the perspective of the persona.
"""

class TextResponse(pydantic.BaseModel):
    type: Literal['string']

class ChoicesResponse(pydantic.BaseModel):
    type: Literal['choices']
    choices: list[str]

class RepeatedResponse(pydantic.BaseModel):
    count: int
    type: Literal['repeated']
    sub_type: Union[TextResponse, ChoicesResponse]

class Question(pydantic.BaseModel):
    id: str
    question: str
    # Target audience for this question. Audience must meet all requirements.
    targets: set[str]
    response_format: Union[RepeatedResponse, ChoicesResponse, TextResponse]
    # Actions to process the responses of the answers of all targets
    # ex: Summarize common themes.  
    post_processing: list[str] = []

    def __hash__(self):
        return hash(f"{self.id}.{self.question}")

    def get_prompt_for(self, persona_summary: str):
        return f"""
        How might the person described below answer the question "{self.question}"
        {persona_summary}
        """

    def get_response_schema(self):
        def get_field_type(resp_format):
            if isinstance(resp_format, ChoicesResponse):
                return Literal[tuple(self.response_format.sub_type.choices)]
            else:
                return str
        if isinstance(self.response_format, RepeatedResponse):
            field_type = get_field_type(self.response_format.sub_type)
            field_definitions = {f"line_{i+1}":(field_type, ...) for i in range(self.response_format.count)}
            return pydantic.create_model("response", **field_definitions)
        else:
            field_type = get_field_type(self.response_format)
            field_definitions = {"answer": (field_type, ...)}
            return pydantic.create_model("response", **field_definitions)

class Survey(pydantic.BaseModel):
    title: str
    description: str
    questions: list[Question]
    analysis: list[dict]

    def __str__(self):
        questions = "\n".join([f"{q.question}" for q in self.questions])
        return (
            f"Title: {self.title}\n"
            f"Desc: {self.description}\n"
            f"Questions:\n{questions}\n"
        )

    def get_question_by_id(self, id):
        for q in self.questions:
            if q.id == id:
                return q
        return None

    def get_relative_path(self):
        return self.title.lower().replace(" ", "_")

SURVEY_RESP_TYPE = Union[dict, str, list]

class SurveyResult(pydantic.BaseModel):
    # Stores <question, response> pairs for each respondent.
    responses: dict[str, dict[Question, SURVEY_RESP_TYPE]]

    def to_json(self):
        j = {}
        for respondent_id, responses in self.responses.items():
            j[respondent_id] = {}
            for question, response in responses.items():
                j[respondent_id][question.id] = response
        return j

    def get_responses_for_question(self, q_id):
        resp = {}
        for respondent_id, responses in self.responses.items():
            for q, response in responses.items():
                if q.id == q_id:
                    resp[respondent_id] = response
        return resp

    def merge_responses(self, q1, q2):
        q1_responses = {
            respondent_id: response[q1] for respondent_id, response in self.responses.items()}
        q2_responses = {
            respondent_id: response[q2] for respondent_id, response in self.responses.items()}
        
        merged = {}
        for respondent_id, q1_resp in q1_responses.items():
            merged[respondent_id] = { k: [v]  for k, v in q1_resp.items()}
            if respondent_id not in q2_responses:
                raise Exception("Question {} response not found for respondent {}".format(q2, respondent_id))
            q2_resp = q2_responses[respondent_id]
            for k, v in q2_resp.items():
                if k not in merged[respondent_id]:
                    raise Exception("Invalid response shape, cannot merge")
                merged[respondent_id][k].append(v)

        return merged

    @classmethod
    def load(cls, base: str, survey: Survey):
        # map question id back to questions.
        questions = {q.id: q for q in survey.questions }
        p = Path(base)

        resp_by_respondent = defaultdict(dict)
        for path in p.glob('*'):
            with open(path, 'r') as f:
                resp_id = Path(path).stem
                lines = list(map(lambda l: l.strip(), f.readlines())) 
                question_id = None
                for line in lines:
                    if not line:
                        question_id = None
                    elif question_id is None:
                        # Saved as <id>.<question>
                        i, _ = line.split(".", 1)
                        question_id = i
                    elif ":" in line:
                        k, resp = line.split(":", 1)
                        if question_id not in questions:
                            raise Exception("Invalid question id found in saved survey result")
                        question = questions[question_id]
                        if question not in resp_by_respondent[resp_id]:
                            resp_by_respondent[resp_id][question] = {}
                        resp_by_respondent[resp_id][question][k] = resp.strip()
                    else:
                        print(f"Invalid line for in survey result: {line}")
    

        return cls.model_validate({"responses": resp_by_respondent})

    def save(self, base: str):
        """Saves each person's repsonses to a separate file."""
        Path(base).mkdir(parents=True, exist_ok=True)
        for respondent_id, resps in self.responses.items():
            path = os.path.join(base, respondent_id)
            with open(path, 'w') as f:
                for question, resp in resps.items():
                    f.write(f"{question.id}.{question.question}\n")
                    if isinstance(resp, dict):
                        for k, r in resp.items():
                            f.write(f"{k}: {r}\n")
                    else:
                        f.write(f"line_1: {resp}\n")
                    f.write("\n")

def load_survey(path: str) -> Survey:
    survey  = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        survey = Survey.model_validate(data)          
        logging.info(f"Successfully loaded survey: '{survey.title}'")

    except json.JSONDecodeError:
        logging.warning(f"Skipping file: '{path}' (Invalid JSON format)")
    except pydantic.ValidationError as e:
        logging.warning(f"Skipping file: '{path}' (Schema validation failed)\n{e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred with file '{path}': {e}")
        
    return survey


def main():
    parser = argparse.ArgumentParser(description="Load and display surveys")
    parser.add_argument(
        "--path",
        type=str,
        default="data/happiness_unhappiness/survey.json",
        help="survey path"
    )
    args = parser.parse_args()
    survey = load_survey(args.path)
    for a in survey.analysis:
        print(f"Analysis: {a}")
    for q in survey.questions:
        print(f"Question: {q.question}")
        print(f"Post Processing: {q.post_processing}")
        print("")

if __name__ == "__main__":
    main()