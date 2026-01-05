import json
from enum import Enum
import argparse
from collections import defaultdict
import os

from analyze import Processors
from personas import Persona, load_personas
from survey import Survey, load_survey, SurveyResult, SURVEYER_SYSTEM_INSTRUCTION
from llm import get_llm_response_and_history, LLMInput

def run_survey(survey, personas: list[Persona]) -> SurveyResult:
    """Runs a virtual survey that answers a set of questions for a list of personas.

    Implementation is very simple: For each person, answer a list questions in series.
    If LLM calls can be parallelized without quota issues, then consider parallelizing this.
    """
    print("Running survey")
    persona_to_results = {}
    for persona in personas:
        persona_to_results[persona.name] = {}
        chat_history = [] # Chat history is per person.
        for question in survey.questions:
            should_answer = question.targets.issubset(persona.attrs)
            if not should_answer:
                continue
            print(f"Survey parcitipant<{persona.name}> is answering question: {question.question}")
            survey_llm_input = LLMInput(
                prompt=question.get_prompt_for(persona.get_summary()),
                max_output_tokens = 3000,
                temperature = 0,
                response_type = 'application/json',
                response_schema = question.get_response_schema(),
                system_instruction = SURVEYER_SYSTEM_INSTRUCTION
            )
            resp, chat_history = get_llm_response_and_history(survey_llm_input, chat_history)
            try:
                resp = json.loads(resp)
            except Exception as e:
                print(f"failed to parse into json: {resp}")

            persona_to_results[persona.name][question] = resp

    survey_result = SurveyResult.model_validate({'responses': persona_to_results})
    return survey_result

def analyze_survey_results(
    final_infos: dict, survey: Survey, survey_result: SurveyResult):
    """Runs analysis specified by the survey using the survey results as input.

    Saves analysis results into final_infos.
    """
    for analysis in survey.analysis:
        print(f"Analyzing: {analysis['desc']}")
        processor = Processors.model_validate({'processor': analysis})
        processor.processor.process(survey_result, final_infos)

def write_final_info(out_dir: str):
    final_info = {}
    with open(
            os.path.join(out_dir, f"final_info.json"), "w"
    ) as f:
        json.dump(final_info, f)
    pass    

def main():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    parser.add_argument(
        "--survey",
        type=str,
        default="data/happiness_unhappiness/survey.json",
        help="Path to a json formatted survey."
    )
    parser.add_argument(
        "--survey-result",
        type=str,
        help="Skip the survey part and load survey results from path."
    )
    parser.add_argument(
        "--personas",
        type=str,
        default="data/happiness_unhappiness/personas",
        help="Path to a folder containing yaml formatted personas."
    )
    args = parser.parse_args()
    output_dir = args.out_dir

    survey = load_survey(args.survey)
    print(f"Sucessfully loaded survey: {survey.title}")
    personas = load_personas(args.personas)
    print(f"Sucessfully loaded {len(personas)} personas")
    for persona in personas:
        print(f"Found persona: {persona.name}")

    final_infos = {}    

    if args.survey_result:
        survey_result = SurveyResult.load(args.survey_result, survey)
    else:
        survey_result = run_survey(survey, personas)
        survey_results_path = os.path.join(output_dir, "survey_results")
        survey_result.save(survey_results_path)

    final_infos["survey_result"] = survey_result.to_json()

    # Saves analysis into final_infos.
    analyze_survey_results(final_infos, survey, survey_result)
    with open(os.path.join(output_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
