import argparse
import json
import os
import time
from ai_scientist.llm import create_client, get_response_from_llm, extract_json_between_markers, AVAILABLE_LLMS

# Define benchmark tasks
TASKS = {
    "idea_gen": {
        "system": "You are a creative AI researcher.",
        "prompt": """Generate a novel research idea for a machine learning experiment based on the following description of a template.
Template description: A NanoGPT implementation for autoregressive character-level language modeling.

Respond in valid JSON format with the following fields:
- "Name": A short name.
- "Title": A descriptive title.
- "Experiment": A brief outline.
- "Novelty": A score 1-10.
""",
        "requires_json": True
    },
    "coding": {
        "system": "You are an expert Python coder.",
        "prompt": """The following Python function has a bug where it doesn't handle the case where the input list is empty.
```python
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)
```
Please provide the corrected function. Respond only with the code block.
""",
        "requires_json": False
    },
    "writeup": {
        "system": "You are a PhD student writing a paper.",
        "prompt": """Write a LaTeX abstract (max 150 words) based on these experimental results:
Results show that adding a specialized 'expert' layer to a Transformer model improves its accuracy on the Enwik8 dataset from 1.25 BPC to 1.18 BPC, with only a 5% increase in parameters.

Respond in valid JSON format with a single field "abstract" containing the LaTeX code.
""",
        "requires_json": True
    }
}

def evaluate_model(model_name, tasks_to_run):
    print(f"\n--- Evaluating Model: {model_name} ---")
    results = {}
    try:
        client, client_model = create_client(model_name)
    except Exception as e:
        print(f"Skipping {model_name}: {e}")
        return None

    for task_name in tasks_to_run:
        task = TASKS[task_name]
        print(f"Running task: {task_name}...")
        start_time = time.time()
        try:
            content, _ = get_response_from_llm(
                task["prompt"],
                client=client,
                model=client_model,
                system_message=task["system"],
                temperature=0.1
            )
            elapsed = time.time() - start_time

            valid_json = True
            parsed_json = None
            if task["requires_json"]:
                parsed_json = extract_json_between_markers(content)
                if parsed_json is None:
                    valid_json = False

            results[task_name] = {
                "elapsed": elapsed,
                "content": content,
                "valid_json": valid_json,
                "parsed_json": parsed_json
            }
            print(f"  Done in {elapsed:.2f}s. JSON valid: {valid_json}")
        except Exception as e:
            print(f"  Error on task {task_name}: {e}")
            results[task_name] = {"error": str(e)}

    return results

def make_recommendations(all_results):
    print("\n--- Model Recommendations ---")
    recommendations = {
        "idea": {"model": None, "score": -1},
        "experiment": {"model": None, "score": -1},
        "writeup": {"model": None, "score": -1},
        "fast": {"model": None, "score": 999}, # Best is lowest time
        "fix": {"model": None, "score": -1},
        "architect": {"model": None, "score": -1}
    }

    # Simple heuristic scoring:
    # JSON validity is paramount.
    # Coding task doesn't require JSON.
    for model, results in all_results.items():
        # Idea Gen Score
        if "idea_gen" in results and not results["idea_gen"].get("error"):
            score = 10 if results["idea_gen"]["valid_json"] else 2
            if score > recommendations["idea"]["score"]:
                recommendations["idea"] = {"model": model, "score": score}

        # Experiment (Coding) Score
        if "coding" in results and not results["coding"].get("error"):
            score = 10
            if score > recommendations["experiment"]["score"]:
                recommendations["experiment"] = {"model": model, "score": score}

            # Fast model (lowest elapsed time)
            elapsed = results["coding"]["elapsed"]
            if elapsed < recommendations["fast"]["score"]:
                recommendations["fast"] = {"model": model, "score": elapsed}

        # Writeup Score
        if "writeup" in results and not results["writeup"].get("error"):
            score = 10 if results["writeup"]["valid_json"] else 2
            if score > recommendations["writeup"]["score"]:
                recommendations["writeup"] = {"model": model, "score": score}

        # Architect (Heuristic: usually the same as experiment or writeup)
        # In a real scenario, we'd have a specific foundation check task
        score = recommendations["experiment"]["score"]
        if score > recommendations["architect"]["score"]:
            recommendations["architect"] = {"model": model, "score": score}

    for phase, rec in recommendations.items():
        print(f"Recommended for {phase}: {rec['model']} (Score: {rec['score']})")

    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Evaluate models for AI Scientist phases")
    parser.add_argument("--models", nargs="+", help="List of models to evaluate", default=["gpt-4o-mini", "claude-3-5-sonnet-20241022"])
    parser.add_argument("--tasks", nargs="+", help="Tasks to run", choices=list(TASKS.keys()), default=list(TASKS.keys()))
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    args = parser.parse_args()

    all_results = {}
    for model in args.models:
        res = evaluate_model(model, args.tasks)
        if res:
            all_results[model] = res

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=4)

    recommendations = make_recommendations(all_results)
    with open("recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=4)

    print(f"\nEvaluation complete. Results saved to {args.output} and recommendations.json")

if __name__ == "__main__":
    main()
