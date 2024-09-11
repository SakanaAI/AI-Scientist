import argparse
import json
import os
from launch_scientist import do_idea
from perform_writeup import perform_writeup
from run_ai_reviews import perform_review
from generate_ideas import generate_next_idea

# Global variables
base_dir = os.path.join("templates", "your_experiment_name")
results_dir = os.path.join("results", "your_experiment_name")
model = "gpt-4o-2024-05-13"
client = None
client_model = None

def load_paper(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def thesis_mode(thesis_data):
    best_paper = None
    best_review_score = 0

    # Use thesis data to generate paper
    idea = thesis_data_to_idea(thesis_data)
    success = do_idea(base_dir, results_dir, idea, model, client, client_model, "latex", True)
    
    if success:
        # Review the paper
        paper_text = load_paper(f"{results_dir}/{idea['Name']}.pdf")
        review = perform_review(paper_text, model="gpt-4o-2024-05-13", client=openai.OpenAI())
        
        best_paper = idea
        best_review_score = review['Overall']
    
    return best_paper

def idea_generation_mode(seed_idea):
    # Generate new idea based on seed idea
    idea = generate_next_idea(base_dir, client, model, prev_idea_archive=[seed_idea])
    
    # Run experiment and generate paper
    success = do_idea(base_dir, results_dir, idea, model, client, client_model, "latex", True)
    
    return [idea] if success else []
def thesis_data_to_idea(thesis_data):
    return {
        "Name": thesis_data["Name"],
        "Title": thesis_data["Title"],
        "Experiment": thesis_data["Experiment"],
        "Chapters": thesis_data["Chapters"],
        "Results": thesis_data["Results"]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Scientist Pipeline")
    parser.add_argument("--mode", choices=["thesis", "idea_generation"], required=True,
                        help="Mode to run the AI Scientist in")
    parser.add_argument("--cycles", type=int, default=10, help="Number of cycles to run")
    args = parser.parse_args()

    for cycle in range(args.cycles):
        print(f"Starting cycle {cycle + 1}/{args.cycles}")
        
        if args.mode == "thesis":
            with open('my_thesis_data.json', 'r') as f:
                thesis_data = json.load(f)
            best_paper = thesis_mode(thesis_data)
            print(f"Cycle {cycle + 1} - Best paper generated: {best_paper['Title']}")
        else:
            with open('seed_idea.json', 'r') as f:
                seed_idea = json.load(f)
            papers = idea_generation_mode(seed_idea)
            print(f"Cycle {cycle + 1} - Generated {len(papers)} papers based on the seed idea")

    print(f"Completed {args.cycles} cycles in {args.mode} mode")