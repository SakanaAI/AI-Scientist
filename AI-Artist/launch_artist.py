import argparse
import json
import os
import os.path as osp
from datetime import datetime

from ai_artist.generate_ideas import generate_ideas, generate_next_artwork_idea, refine_idea_with_feedback
from ai_artist.generate_artwork import generate_artwork
from ai_artist.perform_review import perform_vlm_review
from ai_artist.llm import create_client, AVAILABLE_LLMS

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI artist")
    parser.add_argument(
        "--concepts",
        type=str,
        nargs='+',
        required=True,
        help="List of concepts to combine in the artwork",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Artist",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=5,
        help="Number of ideas to generate",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default="dall-e-3",
        help="Model to use for image generation",
    )
    parser.add_argument(
        "--idea-temperature",
        type=float,
        default=0.75,
        help="Temperature for initial idea generation",
    )
    parser.add_argument(
        "--reflection-temperature",
        type=float,
        default=0.5,
        help="Temperature for idea reflection/refinement steps",
    )
    parser.add_argument(
        "--review-temperature",
        type=float,
        default=0.3,
        help="Temperature for artwork review generation",
    )
    parser.add_argument(
        "--feedback-loops",
        type=int,
        default=0,
        help="Number of times to loop through review and idea refinement feedback",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Use continuous idea generation mode",
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate number of concepts
    if len(args.concepts) < 1:
        raise ValueError("Please provide at least 1 concept")
    
    # Create client
    client, client_model = create_client(args.model)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = osp.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate ideas
    if args.continuous:
        # Load existing ideas if any
        try:
            with open("artwork_ideas.json", "r") as f:
                prev_ideas = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            prev_ideas = []
            
        ideas = generate_next_artwork_idea(
            concepts_list=args.concepts,
            client=client,
            model=client_model,
            idea_temperature=args.idea_temperature,
            reflection_temperature=args.reflection_temperature,
            prev_idea_archive=prev_ideas,
            num_reflections=3,
        )
    else:
        ideas = generate_ideas(
            concepts_list=args.concepts,
            client=client,
            model=client_model,
            idea_temperature=args.idea_temperature,
            reflection_temperature=args.reflection_temperature,
            max_num_generations=args.num_ideas,
            num_reflections=3,
        )
    
    # Save ideas
    with open(osp.join(results_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)
    
    # For each idea, generate artwork, review, and refine the idea based on feedback.
    final_ideas = []
    for idea in ideas:
        current_idea = idea
        feedback_iteration = 0
        total_iterations = args.feedback_loops + 1
        while feedback_iteration < total_iterations:
            print(f"\n=== Feedback Iteration {feedback_iteration + 1} of {total_iterations} ===")
            # Generate artwork from the current idea
            artwork_path = generate_artwork(
                current_idea,
                results_dir,
                client,
                model=args.image_model,
                suffix=f"_iteration_{feedback_iteration + 1}",
            )
            
            if not artwork_path:
                print("Artwork generation failed. Skipping feedback loop for this idea.")
                break
            
            # Review the generated artwork along with the idea
            review = perform_vlm_review(
                artwork_path,
                current_idea,
                model=client_model,
                client=client,
                review_temperature=args.review_temperature,
            )
            
            # Save the review with an iteration suffix
            review_filename = f"{osp.splitext(os.path.basename(artwork_path))[0]}_review_{feedback_iteration + 1}.json"
            review_path = osp.join(results_dir, review_filename)
            with open(review_path, "w") as f:
                json.dump(review, f, indent=4)
            
            # If this was the last allowed feedback iteration, exit the loop
            if feedback_iteration == args.feedback_loops:
                print("Final feedback iteration reached. Keeping the current idea.")
                break
            
            # Refine the idea based on the review feedback
            try:
                current_idea = refine_idea_with_feedback(
                    current_idea, 
                    review, 
                    client, 
                    client_model,
                    temperature=args.reflection_temperature,  # Use reflection temperature for refinement
                )
                print("Idea refined successfully based on review feedback.")
            except Exception as e:
                print(f"Failed to refine idea based on feedback: {e}")
                break
            
            feedback_iteration += 1
        
        # Store the final idea (after refinement, if any)
        final_ideas.append(current_idea)
    
    # Optionally, save the final ideas
    with open(osp.join(results_dir, "final_ideas.json"), "w") as f:
        json.dump(final_ideas, f, indent=4)
    
    # In continuous mode, only process the last refined idea
    if args.continuous:
        final_ideas = [final_ideas[-1]]
    
    # (Optional) You might want to generate one final artwork for each refined idea and review it
    for idea in final_ideas:
        final_artwork = generate_artwork(
            idea,
            results_dir,
            client,
            model=args.image_model,
            suffix="_final"
        )

if __name__ == "__main__":
    main()