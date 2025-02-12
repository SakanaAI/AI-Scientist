import argparse
import json
import os
import os.path as osp
from datetime import datetime

from ai_artist.generate_ideas import brainstorm_concepts, develop_concept, refine_idea_with_feedback
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
        "--num-concepts",
        type=int,
        default=10,
        help="Number of concepts to brainstorm",
    )
    parser.add_argument(
        "--num-ideas",
        type=int,
        default=2,
        help="Number of concepts to develop into full ideas (must be <= num-concepts)",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default="dall-e-3",
        help="Model to use for image generation",
    )
    parser.add_argument(
        "--brainstorm-temperature",
        type=float,
        default=0.9,
        help="Temperature for concept brainstorming",
    )
    parser.add_argument(
        "--idea-temperature",
        type=float,
        default=0.75,
        help="Temperature for initial idea development",
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
        "--num-reflections",
        type=int,
        default=3,
        help="Number of reflection iterations during idea development",
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate number of concepts and ideas
    if len(args.concepts) < 1:
        raise ValueError("Please provide at least 1 concept")
    if args.num_ideas > args.num_concepts:
        raise ValueError(f"num-ideas ({args.num_ideas}) cannot be greater than num-concepts ({args.num_concepts})")
    
    # Create client
    client, client_model = create_client(args.model)
    
    # Create main output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = osp.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save all brainstormed concepts in the main directory
    brainstormed_concepts = brainstorm_concepts(
        concepts_list=args.concepts,
        client=client,
        model=client_model,
        num_concepts=args.num_concepts,
        temperature=args.brainstorm_temperature
    )
    
    with open(osp.join(results_dir, "brainstormed_concepts.json"), "w") as f:
        json.dump(brainstormed_concepts, f, indent=4)
    
    # Step 2: Develop selected concepts into full ideas
    selected_concepts = brainstormed_concepts[:args.num_ideas]
    print(f"\nDeveloping {args.num_ideas} concepts out of {args.num_concepts} brainstormed concepts")
    
    final_ideas = []
    for idx, concept in enumerate(selected_concepts):
        # Create a directory for this specific idea
        idea_dir = osp.join(results_dir, f"idea_{idx+1}_{concept['concept'].replace(' ', '_')}")
        os.makedirs(idea_dir, exist_ok=True)
        
        print(f"\nDeveloping concept: {concept['concept']}")
        
        # Develop the initial idea
        current_idea = develop_concept(
            initial_concepts=args.concepts,
            selected_concept=concept,
            client=client,
            model=client_model,
            num_reflections=args.num_reflections,
            idea_temperature=args.idea_temperature,
            reflection_temperature=args.reflection_temperature
        )
        
        # Save initial idea
        with open(osp.join(idea_dir, "initial_idea.json"), "w") as f:
            json.dump(current_idea, f, indent=4)
        
        feedback_iteration = 0
        total_iterations = args.feedback_loops + 1
        
        # Feedback loop for each developed idea
        while feedback_iteration < total_iterations:
            iteration_prefix = f"iteration_{feedback_iteration + 1}"
            print(f"\n=== {iteration_prefix} of {total_iterations} ===")
            
            # Generate artwork from the current idea
            artwork_path = generate_artwork(
                current_idea,
                idea_dir,  # Save in idea directory
                client,
                model=args.image_model,
                suffix=f"_{iteration_prefix}",
            )
            
            if not artwork_path:
                print("Artwork generation failed. Skipping feedback loop for this idea.")
                break
            
            # Review the generated artwork
            review = perform_vlm_review(
                artwork_path,
                current_idea,
                model=client_model,
                client=client,
                review_temperature=args.review_temperature,
            )
            
            # Save the review in the idea directory
            review_filename = f"review_{iteration_prefix}.json"
            review_path = osp.join(idea_dir, review_filename)
            with open(review_path, "w") as f:
                json.dump(review, f, indent=4)
            
            # Exit loop if this was the last iteration
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
                    temperature=args.reflection_temperature,
                )
                # Save the refined idea
                with open(osp.join(idea_dir, f"refined_idea_{iteration_prefix}.json"), "w") as f:
                    json.dump(current_idea, f, indent=4)
                print("Idea refined successfully based on review feedback.")
            except Exception as e:
                print(f"Failed to refine idea based on feedback: {e}")
                break
            
            feedback_iteration += 1
        
        # Store the final version of this idea
        final_ideas.append(current_idea)
        
        # Save final version of this idea in its directory
        with open(osp.join(idea_dir, "final_idea.json"), "w") as f:
            json.dump(current_idea, f, indent=4)
    
    # Save all final ideas in the main directory
    with open(osp.join(results_dir, "final_ideas.json"), "w") as f:
        json.dump(final_ideas, f, indent=4)
    
    # Generate final artwork for each refined idea
    for idx, idea in enumerate(final_ideas):
        idea_dir = osp.join(results_dir, f"idea_{idx+1}_{idea['New_Concept'].replace(' ', '_')}")
        final_artwork = generate_artwork(
            idea,
            idea_dir,  # Save in idea directory
            client,
            model=args.image_model,
            suffix="_final"
        )

if __name__ == "__main__":
    main()