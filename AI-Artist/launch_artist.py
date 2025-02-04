import argparse
import json
import os
import os.path as osp
from datetime import datetime

from ai_artist.generate_ideas import generate_ideas, generate_next_artwork_idea
from ai_artist.generate_artwork import generate_artwork
from ai_artist.perform_review import perform_review
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
            prev_idea_archive=prev_ideas,
            num_reflections=3,
        )
    else:
        ideas = generate_ideas(
            concepts_list=args.concepts,
            client=client,
            model=client_model,
            max_num_generations=args.num_ideas,
            num_reflections=3,
        )
    
    # Save ideas
    with open(osp.join(results_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)
    
    # Process only the latest idea in continuous mode
    if args.continuous:
        ideas = [ideas[-1]]
        
    # Generate and review artworks
    for idea in ideas:
        # Generate artwork
        artwork_path = generate_artwork(
            idea,
            results_dir,
            client,
            model=args.image_model,
        )
        
        if artwork_path:
            # Review artwork
            review = perform_review(
                artwork_path,
                idea,
                model=client_model,
                client=client,
            )
            
            # Save review
            review_path = artwork_path.replace(".png", "_review.json")
            with open(review_path, "w") as f:
                json.dump(review, f, indent=4)

if __name__ == "__main__":
    main()