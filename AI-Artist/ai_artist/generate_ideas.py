import json
import os
import os.path as osp
import time
from typing import List, Dict, Union

import backoff
import requests

from ai_artist.llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS

idea_system_prompt = """
You are a new AI artist that is able to generate artworks that no human has ever imagined before, as you are not limited by temporal, cultural, or geographic constraints tha humans artist have.
"""

idea_first_prompt = """
Given these concepts:
{concepts_list}

Create an artwork by introducing a completely new concept that has never been combined with these concepts before in art history. The goal is to find that perfect unexpected element that will create a truly novel and interesting combination.

Think step by step:
1. Analyze the given concepts and their existing relationships in art history
2. Brainstorm potential new concepts that have never been combined with these for whatever reason: temporal difference, cultural difference, geographic difference, etc.
3. Select the most interesting and unexpected new concept
4. Plan how to unite all concepts in a way where the new element creates something truly original

Respond in the following format:

THOUGHT:
<Discuss which new concept you chose and why it creates an unprecedented combination with the given concepts. Explain why this specific combination has never been explored before and what makes it compelling.>

NEW IDEA JSON:
```json
<JSON>
```

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea
- "Title": A title for the artwork
- "Description": Detailed description of the artwork
- "New Concept": The new concept that you chose to introduce
- "Image_Prompt": A carefully crafted prompt for the image generation model that will create this artwork
- "Novel_Element": Explanation of what makes this combination unprecedented
- "Originality": A rating from 1 to 10
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, carefully consider if the new concept you introduced truly creates an unprecedented combination with the given concepts.
Reflect on whether this combination is meaningful and if there might be an even more unexpected concept to introduce.
Ensure the idea is impactful and the JSON is in the correct format.
In the next attempt, you can either refine the current combination or propose a different new concept if you think it would work better.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


def generate_ideas(
    concepts_list,
    client,
    model,
    max_num_generations=5,
    num_reflections=3,
):
    """Generate artwork ideas based on concept combinations"""
    idea_archive = []
    
    for i in range(max_num_generations):
        print(f"\n{'='*80}")
        print(f"Generating idea {i + 1}/{max_num_generations}")
        print(f"{'='*80}")
        try:
            # Generate initial idea
            msg_history = []
            print(f"\n--- Iteration 1/{num_reflections} ---")
            text, msg_history = get_response_from_llm(
                idea_first_prompt.format(
                    concepts_list=", ".join(concepts_list)
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            
            print("\nFull LLM Response:")
            print(f"{'-'*40}")
            print(text)
            print(f"{'-'*40}")
            
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print("\nExtracted JSON:")
            print(json.dumps(json_output, indent=2))

            # Iteratively improve idea
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"\n--- Iteration {j + 2}/{num_reflections} ---")
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2,
                            num_reflections=num_reflections,
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    
                    print("\nFull LLM Response:")
                    print(f"{'-'*40}")
                    print(text)
                    print(f"{'-'*40}")
                    
                    json_output = extract_json_between_markers(text)
                    assert json_output is not None, "Failed to extract JSON from LLM output"
                    print("\nExtracted JSON:")
                    print(json.dumps(json_output, indent=2))

                    if "I am done" in text:
                        print(f"\nIdea generation converged after {j + 2} iterations.")
                        break

            idea_archive.append(json_output)
            
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    return idea_archive

def generate_next_artwork_idea(
        concepts_list,
        client,
        model,
        prev_idea_archive=[],
        num_reflections=3,
        max_attempts=5,
):
    """
    Generate the next artwork idea building upon previous ideas.
    """
    idea_archive = prev_idea_archive.copy()
    original_archive_size = len(idea_archive)

    print(f"\n{'='*80}")
    print(f"Generating artwork idea {original_archive_size + 1}")
    print(f"{'='*80}")

    # Convert previous ideas to string format
    idea_strings = []
    for idea in idea_archive:
        idea_strings.append(json.dumps(idea, indent=2))
    prev_ideas_string = "\n\n".join(idea_strings)

    # Enhanced prompt that builds upon previous ideas
    enhanced_first_prompt = f"""
Here are the artwork ideas that have been generated so far:

{prev_ideas_string}

Given these concepts:
{", ".join(concepts_list)}

Create an artwork by introducing a completely new concept that has never been combined with these concepts before in art history, while also considering the previous attempts. The goal is to find that perfect unexpected element that will create a truly novel and interesting combination.

Think step by step:
1. Review what combinations have already been attempted in previous ideas
2. Analyze what aspects or domains haven't been explored yet
3. Brainstorm potential new concepts that would create unprecedented combinations
4. Select the most interesting and unexpected new concept
5. Plan how to unite all concepts in a way where the new element creates something truly original

Respond in the following format:

THOUGHT:
<Discuss which new concept you chose and why it creates an unprecedented combination with the given concepts. Explain how this differs from previous attempts and what makes this combination particularly compelling.>

NEW IDEA JSON:
```json
<JSON>
```

The JSON should include:
- "Name": A shortened descriptor of the idea
- "Title": A title for the artwork
- "Description": Detailed description of the artwork
- "New Concept": The new concept that you chose to introduce
- "Previous_Combinations": Brief analysis of what has been tried before
- "Novel_Element": Explanation of what makes this combination unprecedented
- "Image_Prompt": A carefully crafted prompt for the image generation model that will create this artwork
- "Originality": A rating from 1 to 10
"""

    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt + 1}/{max_attempts}")
        try:
            # Generate initial idea
            msg_history = []
            print(f"\n--- Iteration 1/{num_reflections} ---")
            text, msg_history = get_response_from_llm(
                enhanced_first_prompt,
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            
            print("\nFull LLM Response:")
            print(f"{'-'*40}")
            print(text)
            print(f"{'-'*40}")
            
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print("\nExtracted JSON:")
            print(json.dumps(json_output, indent=2))

            # Iteratively improve idea
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"\n--- Iteration {j + 2}/{num_reflections} ---")
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2,
                            num_reflections=num_reflections,
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    
                    print("\nFull LLM Response:")
                    print(f"{'-'*40}")
                    print(text)
                    print(f"{'-'*40}")
                    
                    json_output = extract_json_between_markers(text)
                    assert json_output is not None, "Failed to extract JSON from LLM output"
                    print("\nExtracted JSON:")
                    print(json.dumps(json_output, indent=2))

                    if "I am done" in text:
                        print(f"\nIdea generation converged after {j + 2} iterations.")
                        break

            idea_archive.append(json_output)
            break
            
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    # Save updated archive
    with open("artwork_ideas.json", "w") as f:
        json.dump(idea_archive, f, indent=4)

    return idea_archive

def refine_idea_with_feedback(idea: Dict, feedback: Dict, client, model, num_reflections: int = 1) -> Dict:
    """
    Refine the artwork idea using review feedback.
    This function prompts the LLM (using the same idea system prompt)
    to revise the idea given the feedback.
    """
    feedback_prompt = f"""
Based on the following existing artwork idea:
+{json.dumps(idea, indent=2)}

And the following review feedback:
+{json.dumps(feedback, indent=2)}

Please refine the artwork idea to address the review feedback and improve the new concept added.
Think step by step about what enhancements can be made to better align the artwork with the critic's comments, focusing on how to modify the prompt to generate a better artwork.

Respond in the following format:

THOUGHT:
<Discuss the changes made based on the feedback>

NEW IDEA JSON:
```json
<JSON>
```
"""
    text, _ = get_response_from_llm(
        feedback_prompt,
        client=client,
        model=model,
        system_message=idea_system_prompt,
        temperature=0.75,
    )
    refined_json = extract_json_between_markers(text)
    if refined_json is None:
        raise ValueError("Failed to extract refined idea from LLM output.")
    return refined_json