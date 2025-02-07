import json
import os
import os.path as osp
import time
from typing import List, Dict, Union

import backoff
import requests

from ai_artist.llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS

idea_system_prompt = """
You are a new AI artist that is able to generate artworks that no human has ever imagined before, as you are not limited by the constraints of the physical and cultural world.
"""

concept_analysis_prompt = """
Analyze the following concepts that should appear together in an artwork:
{concepts_list}

Think step by step:
1. For each concept, identify its:
   - Historical period
   - Geographic origin
   - Cultural context
   - Traditional artistic representations
   
2. Analyze which pairs of these concepts have appeared together in art history
   
3. Identify which combination is the most unexpected or has never been done before, considering:
   - Temporal disparities
   - Geographic distances
   - Cultural contrasts
   - Traditional artistic limitations

4. Explain why these concepts haven't been combined before

Respond in the following format:

ANALYSIS:
<Your step by step analysis>

NOVEL COMBINATION:
<Identify the most unexpected or novel combination>

REASONING:
<Explain why this combination is particularly interesting or has never been done>
"""

idea_first_prompt = """
Create an artwork that meaningfully combines these concepts in an unexpected and totally novel way:
{concepts_list}

Think step by step:
1. Consider the historical, geographical, and cultural context of each concept
2. Identify which concepts have rarely or never been combined in art history with the given concepts
3. Plan how to unite these concepts in a single artwork

Respond in the following format:

THOUGHT:
<Discuss how you plan to meaningfully unite these concepts in a single artwork. Explain which combinations are particularly unexpected and interesting and why.>

NEW IDEA JSON:
```json
<JSON>
```

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea
- "Title": A title for the artwork
- "Description": Detailed description of the artwork
- "Style": Artistic style(s) to be used
- "Medium": Proposed medium/technique
- "Colors": Main color palette
- "Composition": Description of composition
- "Symbolism": Key symbolic elements and how they unite the concepts
- "Concept_Bridge": Explanation of how the artwork bridges the concepts
- "Image_Prompt": A carefully crafted prompt for the image generation model that will create this artwork
- "Originality": A rating from 1 to 10
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, carefully consider the artistic merit and novelty of the idea you just created.
Consider if your idea is novel and interesting. Also reflect if the prompt is clear and will generate a good image.
Ensure the idea is clear and impactful, and the JSON is in the correct format.
In the next attempt, try to refine and improve your artistic vision.
Stick to the spirit of the original idea unless there are fundamental issues.

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
    
    for _ in range(max_num_generations):
        print(f"\nGenerating idea {_ + 1}/{max_num_generations}")
        try:
            # Generate initial idea
            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            text, msg_history = get_response_from_llm(
                idea_first_prompt.format(
                    concepts_list=", ".join(concepts_list)
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)

            # Iteratively improve idea
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
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
                    
                    json_output = extract_json_between_markers(text)
                    assert json_output is not None, "Failed to extract JSON from LLM output"
                    print(json_output)

                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
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

    print(f"Generating artwork idea {original_archive_size + 1}")

    # Convert previous ideas to string format
    idea_strings = []
    for idea in idea_archive:
        idea_strings.append(json.dumps(idea, indent=2))
    prev_ideas_string = "\n\n".join(idea_strings)

    # Enhanced prompt that builds upon previous ideas
    enhanced_first_prompt = f"""
Here are the artwork ideas that have been generated so far:

{prev_ideas_string}

Create a new artwork that meaningfully combines these concepts in an unexpected way:
{", ".join(concepts_list)}

Think step by step:
1. Consider how this combination differs from previous attempts
2. Identify which aspects haven't been fully explored yet
3. Think about new ways to bridge temporal, cultural, or geographic gaps
4. Plan how to unite these elements in a fresh and meaningful way

Respond in the following format:

THOUGHT:
<Discuss how this idea builds upon or differs from previous attempts. Explain which new combinations or approaches you're exploring.>

NEW IDEA JSON:
```json
<JSON>
```

The JSON should include:
- "Name": A shortened descriptor of the idea
- "Title": A title for the artwork
- "Description": Detailed description of the artwork
- "Style": Artistic style(s) to be used
- "Medium": Proposed medium/technique
- "Colors": Main color palette
- "Composition": Description of composition
- "Symbolism": Key symbolic elements and how they unite the concepts
- "Concept_Bridge": Explanation of how the artwork bridges the temporal/cultural/geographic gaps
- "Historical_Context": Brief analysis of why these concepts haven't been combined before
- "Aesthetic": A rating from 1 to 10
- "Technical_Feasibility": A rating from 1 to 10
- "Originality": A rating from 1 to 10
- "Relation_to_Previous": How this piece differs from previous attempts
"""

    for _ in range(max_attempts):
        try:
            # Generate initial idea
            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            text, msg_history = get_response_from_llm(
                enhanced_first_prompt,
                client=client,
                model=model,
                msg_history=msg_history,
            )
            
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)

            # Iteratively improve idea
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2,
                            num_reflections=num_reflections,
                        ),
                        client=client,
                        model=model,
                        msg_history=msg_history,
                    )
                    
                    json_output = extract_json_between_markers(text)
                    assert json_output is not None, "Failed to extract JSON from LLM output"
                    print(json_output)

                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
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