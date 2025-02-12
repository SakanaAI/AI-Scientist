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

Create an artwork by introducing an existing concept that has never been combined with these concepts before in art history. The goal is to find that perfect unexpected element that will create a truly novel and interesting combination that no human could ever find on their own.

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
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, carefully consider if the new concept you introduced truly creates an unprecedented combination with the given concepts.
Ensure the idea is interesting and novel and the JSON is in the correct format.
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

brainstorm_system_prompt = """
You are a new AI artist tasked with brainstorming novel concept combinations that no human has ever imagined before, as you are not limited by temporal, cultural, or geographic constraints that human artists have.
"""

brainstorm_prompt = """
Given these concepts:
{concepts_list}

Brainstorm {num_concepts} different new concepts that have never been combined with these concepts before in art history. Think about unexpected elements that could create truly novel and interesting combinations.

Think step by step:
1. Analyze the given concepts and their existing relationships in art history
2. Consider different domains, time periods, cultures, and disciplines
3. List potential new concepts that would create unprecedented combinations
4. Select the most interesting and unexpected concepts

Respond in the following format:

THOUGHT:
<Discuss your brainstorming process and why these concepts create unprecedented combinations>

CONCEPTS JSON:
```json
{{
    "brainstormed_concepts": [
        {{
            "concept": "<concept name>",
            "rationale": "<why this creates an interesting combination>",
            "novelty_factor": "<what makes this combination unprecedented>"
        }},
        ...
    ]
}}
```
"""

def brainstorm_concepts(
    concepts_list: List[str],
    client,
    model: str,
    num_concepts: int = 5,
    temperature: float = 0.9
) -> List[Dict]:
    """
    Brainstorm new concepts that could be combined with the given concepts.
    
    Args:
        concepts_list: List of initial concepts
        client: LLM client
        model: Model name
        num_concepts: Number of concepts to brainstorm
        temperature: Temperature for generation
        
    Returns:
        List of dictionaries containing brainstormed concepts and their rationales
    """
    print(f"\n{'='*80}")
    print(f"Brainstorming {num_concepts} new concepts")
    print(f"{'='*80}")

    prompt = brainstorm_prompt.format(
        concepts_list=", ".join(concepts_list),
        num_concepts=num_concepts
    )
    
    text, _ = get_response_from_llm(
        prompt,
        client=client,
        model=model,
        system_message=brainstorm_system_prompt,
        temperature=temperature
    )
    
    print("\nFull Brainstorm Response:")
    print("-" * 40)
    print(text)
    print("-" * 40)
    
    json_output = extract_json_between_markers(text)
    if json_output is None or "brainstormed_concepts" not in json_output:
        raise ValueError("Failed to extract concepts from LLM output")
        
    return json_output["brainstormed_concepts"]

def develop_concept(
    initial_concepts: List[str],
    selected_concept: Dict,
    client,
    model: str,
    num_reflections: int = 3,
    idea_temperature: float = 0.75,
    reflection_temperature: float = 0.5
) -> Dict:
    """
    Develop a full artwork idea from a selected concept combination.
    
    Args:
        initial_concepts: Original concept list
        selected_concept: The chosen concept to develop
        client: LLM client
        model: Model name
        num_reflections: Number of reflection iterations
        idea_temperature: Temperature for initial generation
        reflection_temperature: Temperature for reflections
    
    Returns:
        Dictionary containing the developed artwork idea
    """
    development_prompt = f"""
Given these initial concepts:
{", ".join(initial_concepts)}

And this selected new concept to incorporate:
{json.dumps(selected_concept, indent=2)}

Develop a complete artwork idea that realizes this concept combination. Create something truly original that has never been done before in art history.

Respond in the following format:

THOUGHT:
<Discuss how you will develop this concept into a complete artwork>

NEW IDEA JSON:
```json
<JSON>
```

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea
- "Title": A title for the artwork
- "Description": Detailed description of the artwork
- "New_Concept": The concept being incorporated
- "Image_Prompt": A carefully crafted prompt for the image generation model
- "Novel_Element": Explanation of what makes this combination unprecedented
"""

    print(f"\n{'='*80}")
    print(f"Developing artwork idea from concept: {selected_concept['concept']}")
    print(f"{'='*80}")

    # Generate initial idea
    msg_history = []
    text, msg_history = get_response_from_llm(
        development_prompt,
        client=client,
        model=model,
        system_message=idea_system_prompt,
        temperature=idea_temperature
    )
    
    print("\nFull Development Response:")
    print("-" * 40)
    print(text)
    print("-" * 40)
    
    json_output = extract_json_between_markers(text)
    if json_output is None:
        raise ValueError("Failed to extract idea JSON from LLM output")

    # Perform reflections if requested
    if num_reflections > 1:
        for j in range(num_reflections - 1):
            text, msg_history = get_response_from_llm(
                idea_reflection_prompt.format(
                    current_round=j + 2,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
                temperature=reflection_temperature,
            )
            
            print(f"\nReflection {j+2} Response:")
            print("-" * 40)
            print(text)
            print("-" * 40)
            
            new_json = extract_json_between_markers(text)
            if new_json is not None:
                json_output = new_json
            
            if "I am done" in text:
                print(f"\nIdea development converged after {j + 2} iterations.")
                break

    return json_output

def refine_idea_with_feedback(idea: Dict, feedback: Dict, client, model, num_reflections: int = 1, temperature: float = 0.75) -> Dict:
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

Please refine the artwork idea to address the review feedback and improve the visual execution of the artwork.
Think step by step about what enhancements can be made to the image prompt to generate a more novel and interesting artwork, completely new from any artwork that has been painted in human history.

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
        temperature=temperature,
    )
    # Debug: Print the full LLM response.
    print("\n[Refine Idea] Full LLM Response:")
    print("-" * 40)
    print(text)
    print("-" * 40)

    refined_json = extract_json_between_markers(text)
    if refined_json is None:
        raise ValueError("Failed to extract refined idea from LLM output.")
    return refined_json