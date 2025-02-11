import json
import base64
from ai_artist.llm import get_response_from_llm, extract_json_between_markers, get_response_from_vlm

art_reviewer_system_prompt = """You are an experienced art critic and curator reviewing an artwork and its underlying idea.
Evaluate both the visual execution of the artwork and the conceptual strength of the idea. Focus mainly on the novel element of the artwork, the concept that has been added to the original concepts to create a novel and interesting artwork.
Provide constructive feedback on how the idea could be improved to better align with the artwork.
Be thorough in your analysis of technical execution, artistic merit, and conceptual creativity."""

art_review_template = """
Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <JSON>, provide the review in JSON format with the following fields:
- "Summary": A description of the artwork and its impact
- "Novel_Element": Comments about the novelty and interestingness of the novel element of the artwork
- "Strengths": Notable artistic achievements
- "Weaknesses": Areas for improvement
- "Suggested_Improvements": List of potential enhancements
"""

def perform_review(artwork_path: str, idea: dict, client, model, temperature: float = 0.3) -> dict:
    """
    Generate a review for the generated artwork and its underlying idea using text-only input.
    
    Parameters:
        artwork_path (str): The file path to the generated artwork.
        idea (dict): The JSON data representing the artwork idea.
        client: The LLM client.
        model (str): The model name for the LLM.
        temperature (float): Temperature configuration for the review generation.
    
    Returns:
        dict: The review as a JSON dictionary.
    """
    review_prompt = f"""
Artwork File Path: {artwork_path}
Artwork Idea: {json.dumps(idea, indent=2)}

Please review the above artwork and idea.
{art_review_template}
"""
    review_text, _ = get_response_from_llm(
        review_prompt,
        client=client,
        model=model,
        system_message=art_reviewer_system_prompt,
        temperature=temperature,
    )
    review_json = extract_json_between_markers(review_text)
    if review_json is None:
        raise ValueError("Failed to extract review JSON from LLM output.")
    return review_json

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode the given image file to a base64 string.
    """
    with open(image_path, "rb") as f:
        encoded_bytes = base64.b64encode(f.read())
    return encoded_bytes.decode('utf-8')

def perform_vlm_review(artwork_path: str, idea: dict, client, model, temperature: float = 0.3) -> dict:
    """
    Generate a review for the generated artwork using a Vision-Language Model (VLM).
    In this function the artwork image is loaded and encoded to base64 and passed along
    with the artwork idea to the VLM.
    
    Parameters:
        artwork_path (str): The file path to the generated artwork.
        idea (dict): The JSON data representing the artwork idea.
        client: The VLM client.
        model (str): The VLM model name.
        temperature (float): Temperature configuration for the review generation.
    
    Returns:
        dict: The review as a JSON dictionary.
    """
    image_base64 = encode_image_to_base64(artwork_path)
    review_prompt = f"""
Artwork (base64-encoded):
{image_base64}

Artwork Idea: {json.dumps(idea, indent=2)}

Please review the above artwork and its underlying idea, focusing on the visual details provided by the image.
{art_review_template}
"""
    review_text, _ = get_response_from_vlm(
        review_prompt,
        image_base64,
        client=client,
        model=model,
        system_message=art_reviewer_system_prompt,
        temperature=temperature,
    )
    review_json = extract_json_between_markers(review_text)
    if review_json is None:
        raise ValueError("Failed to extract review JSON from VLM output.")
    return review_json