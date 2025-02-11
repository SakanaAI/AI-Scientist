import os
from typing import Dict
import openai
from PIL import Image
import io
import requests

def generate_artwork(
    idea: Dict,
    output_dir: str,
    client,
    model="dall-e-3",
    size="1024x1024",
    quality="standard",
    style="vivid",
    suffix="",
):
    """Generate artwork from idea using DALL-E or other image generation models"""
    
    # Construct prompt from idea
    prompt = f"""{idea['Image_Prompt']}"""

    try:
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1,
        )

        # Save the image
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        image = Image.open(io.BytesIO(image_response.content))
        
        # Create filename from idea name
        filename = f"{idea['Name']}{suffix}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        image.save(filepath)
        
        return filepath
    except Exception as e:
        print(f"Error generating artwork: {e}")
        return None