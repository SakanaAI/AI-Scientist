import json
from dataclasses import dataclass
from typing import Optional
import os

import backoff
import google.genai as genai
from google.genai import types as genai_types

@dataclass
class LLMInput:
    prompt: str
    max_output_tokens: Optional[int] = 1000
    temperature: float = 0
    response_type: Optional[str] = None
    response_schema: Optional[dict] = None
    system_instruction: Optional[str] = None
    
    def format_prompt(self, params):
        formatted = self.prompt.format(**params)

client = genai.client.Client(api_key=os.environ["GEMINI_API_KEY"])


@backoff.on_predicate(backoff.expo, lambda x: x is None, max_tries=5)
def get_llm_response(llm_input: LLMInput):
    config = genai_types.GenerateContentConfig(
        system_instruction=llm_input.system_instruction,
        temperature=llm_input.temperature,
        max_output_tokens=llm_input.max_output_tokens,
        response_mime_type=llm_input.response_type,
        response_schema=llm_input.response_schema
    )
    resp = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=llm_input.prompt,
        config=config
    )
    if not resp.candidates:
        return None
    return resp.text

def get_llm_response_and_history(llm_input: LLMInput, msg_history=None) -> (str, list[str]):
    config = genai_types.GenerateContentConfig(
        system_instruction=llm_input.system_instruction,
        temperature=llm_input.temperature,
        max_output_tokens=llm_input.max_output_tokens,
        response_mime_type=llm_input.response_type,
        response_schema=llm_input.response_schema
    )
    chat = client.chats.create(
        model='gemini-2.5-flash', history=msg_history
    )
    resp = chat.send_message(
        llm_input.prompt,
        config=config,
    )
    return resp.text, chat.get_history()

def main():
    pass

if __name__ == "__main__":
    main()