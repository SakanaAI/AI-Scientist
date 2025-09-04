import dataclasses
import logging
import os

from google import genai
from tenacity import TryAgain, before_sleep_log, retry, wait_exponential

from ab_mcts_arc2.llm.llm_interface import Model


# Reference: https://ai.google.dev/pricing#1_5pro
PRICING = {
    "gemini-1.5-pro-002": {
        "prompt_token_count": 1.25 / 1e6,
        "candidates_token_count": 5.0 / 1e6,
    },
    "gemini-2.0-flash-thinking-exp-01-21": {
        "prompt_token_count": 0.0 / 1e6,
        "candidates_token_count": 0.0 / 1e6,
    },
    "gemini-2.5-pro-preview-05-06": {
        "prompt_token_count": 1.25 / 1e6,
        "candidates_token_count": 10.0 / 1e6,
    },
    "gemini-2.5-flash-preview-05-20": {
        "prompt_token_count": 0.15 / 1e6,
        "candidates_token_count": 0.6 / 1e6,
    },
    "gemini-2.5-flash-preview-05-20:thinking": {
        "prompt_token_count": 0.15 / 1e6,
        "candidates_token_count": 3.5 / 1e6,
    },
}

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def try_generate(self: "GeminiAPIModel", contents, generation_config):
    chat_completion = self.client.models.generate_content(
        model=self.model_name,
        contents=contents,
        config=generation_config,
    )

    for candidate in chat_completion.candidates:
        # For some reason the content parts is not returned in some occasion, so we retry in that case
        if len(candidate.content.parts) == 0:
            raise TryAgain()

    return chat_completion


class GeminiAPIModel(Model):
    def __init__(
        self,
        model: str = "gemini-1.5-pro-002",
        num_trial: int = 15,
    ) -> None:
        self.model_name = model.split(":")[0]
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.num_trial = num_trial
        self.is_thinking = ":" in model

    def call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.6,
        num_samples: int = 1,
    ) -> str:
        generation_config = genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(
                include_thoughts=self.is_thinking,
            ),
            temperature=temperature,
            candidate_count=num_samples,
        )

        contents = []
        for content in messages:
            # content = dataclasses.asdict(message)
            if (
                "assistant" in content
            ):  # Gemini API uses "model" instead of "assistant" as a role
                content = genai.types.ModelContent(
                    parts=[genai.types.Part.from_text(text=content["content"])]
                )
            if "content" in content:
                content = genai.types.UserContent(
                    parts=[genai.types.Part.from_text(text=content["content"])]
                )
            contents.append(content)
        return try_generate(self, contents, generation_config)

    def generate(
        self, messages: list[dict[str, str]], temperature: float = 0.7
    ) -> tuple[str, float]:

        chat_completion = self.call_api(messages, temperature)

        prompt_token_count = chat_completion.usage_metadata.prompt_token_count
        candidates_token_count = chat_completion.usage_metadata.candidates_token_count

        prompt_price = 0
        candidates_price = 0
        if self.model_name == "gemini-2.5-pro-preview-05-06":
            if prompt_token_count <= 200_000:
                prompt_price += (
                    prompt_token_count * PRICING[self.model_name]["prompt_token_count"]
                )
            else:
                prompt_price += 200_000 * PRICING[self.model_name]["prompt_token_count"]
                prompt_price += (
                    (prompt_token_count - 200_000)
                    * PRICING[self.model_name]["prompt_token_count"]
                    * 2
                )
            if candidates_token_count <= 200_000:
                candidates_price += (
                    candidates_token_count
                    * PRICING[self.model_name]["candidates_token_count"]
                )
            else:
                candidates_price += (
                    200_000 * PRICING[self.model_name]["candidates_token_count"]
                )
                candidates_price += (
                    (candidates_token_count - 200_000)
                    * PRICING[self.model_name]["candidates_token_count"]
                    * 1.5
                )
        total_price = prompt_price + candidates_price
        return chat_completion.candidates[0].content.parts[0].text, total_price
