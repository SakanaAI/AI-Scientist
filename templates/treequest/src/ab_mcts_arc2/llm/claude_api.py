import random
import time
from typing import Any, Union

from anthropic import AnthropicBedrock, APIStatusError, BadRequestError, RateLimitError
from anthropic.types import Message, Usage

from ab_mcts_arc2.llm.llm_interface import Model


# Reference: https://aws.amazon.com/bedrock/pricing/
PRICING = {
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "input_tokens": 3.0 / 1e6,
        "output_tokens": 15.0 / 1e6,
    },
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "input_tokens": 3.0 / 1e6,
        "output_tokens": 15.0 / 1e6,
    },
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input_tokens": 3.0 / 1e6,
        "output_tokens": 15.0 / 1e6,
    },  # NOTE: US region only (at 2024-12-21)
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": {
        "input_tokens": 1.0 / 1e6,
        "output_tokens": 5.0 / 1e6,
    },  # NOTE: US region only (at 2024-12-21)
}


class ClaudeBedrockAPIModel(Model):
    def __init__(
        self,
        model: str = "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        num_trial: int = 14,  # We will wait for up to about four and a half hours.
    ) -> None:

        self.model = self.model_name = model
        self.client = AnthropicBedrock(aws_region="us-east-1")
        self.num_trial = num_trial

    def call_api(self, messages: list[dict[str, str]], temperature: float) -> Message:
        base_delay = 10  # Always wait at least 10 seconds
        for i in range(self.num_trial):
            try:
                return self.client.messages.create(
                    max_tokens=2048,
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                    # n=num_samples, # n parameter not supported for Claude
                )
            except (APIStatusError, RateLimitError, BadRequestError) as e:
                if e.status_code == 424 or e.status_code == 429 or e.status_code == 400:
                    print(
                        f"Hit an API error with status code {e.status_code}, retrying..."
                    )

                    # Exponential Backoff and Jitter
                    # See: https://aws.amazon.com/jp/blogs/architecture/exponential-backoff-and-jitter/
                    #
                    # Exponential Backoff:
                    #   2 ** i
                    #
                    # Jitter:
                    #   Multiply by a random factor between 0.5 and 1.5
                    exp_factor = 2**i
                    jitter_factor = random.uniform(0.5, 1.5)

                    # Add base_delay
                    current_backoff = exp_factor * jitter_factor
                    final_delay = base_delay + current_backoff

                    print(
                        f"Sleeping {final_delay:.2f} seconds before retry (attempt {i+1}/{self.num_trial})..."
                    )
                    time.sleep(final_delay)
                    continue
                else:
                    raise

        raise RuntimeError(
            f"Maximum number of trial {self.num_trial} reached in calling Claude API"
        )

    def calculate_price(self, usage: Usage) -> dict[Any]:
        data = usage.model_dump()
        price_data = {
            key: int(data[key]) * PRICING[self.model_name][key]
            for key in PRICING[self.model_name]
        }
        price_data["total"] = sum(price_data.values())

        return price_data

    def generate(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> tuple[str, float]:
        chat_completion = self.call_api(messages, temperature)
        data = chat_completion.usage.model_dump()
        data["price"] = self.calculate_price(chat_completion.usage)

        return chat_completion.content[0].text, data["price"]["total"]
