import json
import logging
import os

from openai import APIConnectionError, OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    stop_never,
    wait_exponential,
)

from ab_mcts_arc2.llm.llm_interface import Model

# Reference: https://openai.com/api/pricing/
PRICING = {
    "gpt-4o": {
        "prompt_tokens": 2.5 / 1e6,
        "completion_tokens": 10.0 / 1e6,
    },  # NOTE: Alias for gpt-4o-2024-08-06 (at 2025-02-05)
    "gpt-4o-2024-05-13": {
        "prompt_tokens": 5.0 / 1e6,
        "completion_tokens": 15.0 / 1e6,
    },
    "gpt-4o-2024-08-06": {
        "prompt_tokens": 2.5 / 1e6,
        "completion_tokens": 10.0 / 1e6,
    },
    "gpt-4o-2024-11-20": {
        "prompt_tokens": 2.5 / 1e6,
        "completion_tokens": 10.0 / 1e6,
    },
    "gpt-4o-mini": {
        "prompt_tokens": 0.15 / 1e6,
        "completion_tokens": 0.6 / 1e6,
    },
    "gpt-4o-mini-2024-07-18": {
        "prompt_tokens": 0.15 / 1e6,
        "completion_tokens": 0.6 / 1e6,
    },
    "gpt-4.1-nano-2025-04-14": {
        "prompt_tokens": 0.10 / 1e6,
        "completion_tokens": 0.40 / 1e6,
    },
    "o1": {
        "prompt_tokens": 15.0 / 1e6,
        "completion_tokens": 60.0 / 1e6,
    },  # NOTE: Alias for o1-2024-12-17 (at 2025-02-05)
    "o1-2024-12-17": {
        "prompt_tokens": 15.0 / 1e6,
        "completion_tokens": 60.0 / 1e6,
    },
    "o1-preview": {
        "prompt_tokens": 15.0 / 1e6,
        "completion_tokens": 60.0 / 1e6,
    },  # NOTE: Alias for o1-preview-2024-09-12 (at 2025-02-05)
    "o1-preview-2024-09-12": {
        "prompt_tokens": 15.0 / 1e6,
        "completion_tokens": 60.0 / 1e6,
    },
    "o1-mini": {
        "prompt_tokens": 3.0 / 1e6,
        "completion_tokens": 12.0 / 1e6,
    },  # NOTE: Alias for o1-mini-2024-09-12 (at 2025-02-05)
    "o1-mini-2024-09-12": {
        "prompt_tokens": 3.0 / 1e6,
        "completion_tokens": 12.0 / 1e6,
    },
    "o3-mini": {
        "prompt_tokens": 1.1 / 1e6,
        "completion_tokens": 4.4 / 1e6,
    },  # NOTE: Alias for o3-mini-2025-01-31 (at 2025-02-05)
    "o3-mini-2025-01-31": {
        "prompt_tokens": 1.1 / 1e6,
        "completion_tokens": 4.4 / 1e6,
    },
    "o4-mini-2025-04-16": {
        "prompt_tokens": 1.1 / 1e6,
        "completion_tokens": 4.4 / 1e6,
    },
    "gpt-5-nano-2025-08-07": {
        "prompt_tokens": 0.05 / 1e6,
        "completion_tokens": 0.40 / 1e6,
    },
    # until 2025-02-08 16:00 (UTC)
    "deepseek-chat": {
        # "prompt_cache_hit_tokens": 0.014 / 1e6,  # 0.07 / 1e6
        # "prompt_cache_miss_tokens": 0.14 / 1e6,  # 0.27 / 1e6
        # "completion_tokens": 0.28 / 1e6,  # 1.10 / 1e6
        "prompt_cache_hit_tokens": 0.07 / 1e6,
        "prompt_cache_miss_tokens": 0.27 / 1e6,
        "completion_tokens": 1.1 / 1e6,
    },
    # For OpenRouter models, to avoid an issue with logging dir, we replace "/" with "_" as a convention
    # Openrouter models should be prepended by openrouter_ prefix.
    # deepseek/deepseek-r1
    "openrouter_deepseek_deepseek-r1": {  # Rough pricing. The price depends on the backend picked by openrouter.
        "prompt_tokens": 0.8 / 1e6,
        "completion_tokens": 2.4 / 1e6,
    },
    "openrouter_deepseek_deepseek-r1-0528": {  # Rough pricing. The price depends on the backend picked by openrouter.
        "prompt_tokens": 0.5 / 1e6,
        "completion_tokens": 2.18 / 1e6,
    },
    # deepseek/deepseek-chat
    "openrouter_deepseek_deepseek-chat": {  # Rough pricing. The price depends on the backend picked by openrouter.
        "prompt_tokens": 0.5 / 1e6,
        "completion_tokens": 0.9 / 1e6,
    },
    "openrouter_deepseek_deepseek-chat-v3.1": {
        "prompt_tokens": 0.2 / 1e6,
        "completion_tokens": 0.8 / 1e6,
    },
    # anthropic/claude-3.7-sonnet
    "openrouter_anthropic_claude-3.7-sonnet": {
        "prompt_tokens": 3.0 / 1e6,
        "completion_tokens": 15.0 / 1e6,
    },
    # anthropic/claude-3.7-sonnet:thinking
    "openrouter_anthropic_claude-3.7-sonnet:thinking": {
        "prompt_tokens": 3.0 / 1e6,
        "completion_tokens": 15.0 / 1e6,
    },
    # google/gemini-2.5-flash-preview-05-20:thinking
    "openrouter_google_gemini-2.5-flash-preview-05-20:thinking": {
        "prompt_tokens": 0.15 / 1e6,
        "completion_tokens": 3.5 / 1e6,
    },
    # google/gemini-2.5-flash
    "openrouter_google_gemini-2.5-flash": {
        "prompt_tokens": 0.30 / 1e6,
        "completion_tokens": 2.5 / 1e6,
    },
    # qwen/qwen3-235b-a22b
    "openrouter_qwen_qwen3-235b-a22b": {
        "prompt_tokens": 0.14 / 1e6,
        "completion_tokens": 0.6 / 1e6,
    },
}
OPENAI_REASONING_MODELS = set(
    [
        model_name
        for model_name in PRICING.keys()
        if model_name.startswith("o1")
        or model_name.startswith("o3")
        or model_name.startswith("o4")
    ]
)


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def will_retry_generate_failure(e: BaseException) -> bool:
    # Error codes that are likely to be resolved by retrying
    # 400 - The request was malformed
    # 429 - Rate limit reached for requests
    # 429 - You exceeded your current quota, please check your plan and billing details
    # 500 - The server had an error while processing your request
    # 502 - Bad Gateway
    # 503 - The engine is currently overloaded, please try again later
    # 504 - Gateway timeout
    # OpenAI API Reference: https://platform.openai.com/docs/guides/error-codes
    # DeepSeek API Reference: https://api-docs.deepseek.com/quick_start/error_codes
    if hasattr(e, "status_code"):
        status_code = int(e.status_code)
        if status_code in (400, 429, 500, 502, 503, 504):
            print(f"Hit an API error with status code {status_code}, retrying...")
            return True
        else:
            return False
    else:
        return False


def to_openai_client_model_name(model_name: str) -> str:
    if not model_name.startswith("openrouter_"):
        return model_name
    else:
        return model_name[len("openrouter_") :].replace("_", "/")


# The sleep durations before the retries are 1s, 2s, 4s, 8s, 8s, 8s...
@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    before_sleep=before_sleep_log(logger, logging.INFO),
    stop=(
        stop_after_attempt(int(os.environ["OPENAI_MAX_RETRY"]))
        if os.environ.get("OPENAI_MAX_RETRY") is not None
        else stop_never
    ),
    retry=retry_if_exception_type(
        json.JSONDecodeError
    )  # deepseek api sometimes raises JSONDecodeError
    | retry_if_exception_type(APIConnectionError)  # Server-side error
    | retry_if_exception(will_retry_generate_failure)
    | retry_if_exception_type(
        TypeError
    ),  # Retry if we raise TypeError for None responses
)
def try_generate(api_model: "OpenAIAPIModel", messages, temperature, request_samples=1):
    response = None
    if api_model.model in OPENAI_REASONING_MODELS:
        response = api_model.client.chat.completions.create(
            messages=messages,
            model=to_openai_client_model_name(api_model.model),
            n=request_samples,
            reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT", "low"),
        )
    else:
        response = api_model.client.chat.completions.create(
            messages=messages,
            model=to_openai_client_model_name(api_model.model),
            n=request_samples,
            temperature=temperature,
        )

    # If response.choices is None, raise an exception to trigger retry
    if response.choices is None:
        raise TypeError("Received None response from API")

    if response.usage is None:
        raise TypeError("Received None usage from API")

    return response


class OpenAIAPIModel(Model):
    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
    ) -> None:
        if model.startswith("deepseek"):
            api_key = (
                os.environ["DEEPSEEK_API_KEY"]
                if "DEEPSEEK_API_KEY" in os.environ
                else os.environ["OPENAI_API_KEY"]
            )
            self.client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)
        elif model.startswith("openrouter_"):
            api_key = (
                os.environ["OPENROUTER_API_KEY"]
                if "OPENROUTER_API_KEY" in os.environ
                else os.environ["OPENAI_API_KEY"]
            )
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=api_key
            )
        else:
            self.client = OpenAI()

        self.model = self.model_name = model

    def generate(
        self, messages: list[dict[str, str]], temperature: float = 0.6
    ) -> tuple[str, float]:
        chat_completion = try_generate(self, messages, temperature)
        usage_data = chat_completion.usage.model_dump()

        cost = (
            PRICING[self.model_name]["prompt_tokens"] * usage_data["prompt_tokens"]
            + PRICING[self.model_name]["completion_tokens"]
            * usage_data["completion_tokens"]
        )

        return chat_completion.choices[0].message.content, cost
