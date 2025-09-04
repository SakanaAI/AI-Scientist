from ab_mcts_arc2.llm.llm_interface import Model
from ab_mcts_arc2.llm.claude_api import PRICING as CLAUDE_PRICING
from ab_mcts_arc2.llm.claude_api import ClaudeBedrockAPIModel
from ab_mcts_arc2.llm.gemini_api import PRICING as GEMINI_PRICING
from ab_mcts_arc2.llm.gemini_api import GeminiAPIModel
from ab_mcts_arc2.llm.openai_api import PRICING as OPENAI_PRICING
from ab_mcts_arc2.llm.openai_api import OpenAIAPIModel


def build_model(model_name: str) -> Model:

    if model_name in CLAUDE_PRICING:
        model_cls = ClaudeBedrockAPIModel
    elif model_name in GEMINI_PRICING:
        model_cls = GeminiAPIModel
    elif model_name in OPENAI_PRICING:
        model_cls = OpenAIAPIModel
    else:
        raise ValueError(f"Unsupported model {model_name}")

    model = model_cls(model_name)
    return model


def call_llm(
    model_name: str, model_temp: float, messages: list[dict]
) -> tuple[str, float]:
    model = build_model(model_name)
    return model.generate(messages, model_temp)
