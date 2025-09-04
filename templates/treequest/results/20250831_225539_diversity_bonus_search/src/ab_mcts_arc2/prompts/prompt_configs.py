from pydantic import BaseModel


class PromptConfig(BaseModel):
    is_o1: bool = False
    initial_prompt_type: str = "baseline"
    with_image: bool = False
