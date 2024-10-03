from pydantic import BaseModel


class ChatbotQueryInput(BaseModel):
    text: str


class ChatbotQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
