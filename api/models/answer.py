from pydantic import BaseModel


class Answer(BaseModel):
    prompt_sent: str
    response: str