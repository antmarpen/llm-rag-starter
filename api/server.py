from fastapi import FastAPI
from pydantic import BaseModel

from api.services.rag import RAGService

app = FastAPI(title="RAG API")
service = RAGService()


class Question(BaseModel):
    query: str


class Answer(BaseModel):
    response: str


@app.post("/ask", response_model=Answer)
async def ask_question(question: Question) -> Answer:
    answer = service.ask(question.query)
    return Answer(response=answer)