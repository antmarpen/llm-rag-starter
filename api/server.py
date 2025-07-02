import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from api.model.answer import Answer
from api.model.question import Question
from api.services.rag import RAGService

app = FastAPI(title="RAG API")
service = RAGService()

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question) -> Answer:
    """Endpoint that receives a question and returns the generated answer."""
    answer = service.ask(question.query)
    return Answer(response=answer)



def main() -> None:
    debug = os.environ.get("DEBUG", "false").lower() in {"1", "true", "yes"}
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=debug,
    )


if __name__ == "__main__":
    main()