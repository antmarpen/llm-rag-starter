from fastapi import FastAPI
from langchain_chroma import Chroma

from api.models.answer import Answer
from api.models.api_response import APIResponse, APIResponseWithData
from api.models.question import Question
from api.services.rag import RAGService


class Server(FastAPI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        vector_store: Chroma = kwargs.get("vector_store", None)
        self.rag_service = RAGService(vector_store)

        self.setup_routes()

    def setup_routes(self):
        self.add_api_route(
            path="/ask",
            endpoint=self.ask_question,
            methods=["POST"],
            response_model=APIResponseWithData[Answer]
        )

        self.add_api_route(
            path="/health",
            endpoint=self.health,
            methods=["GET"],
            response_model=APIResponse
        )

    async def ask_question(self, question: Question) -> APIResponseWithData[Answer]:
        """Endpoint that receives a question and returns the generated answer."""
        answer, final_prompt = self.rag_service.ask(question.query)
        response = APIResponseWithData(success=True, data=Answer(response=answer, prompt_sent=final_prompt))
        return response

    @staticmethod
    async def health() -> APIResponse:
        """Endpoint that returns OK if the server is running"""
        return APIResponse(success=True, message="OK")
