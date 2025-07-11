from fastapi import FastAPI

from common.services.rag import RAGService
from servers.api.models.answer import Answer
from servers.api.models.api_response import APIResponse, APIResponseWithData
from servers.api.models.question import Question


class APIServer(FastAPI):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rag_service = RAGService()

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
