from typing import List

from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma

from api.services.rag import RAGService


class MCPServer(FastMCP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        vector_store: Chroma = kwargs.get("vector_store", None)
        self.rag_service = RAGService(vector_store)

        self.setup_tools()

    def setup_tools(self):
        self.add_tool(self.get_website_context)

    async def get_website_context(self, question: str) -> List[str]:
        """Endpoint that receives a question and returns the generated answer."""
        docs = self.rag_service.retrieve(question)
        return [x.page_content for x in docs]
