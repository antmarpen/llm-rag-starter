import logging
from typing import List, Tuple

from langchain_chroma import Chroma
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import Prompt

from common.services.chroma import ChromaService


class MCPServer(FastMCP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        handlers_names = logging.getHandlerNames()

        self.db_service: ChromaService = ChromaService()

        self.setup_tools()
        self.setup_prompts()

    def setup_tools(self):
        self.add_tool(self.search)

    def setup_prompts(self):
        """
        Programmatically register prompts to guide LLM behavior.
        """
        prompt_obj = Prompt.from_function(
            self.search_internal_process,
            name="search_internal_process",
            title="Search internal Confluence documentation",
            description=(
                "For questions about internal processes or discussions, do not provide generic responses. "
                "Instead, invoke the 'search' tool with the appropriate query."
            )
        )
        self.add_prompt(prompt_obj)

    async def search(self, question: str, threshold: float = 0.9) -> List[Tuple[str, float]]:
        """
        Retrieve relevant cybersecurity context (vulnerabilities and processes) via RAG.
        """
        docs = self.db_service.get_query_results(question, threshold=threshold)
        return [(doc.page_content, score) for doc, score in docs]

    @staticmethod
    def search_internal_process(topic: str) -> str:
        return (
            "You are an assistant specialized in internal documentation. "
            f"When the user asks about '{topic}', DO NOT provide generic instructions. "
            "Instead, call the 'search' tool with the following JSON:\n\n"
            f"{{\"tool\": \"search\", \"arguments\": {{\"question\": \"{topic}\"}}}}"
        )


if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings
    # Initialize and run the server
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    from chromadb import Settings
    vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="C:\\Develop\\Proyectos\\llm-rag-starter\\chroma_langchain_db",
            client_settings=Settings(anonymized_telemetry=False)
        )
    mcp = MCPServer(vector_store=vector_store)
    #mcp.settings.host = "0.0.0.0"
    #mcp.settings.port = 8000
    #mcp.run(transport="streamable-http")
    mcp.run(transport="stdio")