from typing import List, Tuple

from mcp.server.fastmcp import FastMCP
from langchain_chroma import Chroma

from api.services.rag import RAGService

from mcp.server.fastmcp.prompts import Prompt


class MCPServer(FastMCP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        vector_store: Chroma = kwargs.get("vector_store", None)
        self.rag_service = RAGService(vector_store)

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
        docs = self.rag_service.retrieve(question, threshold=threshold)
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