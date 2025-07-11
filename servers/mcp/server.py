from typing import List, Tuple

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import Prompt

from common.services.chroma import ChromaService


class MCPServer(FastMCP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        standalone_execution = kwargs.get("standalone_execution", False)
        self.db_service: ChromaService = ChromaService(use_in_mcp=standalone_execution)

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
    mcp = MCPServer(standalone_execution=True)

    from servers.mcp.transport import TransportType

    transport = TransportType.STDIO
    transport_str = transport.value

    if transport == TransportType.HTTP:
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = 8000

    mcp.run(transport=transport_str)