import asyncio
import os
import uvicorn
from chromadb import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from api.api_server import APIServer
from api.mcp_server import MCPServer
from integrations.core.manager import IntegrationManager
from utils.logger import Logger, LoggerLevel


class Application:

    def __init__(self):
        self.__logger = Logger.get_logger(self.__class__)

        # Printing banner
        banner = [
            "=" * 50,
            "  RAG PoC Server starting",
            "  by antmarpen",
            "  Simple retrieval augmented generation demo",
            "=" * 50,
        ]
        print("\n".join(banner))

        self.__logger.info("Starting RAG PoC Server")

        # Embeddings
        self.__logger.debug("Loading embeddings")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.__logger.debug("Embeddings loaded successfully")

        # Chroma database
        self.__logger.debug("Loading Chroma database")
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self.embeddings,
            persist_directory="./chroma_langchain_db",
            client_settings=Settings(anonymized_telemetry=False)
        )
        self.__logger.debug("Chroma database loaded successfully")

        # API Server
        self.api = APIServer(title="RAG API", vector_store=self.vector_store)

        # MCP Server
        self.mcp = MCPServer(vector_store=self.vector_store)

        # Integration Manager
        self.__logger.info("Initializing integration manager")
        self.integration_manager = IntegrationManager()
        self.integration_manager.register_all(self.vector_store)
        self.__logger.info("Integration manager initialized successfully")

    async def __run_integrations(self):
        await self.integration_manager.start_all()

    def run(self):
        asyncio.run(self.__run_integrations())

        host = "0.0.0.0"
        api_port = 5000
        mcp_port = 8000

        use_api = False
        use_mcp = True
        if use_api:
            config = uvicorn.Config(self.api, host=host, port=api_port, log_config=None, reload=debug)
            server = uvicorn.Server(config)
            self.__logger.info("API Server started successfully")
            self.__logger.info(f"API Server running on http://{host}:{api_port}")
            # asyncio.run(await server.serve())

        if use_mcp:
            self.mcp.settings.host = host
            self.mcp.settings.port = mcp_port

            mcp_transport_protocol = "streamable-http"
            self.mcp.run(transport=mcp_transport_protocol)

debug = os.environ.get("DEBUG", "false").lower() in {"1", "true", "yes"}
if debug:
    Logger.set_level(LoggerLevel.DEBUG)
else:
    Logger.set_level(LoggerLevel.INFO)

_app_instance = Application()
app = _app_instance.api

if __name__ == "__main__":
    _app_instance.run()