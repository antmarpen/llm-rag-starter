import asyncio
import os
import uvicorn
from chromadb import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from api.server import Server
from integrations.core.manager import IntegrationManager
from utils.logger import Logger, LoggerLevel


class Application:

    def __init__(self):
        self.__logger = Logger.get_logger(self.__class__)

        # Embeddings
        self.__logger.debug("Loading Embeddings")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Chroma database
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self.embeddings,
            persist_directory="./chroma_langchain_db",
            client_settings=Settings(anonymized_telemetry=False)
        )

        # API Server
        self.api = Server(title="RAG API", vector_store=self.vector_store)

        # Integration Manager
        self.integration_manager = IntegrationManager()
        self.integration_manager.register_all(self.vector_store)

    async def __run_integrations(self):
        await self.integration_manager.start_all()

    async def run(self):

        await self.__run_integrations()
        debug = os.environ.get("DEBUG", "false").lower() in {"1", "true", "yes"}

        if debug:
            import pathlib
            module_name = pathlib.Path(__file__).stem
            uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=5000)
        else:
            uvicorn.run(self.api, host="0.0.0.0", port=8000)

debug = os.environ.get("DEBUG", "false").lower() in {"1", "true", "yes"}
if debug:
    Logger.set_level(LoggerLevel.DEBUG)

_app_instance = Application()
app = _app_instance.api

if __name__ == "__main__":
    asyncio.run(_app_instance.run())