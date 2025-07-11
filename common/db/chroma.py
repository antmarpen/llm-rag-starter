import os
from pathlib import Path

from chromadb import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from utils.logger import Logger


class ChromaDB:

    _chroma_instance = None

    def __new__(cls, *args, **kwargs):
        raise TypeError("ChromaDB cannot be instanced. Instead, use the class method get_instance().")

    @classmethod
    def get_instance(cls, **config):
        """Return the singleton instance of the Chroma database"""
        if cls._chroma_instance is None:
            use_in_mcp = config.get("use_in_mcp", False)
            _logger = Logger.get_logger(cls, use_in_mcp)

            _logger.debug("Loading Chroma database")

            _logger.debug("Loading embeddings")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            _logger.debug("Embeddings loaded successfully")

            collection_name = "poc_chroma_collection"
            if "collection_name" in config:
                collection_name = config["collection_name"]

            project_root = str(Path(__file__).resolve().parents[2])
            db_folder_name = "chroma_db"
            if "db_folder_name" in config:
                db_folder_name = config["db_folder_name"]

            persist_directory = os.path.join(project_root, db_folder_name)
            cls._chroma_instance = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory,
                client_settings=Settings(anonymized_telemetry=False)
            )

            _logger.debug("Chroma database loaded successfully")

        return cls._chroma_instance