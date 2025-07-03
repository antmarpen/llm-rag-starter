from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from utils.logger import Logger


class BaseIntegration(ABC):
    """Abstract base class for data integrations."""

    def __init__(self, vector_store: Chroma, interval: int = 3600) -> None:
        self.vector_store = vector_store
        self.interval = interval
        self._logger = Logger.get_logger(self.__class__)

    @abstractmethod
    async def load_documents(self) -> List[Document]:
        """Return a list of documents to store in the vector DB."""
        raise NotImplementedError

    async def run(self) -> None:
        """Periodically load documents and store them."""
        while True:
            self._logger.info(f"Running integration task: {self.__class__.__name__}")
            try:
                docs = await self.load_documents()
                if docs:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    chunks = splitter.split_documents(docs)
                    self.vector_store.add_documents(chunks)
                    self._logger.debug(
                        "Added %d chunks to vector store", len(chunks)
                    )
            except Exception as exc:
                self._logger.error("Integration %s failed: %s", self.__class__.__name__, exc)
            await asyncio.sleep(self.interval)