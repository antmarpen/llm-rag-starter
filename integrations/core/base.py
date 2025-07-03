from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import hashlib

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

    def get_interval(self) -> int:
        """Return the interval between executions.

        It can be overridden in case the interval must be changed. Value by default 3600 (1 hour)"""
        return self.interval

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _doc_id(doc: Document) -> str:
        """Return a stable identifier for a document."""
        source = doc.metadata.get("source")
        if source is not None:
            return source
        # Fall back to hashing the content if no source is available
        return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()

    @staticmethod
    def _doc_hash(doc: Document) -> str:
        """Hash of the document content used to detect modifications."""
        return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()

    @staticmethod
    def _split(docs: Iterable[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(list(docs))

    # ------------------------------------------------------------------
    # Document processing
    # ------------------------------------------------------------------
    def _add_chunks(self, chunks: List[Document], base_id: str, doc_hash: str) -> None:
        ids = [f"{base_id}_{i}" for i in range(len(chunks))]
        for idx, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata or {})
            metadata["doc_id"] = base_id
            metadata["doc_hash"] = doc_hash
            metadata["chunk"] = idx
            chunk.metadata = metadata
        self.vector_store.add_documents(chunks, ids=ids)


    async def run(self) -> None:
        """Periodically load documents and store them."""
        while True:
            self._logger.debug(f"Running integration task: {self.__class__.__name__}")
            try:
                docs = await self.load_documents()
                if docs:
                    await self._process_documents(docs)
            except Exception as exc:  # pragma: no cover - log and continue
                self._logger.error(
                    f"Integration {self.__class__.__name__} failed: {exc}"
                )
            self._logger.debug(f"Integration task: {self.__class__.__name__} finished. It will be executed again in {self.get_interval()} seconds.")
            await asyncio.sleep(self.get_interval())

    async def _process_documents(self, docs: Iterable[Document]) -> None:
        new_docs: List[Document] = []
        modified_docs: List[Document] = []

        for doc in docs:
            doc_id = self._doc_id(doc)
            doc_hash = self._doc_hash(doc)

            data = self.vector_store.get(where={"doc_id": doc_id}, include=["metadatas"], limit=1)
            if not data.get("ids"):
                new_docs.append(doc)
            elif data["metadatas"][0].get("doc_hash") != doc_hash:
                modified_docs.append(doc)
            else:
                # Unchanged document, skip processing
                continue

        if new_docs:
            await self._handle_new_documents(new_docs)

        if modified_docs:
            await self._handle_modified_documents(modified_docs)

    async def _handle_new_documents(self, docs: List[Document]) -> None:
        for doc in docs:
            doc_id = self._doc_id(doc)
            doc_hash = self._doc_hash(doc)
            chunks = self._split([doc])
            self._add_chunks(chunks, doc_id, doc_hash)
            self._logger.debug(f"Added {len(chunks)} chunks for new doc {doc_id}")

    async def _handle_modified_documents(self, docs: List[Document]) -> None:
        for doc in docs:
            doc_id = self._doc_id(doc)
            doc_hash = self._doc_hash(doc)
            existing = self.vector_store.get(where={"doc_id": doc_id})
            old_count = len(existing.get("ids", []))
            if old_count:
                self.vector_store.delete(where={"doc_id": doc_id})

            chunks = self._split([doc])
            self._add_chunks(chunks, doc_id, doc_hash)
            self._logger.debug(
                f"Updated {len(chunks)} chunks for doc {doc_id} (was {old_count})"
            )
