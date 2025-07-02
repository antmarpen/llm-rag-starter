import os
import json
import getpass
from typing import List

import bs4
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from api.utils.logger import Logger


class RAGService:
    """Simple RAG service using LangChain and Chroma."""

    def __init__(self) -> None:
        self._logger = Logger.get_logger(self.__class__)

        self._logger.debug("Loading LLM model")
        self._ensure_api_key()
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")

        self._logger.debug("Loading Embeddings")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        self._logger.debug("Loading Chroma Database")
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self.embeddings,
            persist_directory="./chroma_langchain_db",
        )

        self._logger.debug("Loading Prompt")
        self.prompt = hub.pull("rlm/rag-prompt")

        # Load blog by default for PoC
        # FIXME: This will be removed from here as this part will be done periodically and offline later
        if not self.vector_store.get()['ids']:
            self._load_blog()

    @staticmethod
    def _ensure_api_key() -> None:
        if os.environ.get("OPENAI_API_KEY"):
            return

        current_directory = os.getcwd()
        config_path = os.path.join(current_directory + "/api/config/config.json")
        if os.path.exists(config_path):
            with open(config_path) as config_file:
                config = json.load(config_file)
                if "llmModelAPI" in config:
                    os.environ["OPENAI_API_KEY"] = config["llmModelAPI"]
                    return
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    def _load_blog(self) -> None:
        """Load and index blog content."""
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs={
                "parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
            },
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        self.vector_store.add_documents(documents=all_splits)

    def _retrieve(self, question: str) -> List[Document]:
        return self.vector_store.similarity_search(question)

    def _generate(self, question: str, docs: List[Document]) -> str:
        docs_content = "\n\n".join(doc.page_content for doc in docs)
        messages = self.prompt.invoke({"question": question, "context": docs_content})
        response = self.llm.invoke(messages)
        return response.content

    def ask(self, question: str) -> str:
        docs = self._retrieve(question)
        answer = self._generate(question, docs)
        return answer