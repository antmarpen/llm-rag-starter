import getpass
import json
import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from api.ai.custom_models.my_custom_llm import MyCustomLLM
from utils.logger import Logger


class RAGService:
    """Simple RAG service using LangChain and Chroma."""

    def __init__(self, vector_store: Chroma) -> None:
        self._logger = Logger.get_logger(self.__class__)

        self._logger.debug("Loading LLM")
        self._ensure_api_key()
        # self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.llm = MyCustomLLM(model="my_custom_model")

        self.vector_store = vector_store

        self._logger.debug("Loading Prompt")
        self.prompt = ChatPromptTemplate([
            ("human", """You're an assistant for question-and-answer tasks. You will be given the user's question in the following format: "Question: <user's question>", along with related context in the format: "Context: <provided context>". Limit your answer to a maximum of three sentences and keep it concise. If no context is provided, simply respond that you don't have enough information to answer.
                        Question: {question} 
                        Context: {context} 
                        Answer:""")
        ])
        # self.prompt = hub.pull("rlm/rag-prompt")

    @staticmethod
    def _ensure_api_key() -> None:
        if os.environ.get("OPENAI_API_KEY"):
            return

        current_directory = os.getcwd()
        config_path = os.path.join(current_directory + "/api/config/config.json")
        if os.path.exists(config_path):
            with open(config_path) as config_file:
                config = json.load(config_file)
                if "llmAPI" in config:
                    os.environ["OPENAI_API_KEY"] = config["llmAPI"]
                    return
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    def _retrieve(self, question: str) -> List[Document]:
        return self.vector_store.similarity_search(question)

    def _generate(self, question: str, docs: List[Document]) -> tuple[str, str]:
        docs_content = "\n\n".join(doc.page_content for doc in docs)
        messages = self.prompt.invoke({"question": question, "context": docs_content})
        final_prompt = "\n\n".join([f"{x.type}: {x.content}" for x in self.prompt.format_prompt(question=question, context=docs_content).to_messages()])
        response = self.llm.invoke(messages)
        return response.content, final_prompt

    def ask(self, question: str) -> tuple[str, str]:
        docs = self._retrieve(question)
        answer, final_prompt = self._generate(question, docs)
        return answer, final_prompt