from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from common.db.chroma import ChromaDB


class ChromaService:

    def __init__(self):

        self.vector_store: Chroma = ChromaDB.get_instance()


    def get_query_results(self, question: str, k: int = 4, threshold: float = 0.9) -> List[Tuple[Document, float]]:
        results = self.vector_store.similarity_search_with_score(question, k=k)
        return [(doc, score) for doc, score in results if score <= threshold]