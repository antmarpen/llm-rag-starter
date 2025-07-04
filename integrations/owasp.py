from __future__ import annotations

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from integrations.core.base import BaseIntegration


class OWASPIntegration(BaseIntegration):
    """Integration that fetches and indexes a demo blog."""

    async def load_documents(self) -> list[Document]:
        # 1) Point the loader at the OWASP Top 10 URL
        url = "https://owasp.org/www-project-top-ten/"
        loader = WebBaseLoader([url])

        return loader.load()
