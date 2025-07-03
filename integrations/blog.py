from __future__ import annotations

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from integrations.core.base import BaseIntegration


class BlogIntegration(BaseIntegration):
    """Integration that fetches and indexes a demo blog."""

    async def load_documents(self) -> list[Document]:
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs={
                "parse_only": bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            },
        )
        return loader.load()