from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from integrations.core.base import BaseIntegration


class NotExistingIntegration(BaseIntegration):
    """Integration that fetches and indexes a demo blog."""

    def load_documents(self) -> list[Document]:
        project_root = str(Path(__file__).resolve().parents[1])
        relative_path_list = ["docs", "integrations", "Quasirelic.txt"]
        relative_path = os.path.sep.join(relative_path_list)
        absolute_path = os.path.join(project_root, relative_path)
        loader = TextLoader(file_path=absolute_path)

        return loader.load()
