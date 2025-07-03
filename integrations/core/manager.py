from __future__ import annotations

import asyncio
from typing import Iterable, List

from langchain_chroma import Chroma

from integrations.core.base import BaseIntegration
from utils import files
from utils.logger import Logger


class IntegrationManager:
    """Utility to run multiple integrations concurrently."""

    def __init__(self) -> None:
        self._logger = Logger.get_logger(self.__class__)
        self._integrations: dict[str, BaseIntegration] = {}

    def register(self, integration: BaseIntegration, vector_store: Chroma) -> None:

        if integration is not None:
            if integration.__name__ not in self._integrations:
                self._integrations[integration.__name__] = integration(vector_store)
            else:
                self._logger.warning(f"Integration {integration.__name__} already registered")
        else:
            self._logger.warning(f"Invalid integration to register")

    def register_all(self, vector_store) -> None:
        integrations = files.get_all_subclasses(BaseIntegration, "integrations")

        for integration in integrations:
            self.register(integration, vector_store)


    async def start_all(self) -> None:
        for integration_name, integration in self._integrations.items():
            self._logger.info(f"Starting {integration_name}")
            asyncio.create_task(integration.run())
            self._logger.info(f"Finished {integration_name}")