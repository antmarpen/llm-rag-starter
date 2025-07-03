from __future__ import annotations

import asyncio
from typing import Iterable, List, Type

from langchain_chroma import Chroma

from integrations.core.base import BaseIntegration
from utils import files
from utils.logger import Logger


class IntegrationManager:
    """Utility to run multiple integrations concurrently."""

    def __init__(self) -> None:
        self._logger = Logger.get_logger(self.__class__)
        self._integrations: dict[str, BaseIntegration] = {}

    def register(self, integration: Type[BaseIntegration], vector_store: Chroma) -> None:

        if integration is not None:
            if integration.__name__ not in self._integrations:
                self._integrations[integration.__name__] = integration(vector_store)
            else:
                self._logger.warning(f"Integration {integration.__name__} already registered")
        else:
            self._logger.warning(f"Invalid integration to register")

    def register_all(self, vector_store) -> None:
        self._logger.debug("Registering integrations")
        integrations = files.get_all_subclasses(BaseIntegration, "integrations")

        for integration in integrations:
            self._logger.debug(f"Registering integration {integration.__name__}")
            self.register(integration, vector_store)
            self._logger.debug(f"Integration {integration.__name__} registered successfully")

        self._logger.debug("All integrations were registered successfully")


    async def start_all(self) -> None:
        for integration_name, integration in self._integrations.items():
            self._logger.debug(f"Launching integration task for {integration_name}")
            asyncio.create_task(integration.run())
