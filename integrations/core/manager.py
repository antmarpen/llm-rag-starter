from __future__ import annotations

import os
import threading
from typing import Type

from langchain_chroma import Chroma

from common.db.chroma import ChromaDB
from integrations.core.base import BaseIntegration
from utils import files
from utils.logger import Logger


class IntegrationManager:
    """Utility to run multiple integrations concurrently."""

    def __init__(self) -> None:
        self._logger = Logger.get_logger(self.__class__)
        self._integrations: dict[str, BaseIntegration] = {}
        self._threads: list[threading.Thread] = []

    def register(self, integration: Type[BaseIntegration]) -> None:

        if integration is not None:
            if integration.__name__ not in self._integrations:
                vector_store: Chroma = ChromaDB.get_instance()
                self._integrations[integration.__name__] = integration(vector_store)
            else:
                self._logger.warning(f"Integration {integration.__name__} already registered")
        else:
            self._logger.warning(f"Invalid integration to register")

    def register_all(self) -> None:
        self._logger.debug("Registering integrations")
        integrations = files.get_all_subclasses(BaseIntegration, "integrations")

        for integration in integrations:
            self._logger.debug(f"Registering integration {integration.__name__}")
            self.register(integration)
            self._logger.debug(f"Integration {integration.__name__} registered successfully")

        self._logger.debug("All integrations were registered successfully")


    def start_all(self) -> None:
        """Launch all registered integrations in background threads."""
        debug = os.environ.get("DEBUG", "false").lower() in {"1", "true", "yes"}

        for integration_name, integration in self._integrations.items():
            self._logger.debug(f"Launching integration thread for {integration_name}")
            thread = threading.Thread(
                target=lambda integ=integration: integ.run(),
                name=f"Integration-{integration_name}",
                daemon=not debug,
            )
            thread.start()
            self._threads.append(thread)
