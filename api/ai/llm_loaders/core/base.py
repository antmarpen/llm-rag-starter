from abc import ABC, abstractmethod
from typing import Dict, Any

from langchain_core.language_models import BaseLanguageModel

from exceptions import LLMLoaderConfigException, LLMLoaderException
from utils.logger import Logger


class BaseLLMLoader(ABC):

    def __init__(self, config: Dict[str, Any]) -> None:
        self._logger = Logger.get_logger(self.__class__)

        self.__llm_instances: Dict[str, BaseLanguageModel] = {}

        self.initialize(config)

    @abstractmethod
    def check_valid_config(self, config: Dict[str, Any]) -> bool:
        """Method that validates the config for the specific model"""
        pass

    @abstractmethod
    def load_models(self, config: Dict[str, Any]) -> Dict[str, BaseLanguageModel]:
        """Method that returns the models"""
        pass

    def initialize(self, config: Dict[str, Any]) -> None:
        if self.check_valid_config(config):
            self.__llm = self.load_models(config)
        else:
            raise LLMLoaderConfigException(f"Configuration for {self.__class__.__name__} is not valid")

    def get_llm(self, model_name: str) -> BaseLanguageModel:
        if self.__llm is None or model_name not in self.__llm:
            raise LLMLoaderException(f"LLM {self.__class__.__name__} has not been initialized")

        return self.__llm

