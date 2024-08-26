import aiohttp
from abc import ABC, abstractmethod

from ..conversation import Conversation
from ..model_batch_inference_cost import ModelBatchInferenceCost


class AbstractProvider(ABC):
    @abstractmethod
    def get_request_body(self, system_message: str, conversation: Conversation, temperature: float) -> str:
        pass

    @abstractmethod
    def get_inference_request(self, aiohttp_session: aiohttp.ClientSession, request_body: str) -> aiohttp.ClientResponse:
        pass

    @abstractmethod
    def get_batch_inference_cost(self, responses: list) -> ModelBatchInferenceCost:
        pass