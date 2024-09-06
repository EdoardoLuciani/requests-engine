import aiohttp
from abc import ABC, abstractmethod
from typing import Tuple

from ..conversation import Conversation

class AbstractProvider(ABC):
    def get_model_id(self) -> str:
        return self.model_id

    @abstractmethod
    def get_request_body(self, conversation: Conversation, temperature: float) -> str:
        pass

    @abstractmethod
    def _get_completion_request(
        self, aiohttp_session: aiohttp.ClientSession, request_body: str
    ) -> aiohttp.ClientResponse:
        pass

    @abstractmethod
    def _get_input_output_tokens_from_completions(self, responses: list) -> Tuple[int, int]:
        pass
