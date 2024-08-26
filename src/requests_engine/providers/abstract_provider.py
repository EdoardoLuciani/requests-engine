import aiohttp
from abc import ABC, abstractmethod
from typing import Tuple, TypedDict

from ..conversation import Conversation
from ..model_pricing import ModelPricing


class BatchInferenceCost(TypedDict):
    input_tokens: int
    input_tokens_cost: float
    output_tokens: int
    output_tokens_cost: float


class AbstractProvider(ABC):
    def get_model_id(self) -> str:
        return self.model_id

    @abstractmethod
    def get_request_body(
        self, system_message: str, conversation: Conversation, temperature: float
    ) -> str:
        pass

    @abstractmethod
    def _get_completion_request(
        self, aiohttp_session: aiohttp.ClientSession, request_body: str
    ) -> aiohttp.ClientResponse:
        pass

    @abstractmethod
    def _get_input_output_tokens_from_completions(
        self, responses: list
    ) -> Tuple[int, int]:
        pass

    def get_cost_from_completions(self, responses: list) -> BatchInferenceCost:
        (input_tokens, output_tokens) = self._get_input_output_tokens_from_completions(
            responses
        )
        cost = ModelPricing.get_cost_from_tokens_count(
            self.get_model_id(), input_tokens, output_tokens
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_tokens_cost": cost["input_tokens_cost"],
            "output_tokens_cost": cost["output_tokens_cost"],
        }
