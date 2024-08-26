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
    @abstractmethod
    def get_request_body(self, system_message: str, conversation: Conversation, temperature: float) -> str:
        pass

    @abstractmethod
    def get_inference_request(self, aiohttp_session: aiohttp.ClientSession, request_body: str) -> aiohttp.ClientResponse:
        pass

    @abstractmethod
    def get_responses_input_output_tokens(self, responses: list) -> Tuple[int, int]:
        pass

    def get_model_id(self) -> str:
        return self.model_id    

    def get_batch_request_cost(self, responses: list) -> BatchInferenceCost:
        (input_tokens, output_tokens) = self.get_responses_input_output_tokens(responses)
        cost = ModelPricing.calculate_cost(self.get_model_id(), input_tokens, output_tokens)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_tokens_cost": cost["input_tokens_cost"],
            "output_tokens_cost": cost["output_tokens_cost"]
        }