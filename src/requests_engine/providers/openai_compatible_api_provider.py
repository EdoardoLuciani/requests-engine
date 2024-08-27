import aiohttp, json, ssl
from typing import Tuple

from .abstract_provider import AbstractProvider
from ..conversation import Conversation


class OpenAICompatibleApiProvider(AbstractProvider):
    def __init__(self, key: str, base_url: str, model_id: str):
        self.key = key
        self.base_url = base_url
        self.model_id = model_id
        self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

    def get_request_body(self, system_message: str, conversation: Conversation, temperature: float) -> str:
        messages = [{"role": "system", "content": system_message}]
        messages.extend(
            [{"role": message["role"], "content": message["content"][0]["text"]} for message in conversation.messages]
        )

        return json.dumps(
            {
                "model": self.model_id,
                "messages": messages,
                "temperature": temperature,
            }
        )

    def _get_completion_request(
        self, aiohttp_session: aiohttp.ClientSession, request_body: str
    ) -> aiohttp.ClientResponse:
        # https://platform.openai.com/docs/api-reference/chat/create
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

        return aiohttp_session.post(self.base_url, data=request_body, headers=headers, ssl=self.ssl_context)

    def _get_input_output_tokens_from_completions(self, responses: list) -> Tuple[int, int]:
        return (
            sum(response["usage"]["prompt_tokens"] for response in responses),
            sum(response["usage"]["completion_tokens"] for response in responses),
        )
