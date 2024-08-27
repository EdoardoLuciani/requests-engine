import json, aiohttp, ssl, time

from typing import Tuple
from google.oauth2 import service_account
from google.auth.transport.requests import Request

from .abstract_provider import AbstractProvider
from ..conversation import Conversation


class GcpBetaCompletionsProvider(AbstractProvider):
    def __init__(
        self,
        service_account_key: str,
        region: str = "us-central1",
        model_id: str = "meta/llama3-405b-instruct-maas",
    ):
        self.credentials = service_account.Credentials.from_service_account_info(
            json.loads(service_account_key)
        ).with_scopes(["https://www.googleapis.com/auth/cloud-platform"])
        self.token_last_refresh = 0

        self.region = region
        self.base_url = f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{self.credentials.project_id}/locations/{region}/endpoints/openapi/chat/completions"
        self.model_id = model_id
        self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

    def get_request_body(self, conversation: Conversation, temperature: float) -> str:
        return json.dumps(
            {
                "model": self.model_id,
                "messages": conversation.to_openai_format(),
                "max_tokens": 4096,
                "stream": False,
                "temperature": temperature,
            }
        )

    def _get_token(self, aiohttp_session: aiohttp.ClientSession):
        if time.time() - self.token_last_refresh > 3000:
            self.credentials.refresh(Request())
            self.token_last_refresh = time.time()

        return self.credentials.token

    def _get_completion_request(
        self, aiohttp_session: aiohttp.ClientSession, request_body: str
    ) -> aiohttp.ClientResponse:
        # https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama
        headers = {
            "Authorization": f"Bearer {self._get_token(aiohttp_session)}",
            "Content-Type": "application/json",
        }

        return aiohttp_session.post(self.base_url, data=request_body, headers=headers, ssl=self.ssl_context)

    def _get_input_output_tokens_from_completions(self, responses: list) -> Tuple[int, int]:
        return (
            sum(response["usage"]["prompt_tokens"] for response in responses),
            sum(response["usage"]["completion_tokens"] for response in responses),
        )
