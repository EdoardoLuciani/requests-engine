import aiohttp, json

from requests_engine.conversation import Conversation


class OpenAICompatibleApiProvider():
    def __init__(self, key: str, base_url: str, model: str):
        self.key = key
        self.base_url = base_url
        self.model

    def get_request_body(self, system_message: str, conversation: Conversation, temperature: float) -> str:
        messages = [{"role": "system", "content": system_message}]
        messages.append([{"role": message['role'], "content": message['content'][0]["text"]} for message in conversation.messages])

        return json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }
        )

    def get_inference_request(
        self, aiohttp_session: aiohttp.ClientSession, request_body: str
    ):
        # https://platform.openai.com/docs/api-reference/chat/create

        headers = {'Authorization': f'Bearer {self.key}'}

        return aiohttp_session.post(
            self.base_url,
            data=request_body,
            headers=headers
        )
