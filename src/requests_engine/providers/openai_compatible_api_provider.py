import aiohttp, json, ssl, requests_engine


class OpenAICompatibleApiProvider():
    def __init__(self, key: str, base_url: str, model: str):
        self.key = key
        self.base_url = base_url
        self.model = model
        self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

    def get_request_body(self, system_message: str, conversation: requests_engine.Conversation, temperature: float) -> str:
        messages = [{"role": "system", "content": system_message}]
        messages.extend([{"role": message['role'], "content": message['content'][0]["text"]} for message in conversation.messages])

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
        headers = {'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json'}

        return aiohttp_session.post(
            self.base_url,
            data=request_body,
            headers=headers,
            ssl=self.ssl_context
        )
    

    def get_batch_inference_cost(self, responses: list) -> requests_engine.ModelBatchInferenceCost:        
        cost_dict = {
            "input_tokens": sum(response["usage"]["prompt_tokens"] for response in responses),
            "output_tokens": sum(response["usage"]["completion_tokens"] for response in responses),
        }

        if self.model == "gpt-4o-mini":
            cost_dict['input_tokens_cost'] = cost_dict['input_tokens'] / 1_000_000 * 0.15
            cost_dict['output_tokens_cost'] = cost_dict['output_tokens'] / 1_000_000 * 0.6
        elif self.model == "llama-3.1-70b-versatile":
            cost_dict['input_tokens_cost'] = cost_dict['input_tokens'] / 1_000_000 * 0.59
            cost_dict['output_tokens_cost'] = cost_dict['output_tokens'] / 1_000_000 * 0.79
        elif self.model == "gemma2-9b-it":
            cost_dict['input_tokens_cost'] = cost_dict['input_tokens'] / 1_000_000 * 0.20
            cost_dict['output_tokens_cost'] = cost_dict['output_tokens'] / 1_000_000 * 0.20
        else:
            raise ValueError(f"Unsupported model: {self.model}")
        
        return cost_dict
