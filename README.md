# requests-engine

[![PyPI version](https://img.shields.io/pypi/v/requests-engine)](https://pypi.org/project/requests-engine/)

requests-engine is a simple yet powerful library designed for batch LLM inference using API requests.
Key Features:
- Response Caching: Automatically caches responses to disk, eliminating redundant requests and reducing API calls.
- High Throughput: Supports multiple concurrent requests for optimal performance.
- Multiple Providers: Seamlessly integrates with AWS Bedrock, OpenAI API, Google Cloud Platform, and more.
- Unified Request Format: Consistent input format across all supported providers.
- Extensibility: Easily add your own provider.
- Cost Tracking: Optionally retrieve and track the cost of requests.

## Getting started
To quickly get started, use an OpenAI-compatible API key. You can switch to other providers later as needed.
```python
provider = requests_engine.providers.OpenAICompatibleApiProvider(
    os.environ["OPENAI_API_KEY"],
    "https://api.openai.com/v1/chat/completions",
    model_id="gpt-4o-mini",
)
engine = requests_engine.Engine(provider)

conversations = [
    requests_engine.Conversation.with_initial_message('You are an assistant. Answer shortly' 'user', e)
    for e in ['How big is the moon? ', 'How big is the sun?']
]
completions = asyncio.run(engine.schedule_completions(conversations, 0.3, 'example'))

print(completions)
print(engine.get_cost_from_completions(completions))
```

Output:
```
[
    {
        "response": {
            "id": "chatcmpl-A4WDHXbt1VmZw9u80ioYvMmsnKXUk",
            "object": "chat.completion",
            "created": 1725640503,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The Moon has a diameter of about 3,474 kilometers (2,159 miles)."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 18,
                "total_tokens": 43
            }
        },
        "request_hash": "66365aa12f5cc217411e12b6b665d5913a8c13a6182622ee8960e5500398ee85"
    },
    {
        "response": {
            "id": "chatcmpl-A4WDHxiEDMJOVSSW6xhpPmgBLIJZE",
            "object": "chat.completion",
            "created": 1725640503,
            "model": "gpt-4o-mini-2024-07-18",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The Sun has a diameter of about 1.39 million kilometers..."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 24,
                "completion_tokens": 49,
                "total_tokens": 73
            }
        },
        "request_hash": "a8df00350be4e183f2a1adf47921f8a89c0105075e9cad6fd60fac458f1c9750"
    }
]
```
```
{
    "input_tokens": 49,
    "output_tokens": 67,
    "total_tokens": 116,
    "input_cost": 7.35e-06,
    "output_cost": 4.02e-05,
    "total_cost": 4.76e-05
}

```
More examples are available in [examples/main.py](examples/main.py), and additional provider configurations can be found in the tests [tests/test_all_providers.py](tests/test_all_providers.py).

## Providers

### OpenAICompatibleApiProvider
A universal provider following the OpenAI API format for completions. Works with any provider that supports the OpenAI interface, including OpenAI, Groq, Cerebras, and more. Just supply the appropriate API key and endpoint.

Refer to [OpenAI's API documentation](https://platform.openai.com/docs/api-reference/chat/create) for details on formatting requests.

### AwsAnthropicProvider
Leverages Anthropic's models via AWS Bedrock using the [InvokeModel](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html) API. Requires AWS access credentials with InvokeModel permission.
```python
provider = requests_engine.providers.AwsAnthropicProvider(os.environ["AWS_ACCESS_KEY"], os.environ["AWS_SECRET_KEY"])
```

### GcpBetaCompletionsProvider
Integrates with Google's Vertex AI to access the managed LLAMA API. Requires a GCP service account with the [Vertex AI User](https://cloud.google.com/vertex-ai/docs/general/access-control#aiplatform.user) role. See [Google's documentatio](https://cloud.google.com/iam/docs/keys-create-delete) for information on generating service account keys.
```python
provider = requests_engine.providers.GcpBetaCompletionsProvider(your_service_account_info_object).
```

### AbstractProvider
[AbstractProvider](src/requests_engine/providers/abstract_provider.py) serves as the base class for creating custom providers. It provides a standard interface that can be extended to implement your own provider. All method stubs must be fully implemented for the custom provider to function properly. Once defined, the custom provider can be used just like any other supported provider.
```python
class MyProvider(requests_engine.providers.AbstractProvider):
    def get_request_body(self, conversation: Conversation, temperature: float) -> str:
        # Implement your own logic here

    def _get_completion_request(
        self, aiohttp_session: aiohttp.ClientSession, request_body: str
    ) -> aiohttp.ClientResponse:
        # Implement your own logic here

    def _get_input_output_tokens_from_completions(self, responses: list) -> Tuple[int, int]:
        # Implement your own logic here

provider = MyProvider()
engine = requests_engine.Engine(provider)
```
With this setup, MyProvider can be integrated into the engine and utilized for custom batch inference just like the built-in providers.


### License
Code is licensed under the MIT license.
