# requests-engine

A simple library for performing LLM batch inference using API requests. 
Features:
- Caching of responses to disk, duplicate requests will hit the cache instead of the endpoint
- Optimized for throughput, multiple concurrent requests in flight
- Multiple providers, including AWS Bedrock over Anthropic, OpenAI API, Google Cloud Platform...
- Unified format for request input across all the providers
- Easily extendable with your own provider
- Optionally retrive cost of requests 

## Getting started
It is easier to get started with an API key using the OpenAI endpoint format. More providers are available below.
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
```

Output:
```
[{'response': {'id': 'chatcmpl-A4WDHXbt1VmZw9u80ioYvMmsnKXUk', 'object': 'chat.completion', 'created': 1725640503, 'model': 'gpt-4o-mini-2024-07-18', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The Moon has a diameter of about 3,474 kilometers (2,159 miles).', 'refusal': None}, 'logprobs': None, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 25, 'completion_tokens': 18, 'total_tokens': 43}, 'system_fingerprint': 'fp_483d39d857'}, 'request_hash': '66365aa12f5cc217411e12b6b665d5913a8c13a6182622ee8960e5500398ee85'}, {'response': {'id': 'chatcmpl-A4WDHxiEDMJOVSSW6xhpPmgBLIJZE', 'object': 'chat.completion', 'created': 1725640503, 'model': 'gpt-4o-mini-2024-07-18', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The Sun has a diameter of about 1.39 million kilometers (about 864,000 miles) and is roughly 109 times wider than Earth. Its volume is large enough to fit about 1.3 million Earths inside it.', 'refusal': None}, 'logprobs': None, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 24, 'completion_tokens': 49, 'total_tokens': 73}, 'system_fingerprint': 'fp_483d39d857'}, 'request_hash': 'a8df00350be4e183f2a1adf47921f8a89c0105075e9cad6fd60fac458f1c9750'}]
```
```
{'input_tokens': 49, 'output_tokens': 67, 'total_tokens': 116, 'input_cost': 7.349999999999999e-06, 'output_cost': 4.02e-05, 'total_tokens_cost': 4.7550000000000004e-05}
```
Another example is also available [here](examples/main.py) and other providers are available here [tests](tests/test_all_providers.py)

## Providers

### OpenAICompatibleApiProvider
The most universal provider. Follows the OpenAI endpoint format [create completion format](https://platform.openai.com/docs/api-reference/chat/create) to generate a request. Only needs the appropriate API key and the reference endpoint. Can be used with any provider that adheres to the OpenAI format, i.e. Groq, OpenAI, Cerebras...

### AwsAnthropicProvider
Uses Anthropic over AWS Bedrock via [InvokeModel](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html) to generate completions. Needs an aws access and secret key with the InvokeModel permission.
```python
provider = requests_engine.providers.AwsAnthropicProvider(os.environ["AWS_ACCESS_KEY"], os.environ["AWS_SECRET_KEY"])
```

### GcpBetaCompletionsProvider
Uses Google Cloud Platform to send requests to the fully managed LLAMA API on Vertex AI. Needs a gcp service credential with the [Vertex AI User](https://cloud.google.com/vertex-ai/docs/general/access-control#aiplatform.user) permission. For info on creating one head [here](https://cloud.google.com/iam/docs/keys-create-delete)
```python
provider = requests_engine.providers.GcpBetaCompletionsProvider(your_service_account_info_object).
```


### License
Code is licensed under the MIT license.