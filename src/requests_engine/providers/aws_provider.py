import json, botocore, aiohttp, os, ssl

from requests_engine.conversation import Conversation
from requests_engine.model_batch_inference_cost import ModelBatchInferenceCost

import botocore.session
from botocore.awsrequest import AWSRequest
from botocore.auth import SigV4Auth


class AwsProvider:
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region: str = "us-west-2",
    ):
        self.session = botocore.session.get_session()
        self.session.set_credentials(
            os.environ["AWS_ACCESS_KEY"], os.environ["AWS_SECRET_KEY"]
        )
        self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

        self.model_id = model_id
        self.region = region

    def get_request_body(
        self, system_message: str, messages: Conversation, temperature: float
    ) -> str:
        return json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": system_message,
                "messages": messages.messages,
                "temperature": temperature,
            }
        )

    def get_inference_request(
        self, aiohttp_session: aiohttp.ClientSession, request_body: str
    ):
        # Creating an AWSRequest object for a POST request with the service specified endpoint, JSON request body, and HTTP headers
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/invoke_model.html
        # https://docs.anthropic.com/claude/reference/messages_post
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html
        request = AWSRequest(
            method="POST",
            url=f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model_id}/invoke",
            data=request_body,
            headers={"content-type": "application/json"},
        )

        # Adding a SigV4 authentication information to the AWSRequest object, signing the request
        sigv4 = SigV4Auth(self.session.get_credentials(), "bedrock", self.region)
        sigv4.add_auth(request)

        # Prepare the request by formatting it correctly
        prepped = request.prepare()

        return aiohttp_session.post(
            prepped.url,
            data=request_body,
            headers=prepped.headers,
            ssl=self.ssl_context,
        )

    def get_batch_inference_cost(self, responses: list) -> ModelBatchInferenceCost:        
        cost_dict = {
            "input_tokens": sum(response["usage"]["input_tokens"] for response in responses),
            "output_tokens": sum(response["usage"]["output_tokens"] for response in responses),
        }

        if self.model_id == "anthropic.claude-3-haiku-20240307-v1:0":
            cost_dict['input_tokens_cost'] = cost_dict['input_tokens'] / 1_000_000 * 0.25
            cost_dict['output_tokens_cost'] = cost_dict['output_tokens'] / 1_000_000 * 1.25
        else:
            raise ValueError(f"Unsupported model_id: {self.model_id}")
        
        return cost_dict