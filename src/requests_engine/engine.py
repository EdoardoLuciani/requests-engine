import aiohttp
import pickle
import hashlib
from pathlib import Path
from requests_engine.conversation import Conversation
import os
import asyncio
import functools


class Engine:
    def __init__(self, provider):
        self.serialization_path = "cache"
        self.provider = provider

    async def schedule_completions(
        self, system_prompt: str, messages: list, temperature: float, task_name: str
    ) -> list:
        semaphore = asyncio.Semaphore(32)
        async with aiohttp.ClientSession() as session:

            async def task(message):
                async with semaphore:
                    return await self._get_or_generate_inference(
                        session, system_prompt, message, temperature, task_name
                    )

            return await asyncio.gather(*(task(message) for message in messages))

    async def _get_or_generate_inference(
        self,
        session: aiohttp.ClientSession,
        system_message: str,
        messages: Conversation,
        temperature: float,
        task_name: str,
    ):

        request_body = self.provider.get_request_body(
            system_message, messages, temperature
        )
        request_body_digest = _get_request_body_digest(request_body)

        file_path = f"{self.serialization_path}/{task_name}"
        Path(file_path).mkdir(parents=True, exist_ok=True)
        file_path += f"/{request_body_digest}.pkl"

        if os.path.isfile(file_path):
            with open(file_path, "rb") as infile:
                print(f"Retrieving completion from cache file {file_path}")
                return pickle.load(infile)
        else:
            output = await self._generate_inference(session, request_body)
            _save_object_with_hashed_name(file_path, output)
            print(f"Completion has been saved as {file_path}")
            return output

    async def _generate_inference(
        self, session: aiohttp.ClientSession, request_body: str
    ):
        try:
            async with self.provider.get_inference_request(
                session, request_body
            ) as response:
                if response.status == 200:
                    output = await response.json()
                    if output["stop_reason"] == "max_tokens":
                        print("Max tokens in response reached")
                    return output
                elif response.status == 429:
                    await asyncio.sleep(5)
                    return await self._generate_inference(session, request_body)
                else:
                    print(
                        f"Error: Received status code {response.status}, Response: {response.text}"
                    )
                    return None
        except Exception as e:
            print(f"Exception occurred: {e}")
            return None

    def print_inference_cost_from_responses(self, responses: list) -> None:
        costs = self.provider.get_1k_token_input_output_cost()

        input_tokens = map(lambda x: x["usage"]["input_tokens"], responses)
        input_tokens = functools.reduce(lambda a, b: a + b, input_tokens)
        input_tokens_cost = input_tokens / 1000 * costs["input"]

        output_tokens = map(lambda x: x["usage"]["output_tokens"], responses)
        output_tokens = functools.reduce(lambda a, b: a + b, output_tokens)
        output_tokens_cost = output_tokens / 1000 * costs["output"]

        print(f"Input tokens: {input_tokens}, cost: {input_tokens_cost}")
        print(f"Output tokens: {output_tokens}, cost: {output_tokens_cost}")
        print(f"Total cost: {input_tokens_cost + output_tokens_cost}")


def _save_object_with_hashed_name(file_path, output) -> None:
    # Serialize the object
    serialized_obj = pickle.dumps(output)
    # Create a file path including the hash and save the serialized object to the file
    with open(file_path, "wb") as file:
        file.write(serialized_obj)


def _get_request_body_digest(request_body) -> str:
    return hashlib.sha256(request_body.encode("utf-8")).hexdigest()
