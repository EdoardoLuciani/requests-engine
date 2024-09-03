import aiohttp, pickle, hashlib, os, asyncio, pathlib, traceback

from typing import Any
from .providers.abstract_provider import AbstractProvider
from .conversation import Conversation


class Engine:
    def __init__(self, provider: AbstractProvider, serialization_path: str = "cache", max_inflight_requests: int = 32):
        self.serialization_path = serialization_path
        self.provider = provider
        self.max_inflight_requests = max_inflight_requests

    async def schedule_completions(
        self, conversations: list[Conversation], temperature: float, task_name: str
    ) -> list[tuple[Any, str]]:
        semaphore = asyncio.Semaphore(self.max_inflight_requests)
        async with aiohttp.ClientSession() as session:

            async def task(conversation):
                async with semaphore:
                    return await self._get_or_generate_completion(
                        session, conversation, temperature, task_name
                    )

            return await asyncio.gather(*(task(conversation) for conversation in conversations))

    async def _get_or_generate_completion(
        self,
        session: aiohttp.ClientSession,
        conversation: Conversation,
        temperature: float,
        task_name: str,
    ) -> tuple[Any, str]:
        request_body = self.provider.get_request_body(conversation, temperature)
        request_body_digest = _get_request_body_digest(request_body)

        file_path = f"{self.serialization_path}/{task_name}"
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
        file_path += f"/{request_body_digest}.pkl"

        if os.path.isfile(file_path):
            with open(file_path, "rb") as infile:
                print(f"Retrieving completion from cache file {file_path}")
                return (pickle.load(infile), request_body_digest)
        else:
            output = await self._generate_completion(session, request_body)
            _save_object_with_hashed_name(file_path, output)
            print(f"Completion has been saved as {file_path}")
            return (output, request_body_digest)

    async def _generate_completion(self, session: aiohttp.ClientSession, request_body: str):
        try:
            print(f"Sending request to provider {self.provider.__class__.__name__}")
            async with self.provider._get_completion_request(session, request_body) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    await asyncio.sleep(5)
                    return await self._generate_completion(session, request_body)
                else:
                    print(f"Error: {response}")
                    return None
        except Exception as e:
            print(f"Exception occurred: {traceback.print_exc()}")
            return None


def _save_object_with_hashed_name(file_path, output) -> None:
    # Serialize the object
    serialized_obj = pickle.dumps(output)
    # Create a file path including the hash and save the serialized object to the file
    with open(file_path, "wb") as file:
        file.write(serialized_obj)


def _get_request_body_digest(request_body) -> str:
    return hashlib.sha256(request_body.encode("utf-8")).hexdigest()
