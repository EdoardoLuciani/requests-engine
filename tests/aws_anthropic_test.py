import pytest, requests_engine, asyncio
from dotenv import load_dotenv


@pytest.fixture(scope='session', autouse=True)
def load_env():
    load_dotenv()


@pytest.fixture()
def system_prompt():
    return "Answer the following questions accurately like a personal assistant"


@pytest.fixture()
def messages():
    bodies = ['How big is the sun? Respond with max 1 sentence.', 'How big is the moon? Respond with max 1 sentence.']
    return [requests_engine.Conversation.with_initial_message("user", body) for body in bodies]


def test_aws_batch_request(system_prompt: str, messages: list[requests_engine.Conversation]):
    prov = requests_engine.AwsProvider()
    endpoint = requests_engine.Engine(prov)

    responses = asyncio.run(endpoint.schedule_completions(
        system_prompt, messages, 0.4, "aws_tests_cache"
    ))

    requests_engine.print_inference_cost_from_responses(responses)