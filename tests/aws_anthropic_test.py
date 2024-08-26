import pytest, requests_engine, asyncio, shutil, pickle, os, unittest
from dotenv import load_dotenv


@pytest.fixture(scope='session', autouse=True)
def load_env():
    load_dotenv()


@pytest.fixture()
def system_prompt():
    return "Act like a personal assistant"


@pytest.fixture()
def messages():
    bodies = ['Give a number between 1 and 10', 'Give a number between 10 and 20']
    return [requests_engine.Conversation.with_initial_message("user", body) for body in bodies]


def test_aws_batch_request(system_prompt: str, messages: list[requests_engine.Conversation]):
    shutil.rmtree('cache/aws_tests_cache', ignore_errors=True)

    prov = requests_engine.AwsProvider()
    endpoint = requests_engine.Engine(prov)

    responses = asyncio.run(endpoint.schedule_completions(
        system_prompt, messages, 0.4, "aws_tests_cache"
    ))

    assert len(responses) == len(messages)
    assert all(responses) == True

    stats = endpoint.get_inference_cost_from_responses(responses)
    assert all(stats) == True

    pickle_data = {}
    for filename in os.listdir('cache/aws_tests_cache'):
        file_path = os.path.join('cache/aws_tests_cache', filename)
        with open(file_path, 'rb') as f:
            pickle_data[filename] = pickle.load(f)

    unittest.TestCase().assertCountEqual(first=list(pickle_data.values()), second=responses)