import pytest, dotenv, shutil, requests_engine, unittest, pickle, asyncio, os

CACHE_DIR = 'tests_cache'


@pytest.fixture(scope='session', autouse=True)
def load_env():
    dotenv.load_dotenv()


@pytest.fixture(autouse=True)
def clear_cache():
    shutil.rmtree("tests_cache", ignore_errors=True)


@pytest.fixture()
def system_prompt():
    return "Act like a personal assistant"


@pytest.fixture()
def messages():
    return [
        requests_engine.Conversation.with_initial_message("user", body)
        for body in ["Give a number between 1 and 10", "Give a number between 10 and 20"]
    ]


@pytest.fixture()
def aws_provider():
    return requests_engine.providers.AwsProvider()


@pytest.fixture()
def openai_api_groq_provider():
    return requests_engine.providers.OpenAICompatibleApiProvider(os.environ['GROQ_API_KEY'], "https://api.groq.com/openai/v1/chat/completions", model_id='gemma2-9b-it')


@pytest.fixture()
def openai_api_official_provider():
    return requests_engine.providers.OpenAICompatibleApiProvider(os.environ['OPENAI_API_KEY'], "https://api.openai.com/v1/chat/completions", model_id='gpt-4o-mini')


def common_assert(engine: requests_engine.Engine, messages: list[requests_engine.Conversation], responses: list):
    assert len(responses) == len(messages)
    assert all(responses) == True

    job_cache_dir = f"{CACHE_DIR}/{engine.provider.__class__.__name__}"

    pickle_data = {}
    for filename in os.listdir(job_cache_dir):
        file_path = os.path.join(job_cache_dir, filename)
        with open(file_path, "rb") as f:
            pickle_data[filename] = pickle.load(f)

    unittest.TestCase().assertCountEqual(
        first=list(pickle_data.values()), second=responses
    )


def assert_generation_and_response_caching(engine: requests_engine.Engine, system_prompt: str, messages: list[requests_engine.Conversation], capsys: pytest.CaptureFixture,):
    job_cache_dir = f"{CACHE_DIR}/{engine.provider.__class__.__name__}"

    responses = asyncio.run(
        engine.schedule_completions(system_prompt, messages, 0.4, engine.provider.__class__.__name__)
    )
    common_assert(engine, messages, responses)
    assert f"Retrieving completion from cache file {job_cache_dir}" not in capsys.readouterr().out, "Generation was retrieved from cache, when it should have not"

    stats = engine.provider.get_cost_from_completions(responses)
    assert all(stats) == True

    responses = asyncio.run(
        engine.schedule_completions(system_prompt, messages, 0.4, engine.provider.__class__.__name__)
    )
    common_assert(engine, messages, responses)
    assert f"Retrieving completion from cache file {job_cache_dir}" in capsys.readouterr().out, "Generation was not retrieved from cache"


@pytest.mark.parametrize("provider_name", ['aws_provider', 'openai_api_groq_provider', 'openai_api_official_provider'])
def test_generate_response(provider_name, system_prompt: str, messages: list[requests_engine.Conversation], capsys: pytest.CaptureFixture, request):
    provider = request.getfixturevalue(provider_name)
    engine = requests_engine.Engine(provider, serialization_path=CACHE_DIR)

    assert_generation_and_response_caching(engine, system_prompt, messages, capsys)