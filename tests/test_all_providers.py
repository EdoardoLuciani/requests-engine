import pytest, dotenv, shutil, requests_engine, unittest, pickle, asyncio, os, base64, logging

CACHE_DIR = "tests_cache"


@pytest.fixture(scope="session", autouse=True)
def load_env():
    dotenv.load_dotenv()


@pytest.fixture(autouse=True)
def clear_cache():
    shutil.rmtree("tests_cache", ignore_errors=True)


@pytest.fixture()
def conversations():
    return [
        requests_engine.Conversation.with_initial_message("Act like a personal assistant", "user", body)
        for body in [
            "Give a number between 1 and 10",
            "Give a number between 10 and 20",
        ]
    ]


@pytest.fixture()
def aws_anthropic_provider():
    return requests_engine.providers.AwsAnthropicProvider(os.environ["AWS_ACCESS_KEY"], os.environ["AWS_SECRET_KEY"])


@pytest.fixture()
def openai_api_groq_provider():
    return requests_engine.providers.OpenAICompatibleApiProvider(
        os.environ["GROQ_API_KEY"],
        "https://api.groq.com/openai/v1/chat/completions",
        model_id="gemma2-9b-it",
    )


@pytest.fixture()
def openai_api_official_provider():
    return requests_engine.providers.OpenAICompatibleApiProvider(
        os.environ["OPENAI_API_KEY"],
        "https://api.openai.com/v1/chat/completions",
        model_id="gpt-4o-mini",
    )


@pytest.fixture()
def gcp_beta_completions_provider():
    return requests_engine.providers.GcpBetaCompletionsProvider(
        base64.b64decode(os.environ["GCP_SERVICE_CREDENTIAL_BASE64"]).decode("utf-8")
    )


@pytest.mark.parametrize(
    "provider_name",
    [
        "aws_anthropic_provider",
        "openai_api_groq_provider",
        "openai_api_official_provider",
        "gcp_beta_completions_provider",
    ],
)
def test_generate_response(
    provider_name,
    conversations: list[requests_engine.Conversation],
    caplog: pytest.LogCaptureFixture,
    request,
):
    provider = request.getfixturevalue(provider_name)
    engine = requests_engine.Engine(provider, serialization_path=CACHE_DIR)

    assert_generation_and_response_caching(engine, conversations, caplog)


def assert_generation_and_response_caching(
    engine: requests_engine.Engine,
    conversation: list[requests_engine.Conversation],
    caplog: pytest.LogCaptureFixture
):
    job_cache_dir = f"{CACHE_DIR}/{engine.provider.__class__.__name__}"

    completions = asyncio.run(engine.schedule_completions(conversation, 0.4, engine.provider.__class__.__name__))
    common_assert(engine, conversation, completions)
    assert (("root", logging.INFO, f"Retrieving completion from cache file {job_cache_dir}") in caplog.record_tuples,
       "Generation was retrieved from cache, when it should have not")

    stats = engine.get_cost_from_completions(completions)
    assert all(stats)

    completions = asyncio.run(engine.schedule_completions(conversation, 0.4, engine.provider.__class__.__name__))
    common_assert(engine, conversation, completions)
    assert (("root", logging.INFO, f"Retrieving completion from cache file {job_cache_dir}") in caplog.record_tuples,
        "Generation was not retrieved from cache")


def common_assert(
    engine: requests_engine.Engine,
    conversations: list[requests_engine.Conversation],
    completions: list,
):
    responses = [e["response"] for e in completions]

    assert len(responses) == len(conversations)
    assert all(responses)

    job_cache_dir = f"{CACHE_DIR}/{engine.provider.__class__.__name__}"

    pickle_data = {}
    for filename in os.listdir(job_cache_dir):
        file_path = os.path.join(job_cache_dir, filename)
        with open(file_path, "rb") as f:
            pickle_data[filename] = pickle.load(f)

    unittest.TestCase().assertCountEqual(first=list(pickle_data.values()), second=responses)
