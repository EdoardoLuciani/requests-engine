import requests_engine, os, asyncio
from dotenv import load_dotenv

if __name__ == '__main__':
    # Loading OPENAI_API_KEY from an .env file
    load_dotenv()

    provider = requests_engine.providers.OpenAICompatibleApiProvider(
        os.environ["OPENAI_API_KEY"],
        "https://api.openai.com/v1/chat/completions",
        model_id="gpt-4o-mini",
    )
    engine = requests_engine.Engine(provider)

    conversations = [
        requests_engine.Conversation.with_initial_message('You are an assistant. Answer shortly', 'user', e)
        for e in ['How big is the moon? ', 'How big is the sun?']
    ]
    completions = asyncio.run(engine.schedule_completions(conversations, 0.3, 'example'))

    print(completions)
    print(engine.get_cost_from_completions(completions))
