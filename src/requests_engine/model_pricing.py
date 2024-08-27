from typing import Dict, Tuple, TypedDict


class InputOutputPricing(TypedDict):
    input_tokens_cost: float
    output_tokens_cost: float


class ModelPricing:
    _models: Dict[str, InputOutputPricing] = {
        # OpenAI
        "gpt-4o-mini": {"input_tokens_cost": 0.15, "output_tokens_cost": 0.6},
        # Groq
        "llama-3.1-70b-versatile": {
            "input_tokens_cost": 0.59,
            "output_tokens_cost": 0.79,
        },
        "gemma2-9b-it": {"input_tokens_cost": 0.20, "output_tokens_cost": 0.20},
        # AWS
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "input_tokens_cost": 0.25,
            "output_tokens_cost": 1.25,
        },
        # GCP
        "meta/llama3-405b-instruct-maas": {
            "input_tokens_cost": 5.32,
            "output_tokens_cost": 16,
        },
    }

    @staticmethod
    def get_model_pricing(model: str) -> InputOutputPricing:
        return ModelPricing._models[model]

    @staticmethod
    def get_cost_from_tokens_count(model: str, input_tokens: int, output_tokens: int) -> InputOutputPricing:
        model_price = ModelPricing.get_model_pricing(model)
        input_tokens_cost = (input_tokens / 1_000_000) * model_price["input_tokens_cost"]
        output_tokens_cost = (output_tokens / 1_000_000) * model_price["output_tokens_cost"]
        return {
            "input_tokens_cost": input_tokens_cost,
            "output_tokens_cost": output_tokens_cost,
        }
