from typing import TypedDict

class ModelBatchInferenceCost(TypedDict):
    input_tokens: int
    input_tokens_cost: float
    output_tokens: int    
    output_tokens_cost: float