from typing import Dict
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm_base_params(model_name: str) -> Dict:
    """
    Get the base params for a given model, including the API key, base URL, and headers to use RITS.
    Don't change this function! It's used in the labs.
    :param model_name: The name of the LLM model to get the base params for.
    :return:
    """

    model_to_url_path = {
        "meta-llama/llama-3-3-70b-instruct": "llama-3-3-70b-instruct",
        "ibm-granite/granite-3.1-8b-instruct": "granite-3-1-8b-instruct",
    }

    return {
        "model": f"openai/{model_name}",
        "api_key": "NotRequiredSinceWeAreLocal",
        "api_base": f"{os.getenv('RITS_API_BASE_URL')}/{model_to_url_path.get(model_name, model_name)}/v1",
        "headers": {"RITS_API_KEY": os.getenv("RITS_API_KEY")},
    }
