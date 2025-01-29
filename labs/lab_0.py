# Lab for sanity check working with DSPy + RITS
import logging
import dspy
from src.llm.llm import get_llm_base_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def lab_0():
    """
    Init an LM and make a simple completion request, just to make sure everything is working.
    https://dspy.ai/learn/programming/language_models/ (see "Calling the LM directly")
    :return:
    """
    # Init a LM (dsp.LM) with the model "ibm-granite/granite-3.1-8b-instruct"
    # use `get_llm_base_params` to get the technical params, e.g. API key, base URL
    # to use with RITS (Internal IBM Research inference service)
    lm = dspy.LM(
        **get_llm_base_params(model_name="ibm-granite/granite-3.1-8b-instruct"),
    )

    # Make a completion request with the prompt "What is the capital of China?" and temperature 0.01
    completion = lm("What is the capital of China?", temperature=0.01)

    # Log the completion
    logger.info(completion)  # "Beijing"


if __name__ == "__main__":
    lab_0()
