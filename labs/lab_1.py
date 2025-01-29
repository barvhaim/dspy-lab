# Lab for exploring Signature and Predict with DSPy
import logging
import dspy
from src.llm.llm import get_llm_base_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def lab_1():
    """
    A *Signature* is a declarative specification of input/output behavior of a DSPy module.
    It's the most basic form of task description which simply requires inputs and outputs,
    and optionally, a small description about them and the task too.

    Let's make a signature for sentiment classification (True if positive, False if negative).
    (https://dspy.ai/learn/programming/signatures)
    :return:
    """
    # Init a LM
    lm = dspy.LM(
        **get_llm_base_params("ibm-granite/granite-3.1-8b-instruct"),
    )
    # Configure DSPy to use the LM and set the max tokens to 1024
    dspy.settings.configure(lm=lm, max_tokens=1024)

    # Classify the following sentence to determine if it's positive or negative
    sentence = "it's a charming and often affecting journey."

    # TODO: Define Inline Signature for the task (True if positive, False if negative)
    signature =

    # TODO: Define a basic Predict (https://dspy.ai/deep-dive/modules/predict/) module with the signature
    classify = dspy.Predict(signature)
    sentiment =

    # Log the completion
    logger.info(f"Sentiment: {sentiment}")  # Sentiment: True

    # Inspect the auto generated prompt and completion
    logger.info(lm.inspect_history(n=1))


if __name__ == "__main__":
    lab_1()
