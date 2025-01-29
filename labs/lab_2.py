# Lab for exploring Signature and Predict with DSPy
import logging
import dspy
from src.llm.llm import get_llm_base_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def lab_2():
    """
    A *Signature* is a declarative specification of input/output behavior of a DSPy module.
    It's the most basic form of task description which simply requires inputs and outputs,
    and optionally, a small description about them and the task too.

    Let's make a complex signature for Basic Q&A task.
    (https://dspy.ai/learn/programming/signatures)
    :return:
    """
    # Init a LM
    lm = dspy.LM(
        **get_llm_base_params("ibm-granite/granite-3.1-8b-instruct"),
    )
    # Configure DSPy to use the LM and set the max tokens to 1024
    dspy.settings.configure(lm=lm, max_tokens=1024)

    # TODO: Define class Signature for Basic Q&A task, with input question and output answer
    class BasicQA(dspy.Signature):
        """Answer the question with short factoid answer"""
        question =
        answer =

    question = "Who is the winner of the 2020 US presidential election?"

    # TODO: Define a Predict module with the signature, and ask the question
    ask =
    answer =

    # Log the completion
    logger.info(f"Answer: {answer}")  # Answer: "Joe Biden"

    # Inspect the auto generated prompt and completion
    logger.info(lm.inspect_history(n=1))


if __name__ == "__main__":
    lab_2()
