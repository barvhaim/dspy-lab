# Lab for exploring built-in module "Chain-of-Thought" reasoning with DSPy
import logging
import dspy
from src.llm.llm import get_llm_base_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def lab_3():
    """
    Init a LM and solve a Chain-of-Thought reasoning problem.
    Use the built-in module "ChainOfThought" to solve the problem.
    (https://dspy.ai/api/modules/ChainOfThought/)
    :return:
    """
    # Init a LM
    lm = dspy.LM(
        **get_llm_base_params("ibm-granite/granite-3.1-8b-instruct"),
    )
    dspy.settings.configure(lm=lm, max_tokens=1024)

    question = "Two dice are tossed. What is the probability that the sum equals two?"

    # TODO: Define a ChainOfThought module with the problem, answer should be a float
    ask =
    response = ask(question=question)

    # Log the completion
    logger.info(f"Reasoning: {response.reasoning}")
    logger.info(f"Answer: {response.answer}")  # Answer: 0.027777777777777776

    # Inspect the auto generated prompt and completion
    logger.info(lm.inspect_history(n=1))


if __name__ == "__main__":
    lab_3()
