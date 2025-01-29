# Lab for exploring Chain-of-Thought reasoning with examples from math Q&A dataset
import logging
import dspy
from src.llm.llm import get_llm_base_params

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _get_few_shot_examples():
    math_qna = [
        {"question": "1+1+5=?", "answer": "7"},
        {"question": "1+5+5=?", "answer": "11"},
        {"question": "1+1+1=?", "answer": "3"},
        {"question": "3+3+5=?", "answer": "11"},
    ]
    return [
        dspy.Example(question=qa["question"], answer=qa["answer"]) for qa in math_qna
    ]


def lab_4():
    """
    Init a LM and solve a Chain-of-Thought reasoning problem with Few-shot examples.
    Use the built-in module "ChainOfThought" and examples to solve the problem.
    (https://dspy.ai/api/modules/ChainOfThought/)
    :return:
    """
    # Init a LM
    lm = dspy.LM(
        **get_llm_base_params("ibm-granite/granite-3.1-8b-instruct"),
    )
    dspy.settings.configure(lm=lm, max_tokens=1024)

    question = "3+3+5=?"

    # Define a ChainOfThought module with the problem, add examples (demos) parameter to the module
    ask = dspy.ChainOfThought("question -> answer")
    response = ask(question=question, demos=_get_few_shot_examples())

    # Log the completion
    logger.info(f"Reasoning: {response.reasoning}")
    logger.info(f"Answer: {response.answer}")  # Answer: "11"

    # Inspect the auto generated prompt and completion
    logger.info(lm.inspect_history(n=1))


if __name__ == "__main__":
    lab_4()
