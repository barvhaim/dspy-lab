# Lab for exploring evaluation a Q&A module with DSPy
import logging
import ujson
import dspy
from dspy.evaluate import SemanticF1
from dspy.utils import download
from src.llm.llm import get_llm_base_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_dataset():
    # Don't change this code
    download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_500.json")
    with open("ragqa_arena_tech_500.json", "r") as f:
        dataset = ujson.load(f)
    examples = [dspy.Example(**d).with_inputs("question") for d in dataset]
    dev_set = examples[:10]
    return dev_set


def lab_6():
    """
    Init a LM and evaluate your module on a Q&A dataset using SemanticF1 metric
    Read more about SemanticF1 metric:
    https://github.com/stanfordnlp/dspy/blob/main/dspy/evaluate/auto_evaluation.py#L21

    :return:
    """
    # Init a LM
    lm = dspy.LM(
        **get_llm_base_params("ibm-granite/granite-3.1-8b-instruct"),
    )
    dspy.settings.configure(lm=lm, max_tokens=1024)

    # Load the dataset - [{"question": "What is the capital of France?", "response": "Paris"}, ...]
    development_set = _load_dataset()

    # TODO: Define a SemanticF1 metric and a ChainOfThought module for the question-answering task
    metric =
    ask =

    # Define an evaluator on the dev set with your metric
    evaluator = dspy.Evaluate(
        devset=development_set,
        metric=metric,
        num_threads=24,
        display_progress=True,
    )

    # Evaluate the module on the dev set
    evaluator(ask)

    # # Inspect the auto generated prompt and completion
    logger.info(lm.inspect_history(n=1))

    # Bonus: 1. Try to improve the performance of the module by using a different model
    # Bonus: 2. Try to use a different metric
    #        (LLM as a Judge like, https://dspy.ai/cheatsheet/#llm-as-judge)


if __name__ == "__main__":
    lab_6()
