# Lab for exploring MIPRO optimization over GSM8K with DSPy
import logging
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import MIPROv2
from src.llm.llm import get_llm_base_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_dataset():
    # Don't change this code, we will use the GSM8K dataset
    gsm8k = GSM8K()
    train_set, dev_set = gsm8k.train, gsm8k.dev
    return train_set, dev_set


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


def lab_8():
    """
    Init a LM, evaluate a CoT module with GSM8K,
    optimize the CoT module with a MIPRO optimizer, and evaluate the optimized module.
    :return:
    """
    # Init a LM
    lm = dspy.LM(
        **get_llm_base_params("ibm-granite/granite-3.1-8b-instruct"),
    )
    dspy.settings.configure(lm=lm, max_tokens=1024)

    # Load the dataset - Don't change this code
    train_set, dev_set = _load_dataset()

    # Define a CoT module
    cot = CoT()

    # Define an evaluator on the dev set with your metric
    evaluator = dspy.Evaluate(
        devset=dev_set[:30],
        metric=gsm8k_metric,
        num_threads=24,
        display_progress=True,
    )

    # Examine the performance of the module before optimization
    evaluator(cot)

    cot.save("lab_8_cot.json")  # TODO: Examine the file to see the module

    # Define an MIPROv2 optimizer with GSM8K metric and light optimization
    optimizer = MIPROv2(
        metric=gsm8k_metric,
        auto="light",  # Can choose between light, medium, and heavy optimization runs
    )

    # Optimize the module
    optimized_cot = optimizer.compile(
        cot.deepcopy(),
        trainset=train_set[:100],
        max_bootstrapped_demos=3,
        max_labeled_demos=5,
    )

    # TODO: Save the optimized module to disk, inspect the file to see the optimized module and compare it to the original
    optimized_cot.save("lab_8_cot_optimized.json")

    # Examine the performance of the module after optimization
    evaluator(optimized_cot)

    # TODO: Try to improve the performance of the module by using a different model,
    # or other teacher model. You can also try to change the optimizer parameters.


if __name__ == "__main__":
    lab_8()
