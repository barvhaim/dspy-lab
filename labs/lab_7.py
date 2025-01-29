# Lab for exploring BootstrapFewShot optimization a sentiment analysis module with DSPy
import logging
import random
import dspy
from dspy.datasets import DataLoader
from dspy.teleprompt import BootstrapFewShot
from src.llm.llm import get_llm_base_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_dataset():
    # Don't change this code, we will use the IMDB dataset for this lab
    dl = DataLoader()
    imdb = dl.from_huggingface(
        "stanfordnlp/imdb",
        "plain_text",
        input_keys=("text",),
    )
    train_set, dev_set = random.sample(imdb["train"], k=30), random.sample(
        imdb["test"], k=10
    )
    return train_set, dev_set


class SentimentAnalysisSignature(dspy.Signature):
    """Predict the sentiment of the movie review"""
    text: str = dspy.InputField(desc="Movie review text")
    sentiment: int = dspy.OutputField(
        desc="generate sentiment as 1 if positive, 0 if negative"
    )


class SentimentAnalysisModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(SentimentAnalysisSignature)

    def forward(self, **kwargs):
        prediction = self.classify(**kwargs)
        return prediction.sentiment


def lab_7():
    """
    Init a LM, evaluate a sentiment analysis module,
    optimize a sentiment analysis module with BootstrapFewShot, and re-evaluate the module.
    :return:
    """
    # Init a LM
    lm = dspy.LM(
        **get_llm_base_params("ibm-granite/granite-3.1-8b-instruct"),
    )
    dspy.settings.configure(lm=lm, max_tokens=1024)

    # Load the dataset - Don't change this code
    train_set, dev_set = _load_dataset()

    # TODO: Define SentimentAnalysisModule module
    sentiment_analysis =

    # Define a metric to evaluate the module (Exact match)
    def _evaluate_sentiment(example, pred, trace=None) -> bool:
        return example["label"] == pred

    # TODO: Define an evaluator on the dev set with your metric, 24 threads and display progress
    evaluator =

    # Examine the performance of the module before optimization
    evaluator(sentiment_analysis)

    # Define an optimizer to optimize the module
    optimizer = BootstrapFewShot(
        metric=_evaluate_sentiment,
        max_bootstrapped_demos=4,  # TODO: Control How many demos teacher should create
        max_labeled_demos=16,  # TODO: Control How many demos to pick from train set
        max_rounds=1,  # Number of iterations to attempt generating the required bootstrap examples
    )

    # Optimize the module
    sentiment_analysis_optimized = optimizer.compile(
        sentiment_analysis, trainset=train_set
    )

    # Save the optimized module to disk, inspect the file to see the optimized module
    sentiment_analysis_optimized.save("lab_7_sentiment_analysis_optimized.json")

    # Examine the performance of the module after optimization
    evaluator(sentiment_analysis_optimized)

    # Bonus: Try to improve the performance of the module by using a different model,
    # or other teacher model. You can also try to change the optimizer parameters.


if __name__ == "__main__":
    lab_7()
