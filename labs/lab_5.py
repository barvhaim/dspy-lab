# Lab for exploring custom module RAG with DSPy
import logging
import dspy
from src.llm.llm import get_llm_base_params

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Custom module with implements RAG
class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        # Retrieve will use the user’s default retrieval settings
        self.retrieve = dspy.Retrieve(k=3)

        # ChainOfThought with signature that generates
        # answers given retrieval context & question .
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate_answer(context=context, question=question)


def lab_5():
    """
    Init a LM and solve a RAG reasoning problem.
    Use the built-in module "ChainOfThought" and a Wikipedia retrieval model to solve the problem.
    (https://dspy.ai/api/modules/ChainOfThought/)
    :return:
    """
    # Init a LM
    lm = dspy.LM(
        **get_llm_base_params("ibm-granite/granite-3.1-8b-instruct"),
    )

    # Init a Retriever (RM) with Wikipedia's data
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(
        url="http://20.102.90.50:2017/wiki17_abstracts"
    )
    dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts, max_tokens=1024)

    question = "What's the name of the castle that David Gregory inherited?"

    # Define a RAG module with the problem
    rag = RAG()
    response = rag(question=question)

    # Log the completion
    logger.info(f"Reasoning: {response.reasoning}")
    logger.info(f"Answer: {response.answer}")  # Answer: "Kinnairdy Castle"

    # Inspect the auto generated prompt and completion
    logger.info(lm.inspect_history(n=1))


if __name__ == "__main__":
    lab_5()
