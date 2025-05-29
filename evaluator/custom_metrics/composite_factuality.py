from datasets import Dataset
from typing import List, Tuple
from evaluator.custom_metrics.fact_triple_coverage import FactTripleCoverage
from evaluator.custom_metrics.justified_faithfulness import JustifiedFaithfulness

class CompositeFactuality:
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta
        self.triple_metric = FactTripleCoverage()
        self.faithfulness_metric = JustifiedFaithfulness()

    def score(self, dataset: Dataset, llm) -> Tuple[List[float], List[str]]:
        triple_scores = self.triple_metric.score(dataset)
        faithfulness_scores, justifications = self.faithfulness_metric.score(dataset, llm)
        final_scores = [
            self.alpha * t + self.beta * f
            for t, f in zip(triple_scores, faithfulness_scores)
        ]
        return final_scores, justifications
