# evaluator/metric.py
from evaluator.custom_metrics.composite_factuality import CompositeFactuality
from evaluator.custom_metrics.ethical_alignment import EthicalEvaluator


composite_factuality = CompositeFactuality()
ethical_Evaluator = EthicalEvaluator()