# composite_factuality.py

from datasets import Dataset
from typing import List, Tuple
from langchain_core.language_models import BaseChatModel
import spacy
import re

# -------------------------------
# MÉTRICA: FactTripleCoverage
# -------------------------------
nlp = spacy.load("en_core_web_sm")

def extract_facts(text: str) -> List[str]:
    doc = nlp(text)
    triples = []
    for sent in doc.sents:
        subj, verb, obj = "", "", ""
        for token in sent:
            if token.dep_ == "ROOT":
                verb = token.lemma_
                subj = next((w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")), "")
                obj = next((w.text for w in token.rights if w.dep_ in ("dobj", "pobj", "attr")), "")
                if subj and verb and obj:
                    triples.append(f"{subj.lower()}-{verb.lower()}-{obj.lower()}")
    return triples

def fuzzy_in(fact: str, context: str) -> bool:
    return all(part in context.lower() for part in fact.split('-'))

class FactTripleCoverage:
    def score(self, dataset: Dataset) -> List[float]:
        scores = []
        for row in dataset:
            facts = extract_facts(row.get("answer", row.get("model_answer", "")))
            context = row.get("context", " ".join(row.get("contexts", [])))
            present = [fact for fact in facts if fuzzy_in(fact, context)]
            score = len(present) / len(facts) if facts else 0.0
            scores.append(score)
        return scores


# -------------------------------
# MÉTRICA: JustifiedFaithfulness
# -------------------------------
class JustifiedFaithfulness:
    def score(self, dataset: Dataset, llm: BaseChatModel) -> Tuple[List[float], List[str]]:
        scores = []
        justifications = []

        for row in dataset:
            context = row.get("context", " ".join(row.get("contexts", [])))
            answer = row.get("answer", row.get("model_answer", ""))
            prompt = f"""
You are an expert evaluator assigned to assess how well an answer is supported by the given context.

Follow these instructions carefully:
1. Read the context and the answer.
2. Evaluate how much of the answer is explicitly stated in the context.
3. Return a **numerical score between 0 and 1 in steps of 0.25**.
   Use exactly one of these values: 1.00, 0.75, 0.50, 0.25, 0.00

Output format:
Score: <one of the five allowed values>  
Justification: <a clear and concise explanation of your decision, in English>

Context: {context}  
Answer: {answer}
"""

            output_obj = llm.invoke([{"role": "user", "content": prompt}])
            output = output_obj.content if hasattr(output_obj, "content") else str(output_obj)

            score = self._parse_score(output)
            justification = self._parse_justification(output)

            scores.append(score)
            justifications.append(justification)

        return scores, justifications

    def _parse_score(self, text: str) -> float:
        match = re.search(r"Score\s*[:=]?\s*([01](?:\.0+|\.25|\.50|\.75|\.00)?)", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 0.0

    def _parse_justification(self, text: str) -> str:
        match = re.search(r"Justification\s*[:=]?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Justification not found."


# -------------------------------
# MÉTRICA COMPUESTA: CompositeFactuality
# -------------------------------
class CompositeFactuality:
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta
        self.triple_metric = FactTripleCoverage()
        self.faithfulness_metric = JustifiedFaithfulness()

    def score(self, dataset: Dataset, llm: BaseChatModel) -> Tuple[List[float], List[str]]:
        triple_scores = self.triple_metric.score(dataset)
        faithfulness_scores, justifications = self.faithfulness_metric.score(dataset, llm)
        final_scores = [
            self.alpha * t + self.beta * f
            for t, f in zip(triple_scores, faithfulness_scores)
        ]
        return final_scores, justifications
