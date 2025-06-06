from langchain_core.language_models import BaseChatModel
from datasets import Dataset
import re

class JustifiedFaithfulness:
    def score(self, dataset: Dataset, llm: BaseChatModel):
        scores = []
        justifications = []

        for row in dataset:
            context = " ".join(row["contexts"])
            answer = row["answer"]
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
