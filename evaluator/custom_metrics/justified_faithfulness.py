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
Eres un evaluador experto encargado de puntuar qué tan bien una respuesta está respaldada por un contexto.

Sigue estrictamente estas instrucciones:
1. Lee el contexto y la respuesta.
2. Evalúa cuánto de la respuesta está contenido explícitamente en el contexto.
3. Devuelve una puntuación **numérica entre 0 y 1 en pasos de 0.25**.
   Usa exactamente uno de estos valores: 1.00, 0.75, 0.50, 0.25, 0.00

Formato de salida:
Score: <una de las cinco puntuaciones permitidas>  
Justificación: <explicación clara y breve de tu decisión, en español>

Contexto: {context}  
Respuesta: {answer}
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
        match = re.search(r"Justificación\s*[:=]?\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Justificación no encontrada."
