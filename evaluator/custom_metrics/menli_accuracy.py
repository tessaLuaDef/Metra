from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from langchain_community.chat_models import ChatOllama
import torch

class MENLIPrecision:
    def __init__(self, model_name="facebook/bart-large-mnli", ollama_model="mistral", generate_explanation=True):
        # NLI model setup
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Explanation config
        self.generate_explanation = generate_explanation
        if self.generate_explanation:
            self.explainer = ChatOllama(model=ollama_model)

    def _categorize_score(self, score):
        if score >= 0.9:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"

    def _generate_justification(self, context, answer):
        prompt = (
            f"Context: {context}\n"
            f"Answer: {answer}\n"
            "Is the answer a correct inference from the context? Justify your reasoning in a few sentences."
        )
        return self.explainer.invoke(prompt).content.strip()

    def score(self, dataset):
        results = []
        for row in dataset:
            context = " ".join(row["contexts"])
            answer = row["answer"]

            inputs = self.tokenizer(context, answer, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                entailment_score = torch.softmax(logits, dim=-1)[0][2].item()

            result = {
                "score": entailment_score,
                "category": self._categorize_score(entailment_score)
            }

            if self.generate_explanation:
                result["justification"] = self._generate_justification(context, answer)

            results.append(result)

        return results