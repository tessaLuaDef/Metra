from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class MENLIPrecision:
    def __init__(self):
        self.model_name = "facebook/bart-large-mnli"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def score(self, dataset: Dataset):
        scores = []
        for row in dataset:
            context = " ".join(row["contexts"])
            answer = row["answer"]
            inputs = self.tokenizer(context, answer, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                prob = torch.softmax(logits, dim=-1)[0][1].item()  # clase "factual"
                scores.append(prob)
        return scores
