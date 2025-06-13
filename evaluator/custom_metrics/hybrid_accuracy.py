from typing import List
import re
from sentence_transformers import SentenceTransformer, util
import bert_score


class HybridAccuracyMetric:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2", bert_lang="en"):
        # Load embedding model for cosine similarity
        self.embed_model = SentenceTransformer(embed_model_name)
        self.bert_lang = bert_lang  # Language setting for BERTScore

    def normalize(self, text: str) -> str:
        # Lowercase, remove punctuation, and collapse spaces
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def compute(self, dataset: List[dict], alpha=0.3) -> List[dict]:
        # alpha = weight of cosine similarity (0.3), the rest is BERTScore (0.7)
        results = []

        # Preprocess reference and candidate answers
        references = [self.normalize(d["answer"]) for d in dataset]
        candidates = [self.normalize(d["model_answer"]) for d in dataset]

        # Compute BERTScore (F1) for all pairs
        P, R, F1 = bert_score.score(candidates, references, lang=self.bert_lang, verbose=False)
        bert_f1 = [float(f.item()) for f in F1]

        for i, entry in enumerate(dataset):
            ref = references[i]
            cand = candidates[i]

            # Compute cosine similarity between sentence embeddings
            if ref and cand:
                emb_ref = self.embed_model.encode(ref, convert_to_tensor=True)
                emb_cand = self.embed_model.encode(cand, convert_to_tensor=True)
                sim = float(util.cos_sim(emb_ref, emb_cand)[0][0])
                sim = max(0.0, sim)  # Clamp negative scores to 0
            else:
                sim = 0.0

            # Weighted hybrid score
            final_score = alpha * sim + (1 - alpha) * bert_f1[i]

            results.append({
                "question": entry["question"],
                "expected_answer": entry["answer"],
                "model_answer": entry["model_answer"],
                "cosine_similarity": round(sim, 3),
                "bertscore_f1": round(bert_f1[i], 3),
                "hybrid_score": round(final_score, 3)
            })

        return results
