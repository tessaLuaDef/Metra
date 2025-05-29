from datasets import Dataset
from typing import List
import spacy

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
            facts = extract_facts(row["answer"])
            present = [fact for fact in facts if any(fuzzy_in(fact, ctx) for ctx in row["contexts"])]
            score = len(present) / len(facts) if facts else 0.0
            scores.append(score)
        return scores
