import argparse
import os
from datasets import load_dataset
from evaluator.custom_metrics.composite_factuality import CompositeFactuality
from evaluator.custom_metrics.counterfactual_fairness import CounterfactualFairnessEvaluator
from evaluator.custom_metrics.ethical_alignment import evaluate_ethics
from evaluator.custom_metrics.menli_accuracy import MENLIPrecision
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama  # Puedes extender con más modelos si quieres

# Load environment variables (optional use of .env file)
load_dotenv()

# -------------------------
# Load LLM based on model name
# -------------------------
def load_llm(model_name: str):
    """
    Given a model name, load the appropriate LLM.
    Currently supports Ollama backend.
    """
    # TODO: extender en el futuro a OpenAI, Hugging Face, etc.
    return ChatOllama(model=model_name)

# -------------------------
# Main Evaluation Function
# -------------------------
def main(input_file: str, model_name: str):
    dataset_hf = load_dataset("json", data_files=input_file)["train"]
    dataset = [dict(row) for row in dataset_hf]

    # Load model
    llm = load_llm(model_name)

    # Extract folder and filename for routing
    file_path = input_file.lower()
    folder_name = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    if "ethic" in folder_name or "ethic" in filename:
        print("\nEvaluating ethical dimension...\n")
        ethics_result = evaluate_ethics(dataset)
        print("--- EthicalAlignment Results ---")
        print(f"Benchmark (ETHICS):        {ethics_result['benchmark_score']:.2f}")
        print(f"Inverse toxicity:          {ethics_result['avg_toxicity_score']:.2f}")
        print(f"Final score:               {ethics_result['final_score']:.2f}")

    elif "fairness" in folder_name or "fairness" in filename:
        print("\nEvaluating counterfactual fairness...\n")
        evaluator = CounterfactualFairnessEvaluator()
        fairness_results = evaluator.score(dataset, llm, explain=True)

        for i, result in enumerate(fairness_results):
            print(f"\n--- Entry {i + 1} ---")
            print(f"Fairness - Final score:      {result['final_score']:.2f}")
            print(f"* LLM score:                 {result['llm_score']:.2f}")
            print(f"* Sentiment score:           {result['sentiment_score']:.2f}")
            print(f"LLM Justification:           {result['justification']}")

    elif "precision" in folder_name or "precision" in filename:
        print("\nEvaluating precision with MENLI...\n")
        precision_scores = MENLIPrecision().score(dataset)
        for i, score in enumerate(precision_scores):
            print(f"\n--- Entry {i + 1} ---")
            print(f"Precision - Score: {score['score']:.2f} | Category: {score['category']}")
            if 'justification' in score:
                print(f"Justification: {score['justification']}")

    else:
        print("\nEvaluating composite factuality...\n")
        scores, justifications = CompositeFactuality().score(dataset, llm)
        for i, (score, explanation) in enumerate(zip(scores, justifications)):
            print(f"\n--- Entry {i + 1} ---")
            print(f"Score: {score:.2f}")
            print(f"Justification: {explanation}")

# -------------------------
# CLI Entry Point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input .json file")
    parser.add_argument("--model", type=str, default="gemma:2b", help="Model to evaluate (e.g. llama3, mistral)")
    args = parser.parse_args()

    main(args.input, args.model)
