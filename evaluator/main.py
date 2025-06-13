import argparse
import os
import json
from datasets import load_dataset
from evaluator.custom_metrics.composite_factuality import CompositeFactuality
from evaluator.custom_metrics.counterfactual_fairness import CounterfactualFairnessEvaluator
from evaluator.custom_metrics.ethical_alignment import evaluate_ethics
from evaluator.custom_metrics.hybrid_accuracy import HybridAccuracyMetric
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

# Load environment variables (e.g. for API keys if needed)
load_dotenv()


# -- LLM Loader ---------------------------------------------------------------

def load_llm(model_name: str, temperature: float = 0.0):
    return ChatOllama(model=model_name, temperature=temperature)


# -- Main Evaluation Function -------------------------------------------------

def run_evaluation(input_file: str, model_name: str, output_file: str):
    # Load dataset from JSON file
    dataset_hf = load_dataset("json", data_files=input_file)["train"]
    dataset = [dict(row) for row in dataset_hf]

    llm = load_llm(model_name, temperature=0.7) 
    file_path = input_file.lower()
    folder_name = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    results = []

    # -- Ethical Evaluation ---------------------------------------------------

    if "ethics" in folder_name or "ethics" in filename:
        print("\nEvaluating ethical dimension...\n")

        print("--- Per-example Evaluation ---")
        for i, example in enumerate(dataset):
            result = evaluate_ethics([example], alpha=0.7)

            entry = {
                "question": example["question"],
                "model_answer": example["model_answer"],
                "expected_answer": example["expected_answer"],
                "benchmark_score": result["benchmark_score"],
                "toxicity_score": result["avg_toxicity_score"],
                "final_score": result["final_score"]
            }
            results.append(entry)

            print(f"[{i+1:02}] Benchmark: {entry['benchmark_score']:.2f} | Toxicity: {entry['toxicity_score']:.2f} | Final: {entry['final_score']:.2f}")

        avg_benchmark = sum(e["benchmark_score"] for e in results) / len(results)
        avg_toxicity = sum(e["toxicity_score"] for e in results) / len(results)
        avg_final = sum(e["final_score"] for e in results) / len(results)

        print("\n--- EthicalAlignment Results ---")
        print(f"Benchmark (ETHICS):        {avg_benchmark:.2f}")
        print(f"Inverse toxicity:          {avg_toxicity:.2f}")
        print(f"Final score:               {avg_final:.2f}")

    # -- Fairness Evaluation --------------------------------------------------

    elif "fairness" in folder_name or "fairness" in filename:
        print("\nEvaluating counterfactual fairness...\n")
        evaluator = CounterfactualFairnessEvaluator()
        fairness_results = evaluator.score(dataset, llm, explain=True)

        for i, result in enumerate(fairness_results):
            print(f"--- Entry {i + 1} ---")
            print(f"Fairness - Final score:      {result['final_score']:.2f}")
            print(f"LLM score:                 {result['llm_score']:.2f}")
            print(f"Sentiment score:           {result['sentiment_score']:.2f}")
            print(f"Justification:               {result['justification']}\n")
            results.append(result)

        print("\n--- Fairness Results ---")
        print(f"Average LLM score:           {sum(r['llm_score'] for r in results) / len(results):.2f}")
        print(f"Average sentiment score:     {sum(r['sentiment_score'] for r in results) / len(results):.2f}")
        print(f"Average final score:         {sum(r['final_score'] for r in results) / len(results):.2f}")

    # -- Accuracy Evaluation --------------------------------------------------

    elif "accuracy" in folder_name or "accuracy" in filename:
        print("\nEvaluating accuracy with HybridAccuracyMetric...\n")
        evaluator = HybridAccuracyMetric()
        results = evaluator.compute(dataset)

        for i, r in enumerate(results):
            print(f"--- Entry {i + 1} ---")
            print(f"Hybrid Score:      {r['hybrid_score']:.2f}")
            print(f"Cosine Similarity: {r['cosine_similarity']:.2f}")
            print(f"BERTScore F1:      {r['bertscore_f1']:.2f}")
            print()

        avg_score = sum(r["hybrid_score"] for r in results) / len(results)
        print("\n--- Hybrid Accuracy Summary ---")
        print(f"Average Hybrid Score: {avg_score:.2f}")

    # -- Factuality Evaluation ------------------------------------------------

    else:
        print("\nEvaluating composite factuality...\n")
        scores, justifications = CompositeFactuality().score(dataset, llm)

        for i, (score, explanation) in enumerate(zip(scores, justifications)):
            entry = {
                "question": dataset[i]["question"],
                "model_answer": dataset[i]["model_answer"],
                "context": dataset[i].get("context", ""),
                "score": score,
                "justification": explanation
            }
            results.append(entry)

            print(f"--- Entry {i + 1} ---")
            print(f"Score: {score:.2f}")
            print(f"Justification: {explanation}")

        print("\n--- Composite Factuality Results ---")
        print(f"Average Factuality Score:    {sum(r['score'] for r in results) / len(results):.2f}")

    # -- Save Output ----------------------------------------------------------

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults stored in: {output_file}")


# -- CLI wrapper for pyproject.toml ------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, required=True, choices=["accuracy", "factuality", "ethics", "fairness"],
                        help="Name of the metric to be evaluated")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to be evaluated (e.g., mistral, llama3)")
    parser.add_argument("--output", type=str, required=False,
                        help="Path to save the results (by default: results/{metric}_results_{model}.json)")

    args = parser.parse_args()

    
    input_path = os.path.join(
        "data",
        f"{args.metric}_datasets",
        f"{args.metric}_test_{args.model}.json"
    )

    print(f"\nUsing dataset: {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    output_path = args.output or os.path.join(
        "results",
        f"{args.metric}_results_{args.model}.json"
    )

    run_evaluation(input_path, args.model, output_path)


if __name__ == "__main__":
    main()
