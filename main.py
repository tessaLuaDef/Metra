import argparse
import os
import json
from datasets import load_dataset
from evaluator.custom_metrics.composite_factuality import CompositeFactuality
from evaluator.custom_metrics.counterfactual_fairness import CounterfactualFairnessEvaluator
from evaluator.custom_metrics.ethical_alignment import evaluate_ethics
from evaluator.custom_metrics.menli_accuracy import MENLIAccuracy
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

# Load environment variables
load_dotenv()

def load_llm(model_name: str):
    return ChatOllama(model=model_name)

def main(input_file: str, model_name: str, output_file: str):
    dataset_hf = load_dataset("json", data_files=input_file)["train"]
    dataset = [dict(row) for row in dataset_hf]

    llm = load_llm(model_name)

    file_path = input_file.lower()
    folder_name = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    results = []

    if "ethic" in folder_name or "ethic" in filename:
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

    elif "fairness" in folder_name or "fairness" in filename:
        print("\nEvaluating counterfactual fairness...\n")
        evaluator = CounterfactualFairnessEvaluator()
        fairness_results = evaluator.score(dataset, llm, explain=True)

        for i, result in enumerate(fairness_results):
            print(f"--- Entry {i + 1} ---")
            print(f"Fairness - Final score:      {result['final_score']:.2f}")
            print(f"* LLM score:                 {result['llm_score']:.2f}")
            print(f"* Sentiment score:           {result['sentiment_score']:.2f}")
            print(f"Justification:               {result['justification']}\n")
            results.append(result)

        print("\n--- Fairness Results ---")
        print(f"Average LLM score:           {sum(r['llm_score'] for r in results) / len(results):.2f}")
        print(f"Average sentiment score:     {sum(r['sentiment_score'] for r in results) / len(results):.2f}")
        print(f"Average final score:         {sum(r['final_score'] for r in results) / len(results):.2f}")

    elif "accuracy" in folder_name or "accuracy" in filename:
        print("\nEvaluating accuracy with MENLI...\n")
        accuracy_scores = MENLIAccuracy().score(dataset)

        for i, score in enumerate(accuracy_scores):
            print(f"--- Entry {i + 1} ---")
            print(f"Accuracy - Score: {score['score']:.2f} | Category: {score['category']}")
            if 'justification' in score:
                print(f"Justification: {score['justification']}")
            results.append(score)

        print("\n--- MENLI Accuracy Results ---")
        print(f"Average Accuracy Score:     {sum(r['score'] for r in results) / len(results):.2f}")

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

    # Guardar en JSON si se proporcionó archivo de salida
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Resultados guardados en: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input .json file")
    parser.add_argument("--model", type=str, default="gemma:2b", help="Model to evaluate (e.g. llama3, mistral)")
    parser.add_argument("--output", type=str, help="Path to save the results as .json")
    args = parser.parse_args()

    main(args.input, args.model, args.output)
