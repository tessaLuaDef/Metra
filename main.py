import argparse
from datasets import load_dataset
from evaluator.custom_metrics.composite_factuality import CompositeFactuality
from evaluator.custom_metrics.counterfactual_fairness import CounterfactualFairnessEvaluator
from evaluator.custom_metrics.ethical_alignment import evaluate_ethics  # NUEVO IMPORT
#from evaluator.custom_metrics.menli_precision import MENLIPrecision
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

# Cargar variables de entorno
load_dotenv()

# Inicializar modelo SOLO si es necesario
llm = ChatOllama(model="gemma:2b")

def main(input_file: str):
    dataset_hf = load_dataset("json", data_files=input_file)["train"]

    # Convertimos el dataset Hugging Face a lista de diccionarios (compatible con EthicalAlignment)
    dataset = [dict(row) for row in dataset_hf]

    if "ethic" in input_file.lower():
        print("\n Evaluando dimensión ética...")
        ethics_result = evaluate_ethics(dataset)
        print("\n🔹 Resultados EthicalAlignment:")
        print(f"- Benchmark (ETHICS): {ethics_result['benchmark_score']:.2f}")
        print(f"- Toxicidad inversa:  {ethics_result['avg_toxicity_score']:.2f}")
        print(f"- Score final:        {ethics_result['final_score']:.2f}")

    elif "fairness" in input_file.lower():
        print("\n Evaluando equidad contrafactual...")
        evaluator = CounterfactualFairnessEvaluator()
        fairness_results = evaluator.score(dataset, llm, explain=True)

        for i, result in enumerate(fairness_results):
            print(f"\n🔹 Entrada {i + 1}")
            print(f"Equidad - Score final:       {result['final_score']:.2f}")
            print(f"🔸 Score LLM:                {result['llm_score']:.2f}")
            print(f"🔸 Score sentimiento:        {result['sentiment_score']:.2f}")
            print(f"Justificación del LLM:       {result['justification']}")
          
    # elif "precision" in input_file.lower():
    #     print("\n Evaluando precisión con MENLI...")
    #     precision_scores = MENLIPrecision().score(dataset)
    #     for i, score in enumerate(precision_scores):
    #         print(f"\n🔹 Entrada {i + 1}")
    #         print(f"Precisión - Score: {score:.2f}")

    else:
        print("\n Evaluando factualidad compuesta...")
        scores, justifications = CompositeFactuality().score(dataset, llm)
        for i, (score, explanation) in enumerate(zip(scores, justifications)):
            print(f"\n🔹 Entrada {i + 1}")
            print(f"Score: {score:.2f}")
            print(f"Justificación: {explanation}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Ruta al archivo .json de entrada")
    args = parser.parse_args()
    main(args.input)
