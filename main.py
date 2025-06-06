import argparse
from datasets import load_dataset
from evaluator.custom_metrics.composite_factuality import CompositeFactuality
from evaluator.custom_metrics.ethical_evaluator import EthicalEvaluator
from evaluator.custom_metrics.counterfactual_fairness import CounterfactualFairnessEvaluator
#from evaluator.custom_metrics.menli_precision import MENLIPrecision
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

# Cargar variables de entorno
load_dotenv()

# Inicializar modelo
llm = ChatOllama(model="mistral")

def main(input_file: str):
    dataset = load_dataset("json", data_files=input_file)["train"]

    if "ethic" in input_file.lower():
        print("\n Evaluando dimensión ética...")
        ethical_scores, ethical_justifications = EthicalEvaluator().score(dataset, llm)
        for i, (score, explanation) in enumerate(zip(ethical_scores, ethical_justifications)):
            print(f"\n🔹 Entrada {i + 1}")
            print(f"Ética - Score: {score:.2f}")
            print(f"Ética - Justificación: {explanation}")

    elif "fairness" in input_file.lower():
        print("\n Evaluando equidad contrafactual...")
        fairness_scores, fairness_justifications = CounterfactualFairnessEvaluator().score(dataset, llm)
        for i, (score, explanation) in enumerate(zip(fairness_scores, fairness_justifications)):
            print(f"\n🔹 Entrada {i + 1}")
            print(f"Equidad - Score: {score:.2f}")
            print(f"Equidad - Justificación: {explanation}")
            
   # elif "precision" in input_file.lower():
    #    print("\n Evaluando precisión con MENLI...")
     #   precision_scores = MENLIPrecision().score(dataset)
      #  for i, score in enumerate(precision_scores):
       #     print(f"\n🔹 Entrada {i + 1}")
        #    print(f"Precisión - Score: {score:.2f}")

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
