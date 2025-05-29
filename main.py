import argparse
from datasets import load_dataset
from evaluator.custom_metrics.composite_factuality import CompositeFactuality
from evaluator.custom_metrics.ethical_evaluator import EthicalEvaluator
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
