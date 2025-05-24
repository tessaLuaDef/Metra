# main.py

import argparse
import yaml
from dotenv import load_dotenv
load_dotenv()
from evaluator.pipeline import EvaluatorPipeline

def cargar_configuracion(path_yaml):
    with open(path_yaml, "r") as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Evaluador de modelos LLM RAG")
    parser.add_argument("--input", type=str, required=True, help="Ruta al dataset JSON o CSV")
    parser.add_argument("--suite", type=str, required=True, help="Ruta al archivo YAML con la suite de evaluación")
    args = parser.parse_args()

    config = cargar_configuracion(args.suite)

    evaluador = EvaluatorPipeline(config)
    evaluador.evaluar(args.input)

if __name__ == "__main__":
    main()
