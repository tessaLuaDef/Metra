# evaluator/pipeline.py

import pandas as pd
from evaluator.metrics import ejecutar_metricas
from evaluator.utils import guardar_resultados
from datasets import Dataset

class EvaluatorPipeline:
    def __init__(self, config):
        self.config = config
        self.metricas = config["metrics"]

    def cargar_dataset(self, path):
        if path.endswith(".json"):
            df = pd.read_json(path)
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError("Formato de dataset no soportado. Usa .json o .csv")

        # Asegura que están las columnas requeridas
        columnas_requeridas = {"question", "contexts", "answer"}
        if not columnas_requeridas.issubset(set(df.columns)):
            raise ValueError(f"El dataset debe incluir las columnas: {columnas_requeridas}")
        
        return Dataset.from_pandas(df)

    def evaluar(self, ruta_dataset):
        dataset = self.cargar_dataset(ruta_dataset)

        print(f"Evaluando dataset con {len(dataset)} entradas...")
        resultados = ejecutar_metricas(dataset, self.metricas)

        guardar_resultados(resultados)
        print("Evaluación completada y resultados guardados.")
