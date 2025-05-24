# evaluator/metrics.py

from ragas.metrics import faithfulness, context_precision
from ragas import evaluate
import pandas as pd

def ejecutar_metricas(dataset, metricas_config):
    # Mapa de nombre a función real
    metricas_disponibles = {
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        # Luego añadiremos: toxicity, bias, etc.
    }

    metricas_a_usar = []
    nombres_validos = []

    for metrica in metricas_config:
        nombre = metrica["name"]
        if nombre in metricas_disponibles:
            metricas_a_usar.append(metricas_disponibles[nombre])
            nombres_validos.append(nombre)
        else:
            print(f"⚠️  Métrica '{nombre}' no está implementada aún.")

    if not metricas_a_usar:
        raise ValueError("No hay métricas válidas en la configuración.")

    print(f"Ejecutando métricas: {nombres_validos}")

    resultados = evaluate(dataset, metrics=metricas_a_usar)

    # Convertimos el resultado a DataFrame para guardarlo más fácilmente
    df_resultados = resultados.to_pandas()
    return df_resultados
