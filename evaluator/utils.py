# evaluator/utils.py

import os
from datetime import datetime

def guardar_resultados(df, carpeta="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"resultados_{timestamp}.csv")
    df.to_csv(ruta, index=False)
    print(f"Resultados guardados en: {ruta}")
