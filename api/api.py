
from fastapi import FastAPI, UploadFile, File
from evaluator.pipeline import EvaluatorPipeline
import yaml
import tempfile
import os

app = FastAPI()

@app.post("/evaluar/")
async def evaluar_modelo(suite: UploadFile = File(...), datos: UploadFile = File(...)):
    # Guardar archivos temporales
    with tempfile.NamedTemporaryFile(delete=False) as temp_suite:
        suite_content = await suite.read()
        temp_suite.write(suite_content)
        suite_path = temp_suite.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_data:
        data_content = await datos.read()
        temp_data.write(data_content)
        data_path = temp_data.name

    # Cargar configuración y ejecutar evaluación
    with open(suite_path, 'r') as f:
        config = yaml.safe_load(f)

    evaluador = EvaluatorPipeline(config)
    evaluador.evaluar(data_path)

    # Limpieza
    os.unlink(suite_path)
    os.unlink(data_path)

    return {"message": "Evaluación completada correctamente"}
