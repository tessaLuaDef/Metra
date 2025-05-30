from fastapi import FastAPI
from pydantic import BaseModel
from evaluator.custom_metrics.composite_factuality import CompositeFactuality
from evaluator.custom_metrics.ethical_evaluator import EthicalEvaluator
#from evaluator.pipeline import EvaluatorPipeline  # por si se usa después
from langchain_community.chat_models import ChatOllama
import os
from dotenv import load_dotenv

# Cargar entorno y modelo
load_dotenv()
llm = ChatOllama(model="llama3")  # o el modelo que estés usando

app = FastAPI()

# Entrada esperada como JSON
class EvaluationRequest(BaseModel):
    question: str
    answer: str
    context: str
    
@app.post("/evaluate")
def evaluate_input(data: EvaluationRequest):
    try:
        print("📥 Recibida entrada:", data)

        factuality_metric = CompositeFactuality()
        

        item = {
            "question": data.question,
            "answer": data.answer,
            "contexts": [data.context]
        }

        print("📦 Item preparado:", item)

        # Evaluar factualidad (con justificación)
        factuality_score, justifications = factuality_metric.score([item], llm)
        print("📊 Factuality:", factuality_score, "Justificación:", justifications)

       

        return {
            "factuality": factuality_score[0] if factuality_score else None,
            "justification": justifications[0] if justifications else None
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}
