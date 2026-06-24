import os
import sys

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.schemas import NetworkInput, PredictionOutput
from src.predict import predict_intrusion


app = FastAPI(
    title="IDS MLOps Cybersecurity API",
    description="API REST pour la détection d'intrusions réseau avec Machine Learning",
    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)


@app.get("/")
def home():
    return {
        "message": "API IDS MLOps fonctionne correctement",
        "status": "OK"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "Random Forest",
        "task": "Network Intrusion Detection"
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: NetworkInput):
    result = predict_intrusion(data.features)
    return result