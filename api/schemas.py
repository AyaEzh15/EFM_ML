from pydantic import BaseModel
from typing import Dict, Any


class NetworkInput(BaseModel):
    features: Dict[str, Any]


class PredictionOutput(BaseModel):
    prediction: int
    label: str
    interpretation: str