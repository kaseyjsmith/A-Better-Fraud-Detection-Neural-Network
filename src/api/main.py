from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from .service.model import PredicitonModel
from .schemas.transaction import Transaction
from .schemas.feature_names import FEATURE_NAMES
import joblib
from pathlib import Path
import numpy as np
import torch

# Load scaler once at module level
proj_root = "/home/ksmith/birds/neural_networks/fraud_detection"
scaler = joblib.load(f"{proj_root}/src/models/scalers/scaler.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.fraud_service = PredicitonModel().model
    yield


app = FastAPI(title="Fraud Detection API", version="0.0.1", lifespan=lifespan)


@app.post("/predict")
async def predict(transaction: Transaction, request: Request):
    # convert input to array and scale
    values = [getattr(transaction, name) for name in FEATURE_NAMES]
    raw_array = np.array([values])
    scaled_array = scaler.transform(raw_array)

    # convert to tensor (model expects this)
    input_tensor = torch.FloatTensor(scaled_array)

    # model to eval mode
    model = request.app.state.fraud_service
    model.eval()

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)

        probability = torch.sigmoid(logits).item()

        is_fraud = probability > model.threshold

    return {
        "is_fraud": bool(is_fraud),
        "fraud_probability": float(probability),
        "threshold": float(model.threshold),
    }
