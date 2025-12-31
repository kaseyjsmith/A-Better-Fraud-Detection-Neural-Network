from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from .service.model import PredicitonModel
from .schemas.transaction import Transaction
import joblib
from pathlib import Path

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
    import numpy as np
    import torch

    # Feature names in correct order
    feature_names = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
    ]

    # Extract values in the correct order
    values = [getattr(transaction, name) for name in feature_names]

    # Shape: (1, 30) - one transaction, 30 features
    raw_array = np.array([values])

    # Apply scaling using the module-level scaler
    scaled_array = scaler.transform(raw_array)

    # Convert to tensor
    input_tensor = torch.FloatTensor(scaled_array)

    # Get model from app state
    model = request.app.state.fraud_service

    # Set model to eval mode (disables dropout)
    model.eval()

    # Disable gradient computation (saves memory)
    with torch.no_grad():
        # Forward pass - get logits
        logits = model(input_tensor)

        # Apply sigmoid to get probability
        probability = torch.sigmoid(logits).item()

        # Apply threshold
        is_fraud = probability > model.threshold

    return {
        "is_fraud": bool(is_fraud),
        "fraud_probability": float(probability),
        "threshold": float(model.threshold),
    }
