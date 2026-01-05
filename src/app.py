"""
FastAPI service for Marketing Customer Churn Prediction.

Loads:
- models/final_churn_pipeline.joblib
- models/threshold.json

Endpoints:
- GET  /                 -> health check
- POST /predict          -> single customer churn score
- POST /predict-batch    -> batch scoring for campaigns

Run:
    uvicorn src.app:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json
from typing import List


# -----------------------
# Load model + threshold
# -----------------------

MODEL_PATH = "models/final_churn_pipeline.joblib"
THRESHOLD_PATH = "models/threshold.json"

pipeline = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold_config = json.load(f)

THRESHOLD = threshold_config["threshold"]
TARGET_FRACTION = threshold_config["target_fraction"]


# -----------------------
# FastAPI app
# -----------------------

app = FastAPI(
    title="Marketing Customer Churn API",
    description="Predict customer churn probability for targeted retention campaigns.",
    version="1.0"
)


# -----------------------
# Request schemas
# -----------------------

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class CustomerBatch(BaseModel):
    customers: List[Customer]


# -----------------------
# Endpoints
# -----------------------

@app.get("/")
def health_check():
    return {
        "status": "running",
        "model": "RandomForest",
        "threshold": THRESHOLD,
        "target_fraction": TARGET_FRACTION
    }


@app.post("/predict")
def predict_churn(customer: Customer):
    df = pd.DataFrame([customer.dict()])

    proba = pipeline.predict_proba(df)[0][1]
    churn_flag = int(proba >= THRESHOLD)

    return {
        "churn_probability": float(proba),
        "churn_prediction": churn_flag,
        "threshold_used": THRESHOLD,
        "recommendation": (
            "Target for retention campaign"
            if churn_flag == 1
            else "No action needed"
        )
    }


@app.post("/predict-batch")
def predict_churn_batch(batch: CustomerBatch):
    df = pd.DataFrame([c.dict() for c in batch.customers])

    probabilities = pipeline.predict_proba(df)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    results = []
    for i, row in df.iterrows():
        results.append({
            "customer_index": i,
            "churn_probability": float(probabilities[i]),
            "churn_prediction": int(predictions[i]),
            "recommendation": (
                "Target for retention campaign"
                if predictions[i] == 1
                else "No action needed"
            )
        })

    return {
        "threshold_used": THRESHOLD,
        "customers_scored": len(results),
        "results": results
    }
