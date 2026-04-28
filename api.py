"""
FastAPI backend serving predictions for all 4 ML projects.
"""
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="ML Projects API", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_model(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail=f"Model {name} not found. Run train_models.py first.")
    return joblib.load(path)


# ── Request schemas ──────────────────────────────────

class CreditRiskRequest(BaseModel):
    person_age: float
    person_income: float
    person_home_ownership: str  # RENT, OWN, MORTGAGE, OTHER
    person_emp_length: float
    loan_intent: str  # EDUCATION, MEDICAL, VENTURE, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT
    loan_grade: str   # A-G
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str  # Y / N
    cb_person_cred_hist_length: float


class FraudDetectionRequest(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float; Amount: float


class LoanPredictionRequest(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: float
    YearsExperience: int
    Gender: str          # Male / Female
    Education: str       # High School, PhD, Bachelor's, Master's, Unknown
    City: str            # Houston, San Francisco, New York, Chicago, etc.
    EmploymentType: str  # Unemployed, Self-Employed, Employed, etc.


class SentimentRequest(BaseModel):
    text: str


# ── Endpoints ────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "ML Projects API is running", "endpoints": [
        "/predict/credit-risk",
        "/predict/fraud-detection",
        "/predict/loan-prediction",
        "/predict/sentiment-analysis",
    ]}


@app.post("/predict/credit-risk")
def predict_credit_risk(req: CreditRiskRequest):
    artifact = load_model("credit_risk_model.pkl")
    model = artifact['model']
    encoders = artifact['label_encoders']
    features = artifact['feature_names']

    data = req.model_dump()
    # Encode categoricals
    for col, le in encoders.items():
        if col in data:
            try:
                data[col] = le.transform([data[col]])[0]
            except ValueError:
                raise HTTPException(400, f"Invalid value for {col}: {data[col]}. Expected one of {list(le.classes_)}")

    df = pd.DataFrame([data])
    df = df[features]  # ensure column order

    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {"prediction": pred, "default_probability": round(prob, 4),
            "label": "Default" if pred == 1 else "No Default"}


@app.post("/predict/fraud-detection")
def predict_fraud(req: FraudDetectionRequest):
    artifact = load_model("fraud_detection_model.pkl")
    model = artifact['model']
    features = artifact['feature_names']

    df = pd.DataFrame([req.model_dump()])
    df = df[features]

    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {"prediction": pred, "fraud_probability": round(prob, 4),
            "label": "Fraud" if pred == 1 else "Legitimate"}


@app.post("/predict/loan-prediction")
def predict_loan(req: LoanPredictionRequest):
    artifact = load_model("loan_prediction_model.pkl")
    model = artifact['model']
    dummy_columns = artifact['dummy_columns']

    data = req.model_dump()
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, drop_first=True)

    # Align columns with training data
    for col in dummy_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[dummy_columns]

    # Ensure all boolean columns become int
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {"prediction": pred, "approval_probability": round(prob, 4),
            "label": "Approved" if pred == 1 else "Rejected"}


@app.post("/predict/sentiment-analysis")
def predict_sentiment(req: SentimentRequest):
    artifact = load_model("sentiment_analysis_model.pkl")
    model = artifact['model']
    tfidf = artifact['tfidf']
    le = artifact['label_encoder']

    text_tfidf = tfidf.transform([req.text])
    pred = int(model.predict(text_tfidf)[0])
    probs = model.predict_proba(text_tfidf)[0]
    label = le.inverse_transform([pred])[0]

    return {"prediction": pred, "label": label,
            "probabilities": {le.inverse_transform([i])[0]: round(float(p), 4)
                              for i, p in enumerate(probs)}}
