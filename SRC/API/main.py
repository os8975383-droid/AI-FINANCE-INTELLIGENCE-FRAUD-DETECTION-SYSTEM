from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path


app = FastAPI(
    title="AI Finance Intelligence & Fraud Detection API",
    description="Machine Learning based fraud detection API",
    version="1.0"
)


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "MODELS" / "fraud_detection_model.joblib"


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(MODEL_PATH)


# ==========================================
# INPUT SCHEMA
# EXACTLY SAME AS MODEL FEATURES
# ==========================================

class Transaction(BaseModel):

    amount: float
    hour: int
    day_of_week: int
    customer_age: int
    account_age_days: int

    avg_amount_30d: float

    transaction_count_24h: int
    transaction_count_7d: float

    distance_from_home: float
    distance_from_usual_location: float

    merchant_risk: float
    merchant_category: str

    is_new_device: float
    is_international: int
    is_online: int

    failed_attempts_24h: int
    password_changed_recently: int
    beneficiary_added_recently: int

    amount_deviation: float


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "AI Finance Fraud Detection API is running",
        "docs": "/docs"
    }


# ==========================================
# PREDICT
# ==========================================

@app.post("/predict")
def predict(data: Transaction):

    # Convert input into dictionary
    input_data = data.model_dump()

    # Create DataFrame
    input_df = pd.DataFrame([input_data])


    # EXACT MODEL FEATURE ORDER
    feature_order = [
        "amount",
        "hour",
        "day_of_week",
        "customer_age",
        "account_age_days",
        "avg_amount_30d",
        "transaction_count_24h",
        "transaction_count_7d",
        "distance_from_home",
        "distance_from_usual_location",
        "merchant_risk",
        "merchant_category",
        "is_new_device",
        "is_international",
        "is_online",
        "failed_attempts_24h",
        "password_changed_recently",
        "beneficiary_added_recently",
        "amount_deviation"
    ]


    # Arrange columns exactly like training
    input_df = input_df[feature_order]


    # Prediction
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]


    # Risk level
    if probability >= 0.75:

        risk = "HIGH"

    elif probability >= 0.40:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    # Response
    return {

        "prediction": int(prediction),

        "fraud_probability": round(
            float(probability),
            4
        ),

        "fraud_probability_percentage": round(
            float(probability) * 100,
            2
        ),

        "risk_level": risk
    }