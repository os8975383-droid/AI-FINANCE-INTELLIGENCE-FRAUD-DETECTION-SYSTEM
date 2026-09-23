from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/fraud_detection_model.joblib")


def predict_transaction(transaction):
    model = joblib.load(MODEL_PATH)

    data = pd.DataFrame([transaction])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    print("=" * 60)
    print("FRAUD DETECTION RESULT")
    print("=" * 60)

    if prediction == 1:
        print("🚨 FRAUDULENT TRANSACTION")
    else:
        print("✅ LEGITIMATE TRANSACTION")

    print(f"Fraud Probability: {probability:.2%}")
    print("=" * 60)


transaction = {
    "amount": 25000,
    "hour": 2,
    "day_of_week": 2,
    "customer_age": 25,
    "account_age_days": 500,
    "avg_amount_30d": 3000,
    "transaction_count_24h": 15,
    "transaction_count_7d": 40,
    "distance_from_home": 100,
    "distance_from_usual_location": 80,
    "merchant_risk": 0.7,
    "merchant_category": "Electronics",
    "is_new_device": 1,
    "is_international": 1,
    "is_online": 1,
    "failed_attempts_24h": 3,
    "password_changed_recently": 1,
    "beneficiary_added_recently": 1,
    "amount_deviation": 8.33,
}


if __name__ == "__main__":
    predict_transaction(transaction)