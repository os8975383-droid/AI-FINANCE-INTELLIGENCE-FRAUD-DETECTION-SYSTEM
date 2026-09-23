import requests


API_URL = "http://127.0.0.1:8808/predict"


def sample_transaction():

    return {
        "amount": 5000.0,
        "hour": 23,
        "day_of_week": 0,
        "customer_age": 30,
        "account_age_days": 500,
        "avg_amount_30d": 2000.0,
        "transaction_count_24h": 5,
        "transaction_count_7d": 15.0,
        "distance_from_home": 50.0,
        "distance_from_usual_location": 40.0,
        "merchant_risk": 0.9,
        "merchant_category": "Electronics",
        "is_new_device": 1,
        "is_international": 1,
        "is_online": 1,
        "failed_attempts_24h": 2,
        "password_changed_recently": 1,
        "beneficiary_added_recently": 1,
        "amount_deviation": 3000.0
    }


def test_api_is_running():

    response = requests.get(
        "http://127.0.0.1:8808/docs"
    )

    assert response.status_code == 200


def test_prediction_endpoint():

    response = requests.post(
        API_URL,
        json=sample_transaction()
    )

    assert response.status_code == 200


def test_prediction_value():

    response = requests.post(
        API_URL,
        json=sample_transaction()
    )

    result = response.json()

    assert result["prediction"] in [0, 1]


def test_fraud_probability():

    response = requests.post(
        API_URL,
        json=sample_transaction()
    )

    result = response.json()

    percentage = result["fraud_probability"]

    assert 0 <= percentage <= 1


def test_risk_level():

    response = requests.post(
        API_URL,
        json=sample_transaction()
    )

    result = response.json()

    assert result["risk_level"] in [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]


def test_probability_percentage():

    response = requests.post(
        API_URL,
        json=sample_transaction()
    )

    result = response.json()

    percentage = result["fraud_probability_percentage"]

    assert 0 <= percentage <= 100