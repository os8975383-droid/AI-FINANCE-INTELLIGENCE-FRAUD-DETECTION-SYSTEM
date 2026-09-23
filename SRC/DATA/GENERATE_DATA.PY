from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# AI FINANCE INTELLIGENCE & FRAUD DETECTION
# Improved Synthetic Financial Transaction Dataset Generator
# ============================================================

RANDOM_SEED = 42
NUMBER_OF_TRANSACTIONS = 20000

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# GENERATE TRANSACTIONS
# ============================================================

def generate_transactions(
    n: int = NUMBER_OF_TRANSACTIONS
) -> pd.DataFrame:

    # --------------------------------------------------------
    # 1. Basic transaction information
    # --------------------------------------------------------

    transaction_id = [
        f"TXN{i:07d}"
        for i in range(1, n + 1)
    ]

    amount = np.round(
        rng.lognormal(
            mean=4.2,
            sigma=1.15,
            size=n
        ),
        2
    )

    hour = rng.integers(
        0,
        24,
        size=n
    )

    day_of_week = rng.integers(
        0,
        7,
        size=n
    )


    # --------------------------------------------------------
    # 2. Customer behaviour
    # --------------------------------------------------------

    customer_age = rng.integers(
        18,
        70,
        size=n
    )

    account_age_days = rng.integers(
        30,
        5000,
        size=n
    )

    avg_amount_30d = np.round(
        rng.lognormal(
            mean=4.0,
            sigma=0.75,
            size=n
        ),
        2
    )

    transaction_count_24h = (
        rng.poisson(
            lam=3,
            size=n
        )
    )

    transaction_count_7d = (
        transaction_count_24h
        + rng.poisson(
            lam=12,
            size=n
        )
    )


    # --------------------------------------------------------
    # 3. Location behaviour
    # --------------------------------------------------------

    distance_from_home = np.round(
        rng.gamma(
            shape=2.2,
            scale=18,
            size=n
        ),
        2
    )

    distance_from_usual_location = np.round(
        rng.gamma(
            shape=2.0,
            scale=12,
            size=n
        ),
        2
    )


    # --------------------------------------------------------
    # 4. Merchant information
    # --------------------------------------------------------

    merchant_categories = [
        "Groceries",
        "Food",
        "Transport",
        "Shopping",
        "Bills",
        "Entertainment",
        "Healthcare",
        "Travel",
        "Electronics",
        "Online Services"
    ]

    merchant_category = rng.choice(
        merchant_categories,
        size=n
    )

    merchant_risk = np.round(
        rng.beta(
            a=2,
            b=7,
            size=n
        ),
        4
    )


    # --------------------------------------------------------
    # 5. Device and payment behaviour
    # --------------------------------------------------------

    is_new_device = rng.binomial(
        1,
        0.08,
        size=n
    )

    is_international = rng.binomial(
        1,
        0.12,
        size=n
    )

    is_online = rng.binomial(
        1,
        0.55,
        size=n
    )


    # --------------------------------------------------------
    # 6. Security behaviour
    # --------------------------------------------------------

    failed_attempts_24h = rng.poisson(
        lam=0.35,
        size=n
    )

    password_changed_recently = rng.binomial(
        1,
        0.04,
        size=n
    )

    beneficiary_added_recently = rng.binomial(
        1,
        0.05,
        size=n
    )


    # --------------------------------------------------------
    # 7. Amount deviation
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # This is a ratio, not a difference.
    #
    # amount_deviation = amount / avg_amount_30d
    #
    # Example:
    # amount = 1500
    # average = 2500
    #
    # deviation = 1500 / 2500 = 0.60
    # --------------------------------------------------------

    amount_deviation = np.round(
        amount /
        np.maximum(
            avg_amount_30d,
            1
        ),
        3
    )


    # ========================================================
    # 8. INDIVIDUAL RISK FEATURES
    # ========================================================

    unusual_hour = (
        (hour <= 4) |
        (hour >= 23)
    ).astype(int)

    high_amount_ratio = (
        amount_deviation >= 4
    ).astype(int)

    extreme_amount_ratio = (
        amount_deviation >= 8
    ).astype(int)

    high_transaction_frequency = (
        transaction_count_24h >= 12
    ).astype(int)

    unusual_location = (
        distance_from_usual_location >= 60
    ).astype(int)

    high_failed_attempts = (
        failed_attempts_24h >= 3
    ).astype(int)

    new_account = (
        account_age_days <= 90
    ).astype(int)


    # ========================================================
    # 9. FRAUD RISK SCORE
    # ========================================================
    #
    # IMPORTANT:
    # The score is intentionally moderate.
    # We don't want almost every unusual transaction
    # to become 100% fraud.
    # ========================================================

    risk_score = (

        -5.0

        # Transaction amount
        + 0.000025 * amount

        # Merchant behaviour
        + 1.20 * merchant_risk

        # Location
        + 0.012 * distance_from_home
        + 0.018 * distance_from_usual_location

        # International transaction
        + 0.70 * is_international

        # New device
        + 0.85 * is_new_device

        # Online transaction
        + 0.25 * is_online

        # Security
        + 0.25 * failed_attempts_24h
        + 0.60 * password_changed_recently
        + 0.65 * beneficiary_added_recently

        # Time
        + 0.55 * unusual_hour

        # Transaction amount anomaly
        + 0.90 * high_amount_ratio
        + 0.75 * extreme_amount_ratio

        # Frequency
        + 0.70 * high_transaction_frequency

        # Location anomaly
        + 0.50 * unusual_location

        # New account
        + 0.45 * new_account

        # Amount difference effect
        + 0.000025 *
        np.maximum(
            amount - avg_amount_30d,
            0
        )
    )


    # ========================================================
    # 10. CONVERT RISK SCORE → FRAUD PROBABILITY
    # ========================================================

    fraud_probability = (
        1 /
        (
            1 +
            np.exp(
                -risk_score
            )
        )
    )


    # Keep probability away from absolute 0 and 1

    fraud_probability = np.clip(
        fraud_probability,
        0.01,
        0.85
    )


    # ========================================================
    # 11. GENERATE FRAUD LABEL
    # ========================================================

    is_fraud = rng.binomial(
        1,
        fraud_probability,
        size=n
    )


    # ========================================================
    # 12. CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame({

        "transaction_id":
            transaction_id,

        "amount":
            amount,

        "hour":
            hour,

        "day_of_week":
            day_of_week,

        "customer_age":
            customer_age,

        "account_age_days":
            account_age_days,

        "avg_amount_30d":
            avg_amount_30d,

        "transaction_count_24h":
            transaction_count_24h,

        "transaction_count_7d":
            transaction_count_7d,

        "distance_from_home":
            distance_from_home,

        "distance_from_usual_location":
            distance_from_usual_location,

        "merchant_risk":
            merchant_risk,

        "merchant_category":
            merchant_category,

        "is_new_device":
            is_new_device,

        "is_international":
            is_international,

        "is_online":
            is_online,

        "failed_attempts_24h":
            failed_attempts_24h,

        "password_changed_recently":
            password_changed_recently,

        "beneficiary_added_recently":
            beneficiary_added_recently,

        "amount_deviation":
            amount_deviation,

        "is_fraud":
            is_fraud
    })


    return df


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    df: pd.DataFrame
) -> None:

    # Project root se DATA folder
    project_root = Path(__file__).resolve().parents[2]

    output_directory = (
        project_root / "DATA"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory /
        "transactions.csv"
    )


    df.to_csv(
        output_file,
        index=False
    )


    print("=" * 70)
    print(
        "AI FINANCE INTELLIGENCE"
    )
    print(
        "DATASET CREATED SUCCESSFULLY"
    )
    print("=" * 70)

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Columns    : {len(df.columns)}"
    )

    print(
        f"File       : {output_file}"
    )


    print("\nFraud Distribution:")

    print(
        df["is_fraud"]
        .value_counts()
        .sort_index()
    )


    print("\nFraud Percentage:")

    print(
        (
            df["is_fraud"]
            .value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        ).round(2)
    )


    print("\nAverage Transaction Amount:")

    print(
        round(
            df["amount"].mean(),
            2
        )
    )


    print("\nAverage Amount Deviation:")

    print(
        round(
            df["amount_deviation"].mean(),
            3
        )
    )


    print("\nDataset Preview:")

    print(
        df.head()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = generate_transactions()

    save_dataset(df)


if __name__ == "__main__":

    main()