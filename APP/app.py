import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from pathlib import Path


# ============================================================
# AI FINANCE INTELLIGENCE & FRAUD DETECTION SYSTEM
# Streamlit Dashboard
# ============================================================


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="AI Finance Intelligence",
    page_icon="💳",
    layout="wide"
)


# ------------------------------------------------------------
# 2. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "DATA" / "transactions.csv"

CONFUSION_MATRIX_PATH = (
    BASE_DIR / "MODELS" / "confusion_matrix.png"
)

ROC_CURVE_PATH = (
    BASE_DIR / "MODELS" / "roc_curve.png"
)

FEATURE_IMPORTANCE_PATH = (
    BASE_DIR / "MODELS" / "feature_importance.png"
)


# ------------------------------------------------------------
# 3. DOCKER API URL
# ------------------------------------------------------------

API_URL = "http://api:8000/predict"


# ------------------------------------------------------------
# 4. LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


try:

    df = load_data()

except Exception as e:

    st.error(
        f"Unable to load transaction dataset.\n\n{e}"
    )

    st.stop()


# ------------------------------------------------------------
# 5. SIDEBAR
# ------------------------------------------------------------

st.sidebar.title("💳 AI Finance Intelligence")

st.sidebar.markdown(
    "### Fraud Detection System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Fraud Analytics",
        "🔍 Transaction Risk Checker"
    ]
)


# ============================================================
# PAGE 1 — FRAUD ANALYTICS
# ============================================================

if page == "📊 Fraud Analytics":

    st.title("💳 AI Finance Intelligence")
    st.subheader("Fraud Analytics Dashboard")

    st.markdown(
        """
        This dashboard provides transaction-level analytics
        and machine-learning-based fraud insights.
        """
    )

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_transactions = len(df)

    fraud_transactions = int(
        df["is_fraud"].sum()
    )

    fraud_rate = (
        fraud_transactions /
        total_transactions *
        100
    )

    average_transaction = df["amount"].mean()

    total_transaction_value = df["amount"].sum()

    fraud_transaction_value = df.loc[
        df["is_fraud"] == 1,
        "amount"
    ].sum()


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:

        st.metric(
            "Fraud Transactions",
            f"{fraud_transactions:,}"
        )

    with col3:

        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )


    col4, col5, col6 = st.columns(3)

    with col4:

        st.metric(
            "Average Transaction",
            f"₹{average_transaction:,.2f}"
        )

    with col5:

        st.metric(
            "Total Transaction Value",
            f"₹{total_transaction_value:,.2f}"
        )

    with col6:

        st.metric(
            "Fraud Transaction Value",
            f"₹{fraud_transaction_value:,.2f}"
        )


    st.divider()


    # --------------------------------------------------------
    # FRAUD VS LEGITIMATE
    # --------------------------------------------------------

    st.subheader("Fraud vs Legitimate Transactions")

    fraud_counts = (
        df["is_fraud"]
        .value_counts()
        .rename(
            index={
                0: "Legitimate",
                1: "Fraud"
            }
        )
        .reset_index()
    )

    fraud_counts.columns = [
        "Transaction Type",
        "Count"
    ]

    fig_fraud = px.pie(
        fraud_counts,
        names="Transaction Type",
        values="Count",
        hole=0.45,
        title="Transaction Distribution"
    )

    st.plotly_chart(
        fig_fraud,
        use_container_width=True
    )


    # --------------------------------------------------------
    # FRAUD BY MERCHANT CATEGORY
    # --------------------------------------------------------

    st.subheader("Fraud by Merchant Category")

    fraud_category = (
        df.groupby("merchant_category")["is_fraud"]
        .sum()
        .reset_index()
        .sort_values(
            "is_fraud",
            ascending=False
        )
    )

    fig_category = px.bar(
        fraud_category,
        x="merchant_category",
        y="is_fraud",
        title="Fraud Transactions by Merchant Category",
        labels={
            "merchant_category": "Merchant Category",
            "is_fraud": "Fraud Transactions"
        }
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )


    # --------------------------------------------------------
    # FRAUD BY HOUR
    # --------------------------------------------------------

    st.subheader("Fraud Transactions by Hour")

    fraud_hour = (
        df.groupby("hour")["is_fraud"]
        .sum()
        .reset_index()
    )

    fig_hour = px.line(
        fraud_hour,
        x="hour",
        y="is_fraud",
        markers=True,
        title="Fraud Activity by Hour",
        labels={
            "hour": "Hour",
            "is_fraud": "Fraud Transactions"
        }
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )


    # --------------------------------------------------------
    # INTERNATIONAL VS DOMESTIC
    # --------------------------------------------------------

    st.subheader("International vs Domestic Transactions")

    international_data = (
        df.groupby("is_international")["is_fraud"]
        .sum()
        .reset_index()
    )

    international_data["Type"] = (
        international_data["is_international"]
        .map({
            0: "Domestic",
            1: "International"
        })
    )

    fig_international = px.bar(
        international_data,
        x="Type",
        y="is_fraud",
        title="Fraud by Transaction Location",
        labels={
            "Type": "Transaction Type",
            "is_fraud": "Fraud Transactions"
        }
    )

    st.plotly_chart(
        fig_international,
        use_container_width=True
    )


    # --------------------------------------------------------
    # NEW DEVICE
    # --------------------------------------------------------

    st.subheader("Fraud by Device Status")

    device_data = (
        df.groupby("is_new_device")["is_fraud"]
        .sum()
        .reset_index()
    )

    device_data["Device Status"] = (
        device_data["is_new_device"]
        .map({
            0: "Existing Device",
            1: "New Device"
        })
    )

    fig_device = px.bar(
        device_data,
        x="Device Status",
        y="is_fraud",
        title="Fraud by Device Status",
        labels={
            "Device Status": "Device",
            "is_fraud": "Fraud Transactions"
        }
    )

    st.plotly_chart(
        fig_device,
        use_container_width=True
    )


    st.divider()


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("🤖 Machine Learning Model Performance")

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)

    with metric1:

        st.metric(
            "Accuracy",
            "75.75%"
        )

    with metric2:

        st.metric(
            "Precision",
            "46.03%"
        )

    with metric3:

        st.metric(
            "Recall",
            "71.96%"
        )

    with metric4:

        st.metric(
            "F1 Score",
            "56.15%"
        )

    with metric5:

        st.metric(
            "ROC-AUC",
            "82.01%"
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    if CONFUSION_MATRIX_PATH.exists():

        st.image(
            str(CONFUSION_MATRIX_PATH),
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix image not found."
        )


    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.subheader("ROC Curve")

    if ROC_CURVE_PATH.exists():

        st.image(
            str(ROC_CURVE_PATH),
            use_container_width=True
        )

    else:

        st.warning(
            "ROC curve image not found."
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("Feature Importance")

    if FEATURE_IMPORTANCE_PATH.exists():

        st.image(
            str(FEATURE_IMPORTANCE_PATH),
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance image not found."
        )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader("📄 Transaction Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )


    st.info(
        "⚠️ This project uses synthetic transaction data "
        "for educational and portfolio purposes. "
        "It is not a production banking fraud detection system."
    )


# ============================================================
# PAGE 2 — TRANSACTION RISK CHECKER
# ============================================================

else:

    st.title("🔍 Transaction Risk Checker")

    st.markdown(
        """
        Enter transaction and customer behaviour details.
        The ML model will estimate the probability of fraud.
        """
    )


    # --------------------------------------------------------
    # TRANSACTION INFORMATION
    # --------------------------------------------------------

    st.subheader("💰 Transaction Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1500.0,
            step=100.0
        )

    with col2:

        hour = st.number_input(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=14,
            step=1
        )

    with col3:

        day_of_week = st.number_input(
            "Day of Week",
            min_value=0,
            max_value=6,
            value=2,
            step=1,
            help="0 = Monday, 6 = Sunday"
        )


    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        customer_age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=35
        )

    with col2:

        account_age_days = st.number_input(
            "Account Age (Days)",
            min_value=1,
            value=900
        )

    with col3:

        avg_amount_30d = st.number_input(
            "Average Transaction Amount (30 Days)",
            min_value=0.01,
            value=2500.0,
            step=100.0
        )


    # --------------------------------------------------------
    # TRANSACTION FREQUENCY
    # --------------------------------------------------------

    st.subheader("📈 Transaction Frequency")

    col1, col2 = st.columns(2)

    with col1:

        transaction_count_24h = st.number_input(
            "Transactions in Last 24 Hours",
            min_value=0,
            value=2
        )

    with col2:

        transaction_count_7d = st.number_input(
            "Transactions in Last 7 Days",
            min_value=0.0,
            value=8.0
        )


    # --------------------------------------------------------
    # LOCATION INFORMATION
    # --------------------------------------------------------

    st.subheader("📍 Location Behaviour")

    col1, col2 = st.columns(2)

    with col1:

        distance_from_home = st.number_input(
            "Distance From Home",
            min_value=0.0,
            value=3.0,
            step=1.0
        )

    with col2:

        distance_from_usual_location = st.number_input(
            "Distance From Usual Location",
            min_value=0.0,
            value=2.0,
            step=1.0
        )


    # --------------------------------------------------------
    # MERCHANT INFORMATION
    # --------------------------------------------------------

    st.subheader("🏪 Merchant Information")

    col1, col2 = st.columns(2)

    with col1:

        merchant_risk = st.number_input(
            "Merchant Risk",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01
        )

    with col2:

        merchant_category = st.selectbox(
            "Merchant Category",
            [
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
        )


    # --------------------------------------------------------
    # DEVICE & PAYMENT
    # --------------------------------------------------------

    st.subheader("📱 Device & Payment Behaviour")

    col1, col2, col3 = st.columns(3)

    with col1:

        is_new_device = int(
            st.checkbox(
                "New Device"
            )
        )

    with col2:

        is_international = int(
            st.checkbox(
                "International Transaction"
            )
        )

    with col3:

        is_online = int(
            st.checkbox(
                "Online Transaction",
                value=True
            )
        )


    # --------------------------------------------------------
    # SECURITY INFORMATION
    # --------------------------------------------------------

    st.subheader("🔐 Security Behaviour")

    col1, col2, col3 = st.columns(3)

    with col1:

        failed_attempts_24h = st.number_input(
            "Failed Attempts (24h)",
            min_value=0,
            value=0
        )

    with col2:

        password_changed_recently = int(
            st.checkbox(
                "Password Changed Recently"
            )
        )

    with col3:

        beneficiary_added_recently = int(
            st.checkbox(
                "New Beneficiary Added Recently"
            )
        )


    # ========================================================
    # AUTOMATIC AMOUNT DEVIATION
    # ========================================================

    st.subheader("📊 Automatically Calculated Feature")

    # IMPORTANT:
    # The training dataset defines amount_deviation as:
    #
    # amount / avg_amount_30d
    #
    # Therefore the user should NOT manually enter it.

    amount_deviation = (
        amount /
        max(avg_amount_30d, 1)
    )

    st.metric(
        "Amount Deviation Ratio",
        f"{amount_deviation:.3f}"
    )

    st.caption(
        "Calculated automatically as Transaction Amount ÷ "
        "Average Transaction Amount (30 Days)."
    )


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    st.divider()

    check_risk = st.button(
        "🚨 Check Transaction Risk",
        use_container_width=True
    )


    # ========================================================
    # API REQUEST
    # ========================================================

    if check_risk:

        transaction_data = {

            "amount":
                float(amount),

            "hour":
                int(hour),

            "day_of_week":
                int(day_of_week),

            "customer_age":
                int(customer_age),

            "account_age_days":
                int(account_age_days),

            "avg_amount_30d":
                float(avg_amount_30d),

            "transaction_count_24h":
                int(transaction_count_24h),

            "transaction_count_7d":
                float(transaction_count_7d),

            "distance_from_home":
                float(distance_from_home),

            "distance_from_usual_location":
                float(distance_from_usual_location),

            "merchant_risk":
                float(merchant_risk),

            "merchant_category":
                merchant_category,

            "is_new_device":
                int(is_new_device),

            "is_international":
                int(is_international),

            "is_online":
                int(is_online),

            "failed_attempts_24h":
                int(failed_attempts_24h),

            "password_changed_recently":
                int(password_changed_recently),

            "beneficiary_added_recently":
                int(beneficiary_added_recently),

            "amount_deviation":
                float(amount_deviation)
        }


        with st.spinner(
            "Analyzing transaction..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json=transaction_data,
                    timeout=30
                )


                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                if response.status_code == 200:

                    result = response.json()

                    prediction = result.get(
                        "prediction"
                    )

                    probability = result.get(
                        "fraud_probability_percentage"
                    )

                    risk_level = result.get(
                        "risk_level"
                    )


                    st.divider()

                    st.subheader(
                        "🧠 AI Fraud Detection Result"
                    )


                    # --------------------------------------------
                    # RESULT CARDS
                    # --------------------------------------------

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Fraud Probability",
                            f"{probability:.2f}%"
                        )

                    with col2:

                        st.metric(
                            "Prediction",
                            "FRAUD"
                            if prediction == 1
                            else "LEGITIMATE"
                        )

                    with col3:

                        st.metric(
                            "Risk Level",
                            risk_level
                        )


                    # --------------------------------------------
                    # RISK MESSAGE
                    # --------------------------------------------

                    if risk_level == "HIGH":

                        st.error(
                            "🚨 HIGH FRAUD RISK — "
                            "This transaction has strong "
                            "fraud indicators."
                        )

                    elif risk_level == "MEDIUM":

                        st.warning(
                            "⚠️ MEDIUM FRAUD RISK — "
                            "Additional verification may "
                            "be required."
                        )

                    else:

                        st.success(
                            "✅ LOW FRAUD RISK — "
                            "The transaction appears "
                            "relatively normal."
                        )


                    # --------------------------------------------
                    # DEBUG INFORMATION
                    # --------------------------------------------

                    with st.expander(
                        "View API Response"
                    ):

                        st.json(result)


                # ------------------------------------------------
                # API ERROR
                # ------------------------------------------------

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

                    try:

                        st.json(
                            response.json()
                        )

                    except Exception:

                        st.code(
                            response.text
                        )


            # ----------------------------------------------------
            # CONNECTION ERROR
            # ----------------------------------------------------

            except requests.exceptions.ConnectionError:

                st.error(
                    """
                    ❌ Could not connect to the FastAPI server.

                    Make sure the Docker API container is running.
                    """
                )


            except requests.exceptions.Timeout:

                st.error(
                    "❌ API request timed out."
                )


            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "AI Finance Intelligence & Fraud Detection System"
)

st.sidebar.caption(
    "Educational / Portfolio Project"
)
