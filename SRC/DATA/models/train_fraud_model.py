from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = Path("data/transactions.csv")
MODEL_PATH = Path("models/fraud_detection_model.joblib")

TARGET = "is_fraud"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("DATA LOADED")
    print("=" * 60)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    # Transaction ID is only an identifier.
    # It should NOT be given to the ML model.
    X = df.drop(
        columns=[
            TARGET,
            "transaction_id"
        ]
    )

    y = df[TARGET]

    return X, y


# ============================================================
# PREPROCESSING
# ============================================================

def create_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(X_train, y_train, preprocessor):

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    print("\nTraining Logistic Regression...")

    logistic_model.fit(
        X_train,
        y_train
    )

    print("Logistic Regression trained.")

    print("\nTraining Random Forest...")

    random_forest_model.fit(
        X_train,
        y_train
    )

    print("Random Forest trained.")

    return {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model
    }


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    name,
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n")
    print("=" * 60)
    print(name.upper())
    print("=" * 60)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # 1. Load dataset
    df = load_data()

    # 2. Prepare X and y
    X, y = prepare_data(df)

    print("\nFraud distribution:")

    print(
        y.value_counts()
    )

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Test split completed.")

    print(
        f"Training rows : {len(X_train):,}"
    )

    print(
        f"Testing rows  : {len(X_test):,}"
    )

    # 4. Create preprocessing pipeline
    preprocessor = create_preprocessor(
        X_train
    )

    # 5. Train models
    models = train_models(
        X_train,
        y_train,
        preprocessor
    )

    # 6. Evaluate
    results = {}

    for name, model in models.items():

        results[name] = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

    # 7. Select model based on ROC-AUC
    best_model_name = max(
        results,
        key=lambda name:
        results[name]["roc_auc"]
    )

    best_model = models[
        best_model_name
    ]

    print("\n")
    print("=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(
        best_model_name
    )

    print(
        f"ROC-AUC: "
        f"{results[best_model_name]['roc_auc']:.4f}"
    )

    # 8. Create models folder
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 9. Save complete pipeline
    joblib.dump(
        best_model,
        MODEL_PATH
    )

    print("\nModel saved successfully!")

    print(
        f"Location: {MODEL_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()