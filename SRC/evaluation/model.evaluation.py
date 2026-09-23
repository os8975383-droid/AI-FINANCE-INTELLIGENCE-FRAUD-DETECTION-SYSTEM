from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.inspection import permutation_importance


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "DATA" / "transactions.csv"
MODEL_PATH = PROJECT_ROOT / "MODELS" / "fraud_detection_model.joblib"
MODELS_DIR = PROJECT_ROOT / "MODELS"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["transaction_id", "is_fraud"])
y = df["is_fraud"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# 6. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 7. MODEL METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("\n" + "=" * 55)
print("              MODEL EVALUATION RESULTS")
print("=" * 55)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc * 100:.2f}%")

print("=" * 55)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

print("\nCreating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

disp.plot()

plt.title("Fraud Detection - Confusion Matrix")
plt.tight_layout()

confusion_path = MODELS_DIR / "confusion_matrix.png"

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {confusion_path}")


# ============================================================
# 10. ROC CURVE
# ============================================================

print("\nCreating ROC curve...")

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Fraud Detection - ROC Curve")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

roc_path = MODELS_DIR / "roc_curve.png"

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {roc_path}")


# ============================================================
# 11. PERMUTATION FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="roc_auc",
    n_jobs=-1,
)

importance = pd.Series(
    result.importances_mean,
    index=X_test.columns
)

importance = importance.sort_values(
    ascending=False
)

print("\nTop Features:")

print(importance.head(10))


# ============================================================
# 12. FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = importance.head(10).sort_values()

plt.figure(figsize=(9, 6))

top_features.plot(
    kind="barh"
)

plt.xlabel("Permutation Importance")

plt.ylabel("Feature")

plt.title(
    "Top 10 Features - Fraud Detection Model"
)

plt.tight_layout()

feature_path = MODELS_DIR / "feature_importance.png"

plt.savefig(
    feature_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {feature_path}")


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 55)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 55)

print("\nGenerated files:")

print("1. MODELS/confusion_matrix.png")
print("2. MODELS/roc_curve.png")
print("3. MODELS/feature_importance.png")

print("\nYour model is ready for dashboard evaluation.")