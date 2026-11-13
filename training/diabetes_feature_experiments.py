
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score,
    precision_score, f1_score, confusion_matrix,
    balanced_accuracy_score
)


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "datasets/processed/diabetes_cleaned.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "evaluation/results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# Create selected interaction features
class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = np.asarray(X, dtype=float).copy()

        columns = X.shape[1]

        # Original diabetes feature order:
        # 0 Pregnancies
        # 1 Glucose
        # 2 BloodPressure
        # 3 SkinThickness
        # 4 Insulin
        # 5 BMI
        # 6 DiabetesPedigreeFunction
        # 7 Age

        interactions = np.column_stack([
            X[:, 1] * X[:, 5],  # Glucose x BMI
            X[:, 7] * X[:, 5],  # Age x BMI
            X[:, 0] * X[:, 7],  # Pregnancies x Age
            X[:, 1] * X[:, 7]   # Glucose x Age
        ])

        return np.hstack([X, interactions])


# Load dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Outcome"]).apply(
    pd.to_numeric, errors="coerce"
)

y = df["Outcome"].astype(int)


# Preserve the original split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Experiments
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("features", DiabetesFeatureEngineer()),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=5000,
        random_state=42
    ))
])

parameters = [
    {
        "imputer": [
            SimpleImputer(strategy="median")
        ],
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"]
    },
    {
        "imputer": [
            KNNImputer(n_neighbors=3),
            KNNImputer(n_neighbors=5),
            KNNImputer(n_neighbors=7)
        ],
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"]
    }
]


search = GridSearchCV(
    pipeline,
    parameters,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)

print("Training feature-engineered models...")

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest cross-validation ROC-AUC:")
print(round(search.best_score_, 4))

print("\nBest parameters:")
print(search.best_params_)


# Validation evaluation
val_probabilities = best_model.predict_proba(X_val)[:, 1]

val_auc = roc_auc_score(y_val, val_probabilities)

thresholds = np.linspace(0.10, 0.90, 161)

threshold_results = []

for threshold in thresholds:

    predictions = (val_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0

    threshold_results.append({
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_val, predictions),
        "sensitivity": sensitivity,
        "specificity": specificity
    })


# Select the threshold with the best validation accuracy
selected = max(
    threshold_results,
    key=lambda x: (x["accuracy"], x["sensitivity"])
)

best_threshold = selected["threshold"]


# Final test evaluation
test_probabilities = best_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

specificity = tn / (tn + fp) if tn + fp else 0

metrics = {
    "model": "Feature Engineered Logistic Regression",
    "threshold": float(best_threshold),
    "validation_accuracy": float(selected["accuracy"]),
    "validation_auc": float(val_auc),
    "test_accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "test_balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_predictions)
    ),
    "test_precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "test_sensitivity": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "test_specificity": float(specificity),
    "test_f1": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "test_roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp)
}


# Save model
joblib.dump({
    "model": best_model,
    "threshold": best_threshold,
    "dataset": "Diabetes",
    "target_column": "Outcome",
    "feature_columns": X.columns.tolist(),
    "selected_model": "Feature Engineered Logistic Regression"
}, MODEL_DIR / "diabetes_feature_model.joblib")


with open(
    RESULT_DIR / "diabetes_feature_metrics.json", "w"
) as file:
    json.dump(metrics, file, indent=4)


print("\nFINAL RESULTS")

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nFeature experiment completed.")
