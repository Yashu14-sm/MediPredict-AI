
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    balanced_accuracy_score
)

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "datasets/processed/diabetes_cleaned.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "evaluation/results"
PLOT_DIR = ROOT / "evaluation/plots"

for folder in [MODEL_DIR, RESULT_DIR, PLOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# 1. Load data
df = pd.read_csv(DATA)

X = df.drop(columns=["Outcome"]).apply(
    pd.to_numeric, errors="coerce"
)
y = df["Outcome"].astype(int)

# Keep the same 60/20/20 split.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40,
    random_state=RANDOM_STATE, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=RANDOM_STATE, stratify=y_temp
)

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE
)

# 2. Define model pipelines and search spaces
experiments = {
    "Logistic Regression - Median": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=4000, random_state=RANDOM_STATE
            ))
        ]),
        {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    ),

    "Logistic Regression - KNN": (
        Pipeline([
            ("imputer", KNNImputer()),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=4000, random_state=RANDOM_STATE
            ))
        ]),
        {
            "imputer__n_neighbors": [3, 5, 7],
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    ),

    "Logistic Regression - Interactions": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler1", StandardScaler()),
            ("polynomial", PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False
            )),
            ("scaler2", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=5000, random_state=RANDOM_STATE
            ))
        ]),
        {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    ),

    "Random Forest": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [3, 5, 8, None],
            "classifier__min_samples_leaf": [2, 4, 6],
            "classifier__class_weight": [None, "balanced"]
        }
    ),

    "Gradient Boosting": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(
                random_state=RANDOM_STATE
            ))
        ]),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.02, 0.05, 0.1],
            "classifier__max_depth": [1, 2, 3],
            "classifier__min_samples_leaf": [3, 5, 10]
        }
    )
}

# 3. Tune each experiment using training data only
results = []
fitted = {}

for name, (pipeline, parameters) in experiments.items():

    print("\n" + "=" * 60)
    print("Tuning:", name)
    print("=" * 60)

    search = GridSearchCV(
        pipeline,
        parameters,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    search.fit(X_train, y_train)

    model = search.best_estimator_
    probabilities = model.predict_proba(X_val)[:, 1]

    val_auc = roc_auc_score(y_val, probabilities)

    # Initial validation accuracy at 0.5 threshold
    val_predictions = (probabilities >= 0.5).astype(int)
    val_accuracy = accuracy_score(y_val, val_predictions)

    fitted[name] = {
        "model": model,
        "probabilities": probabilities,
        "cv_accuracy": search.best_score_,
        "validation_auc": val_auc,
        "validation_accuracy_05": val_accuracy,
        "parameters": search.best_params_
    }

    results.append({
        "Model": name,
        "CV Accuracy": search.best_score_,
        "Validation Accuracy at 0.5": val_accuracy,
        "Validation ROC-AUC": val_auc,
        "Parameters": str(search.best_params_)
    })

    print("CV accuracy:", round(search.best_score_, 4))
    print("Validation accuracy:", round(val_accuracy, 4))
    print("Validation ROC-AUC:", round(val_auc, 4))
    print("Best parameters:", search.best_params_)

# 4. Select model using validation accuracy at threshold 0.5
best_name = max(
    fitted,
    key=lambda n: (
        fitted[n]["validation_accuracy_05"],
        fitted[n]["validation_auc"]
    )
)

best_model = fitted[best_name]["model"]
val_probabilities = fitted[best_name]["probabilities"]

# 5. Find threshold maximizing validation accuracy
thresholds = np.linspace(0.05, 0.95, 181)
threshold_records = []

for threshold in thresholds:

    predictions = (val_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0

    threshold_records.append({
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_val, predictions),
        "sensitivity": sensitivity,
        "specificity": specificity
    })

# Highest validation accuracy; sensitivity breaks ties.
selected = max(
    threshold_records,
    key=lambda x: (x["accuracy"], x["sensitivity"])
)

best_threshold = selected["threshold"]

print("\nSelected model:", best_name)
print("Validation threshold:", best_threshold)
print("Validation accuracy:", selected["accuracy"])
print("Validation sensitivity:", selected["sensitivity"])
print("Validation specificity:", selected["specificity"])

# 6. Evaluate once on untouched test data
test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

specificity = tn / (tn + fp) if tn + fp else 0

metrics = {
    "selected_model": best_name,
    "threshold": float(best_threshold),
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "sensitivity": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "specificity": float(specificity),
    "f1_score": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
    "TN": int(tn), "FP": int(fp),
    "FN": int(fn), "TP": int(tp)
}

# 7. Save selected model
joblib.dump({
    "model": best_model,
    "threshold": best_threshold,
    "dataset": "Diabetes",
    "target_column": "Outcome",
    "feature_columns": X.columns.tolist(),
    "selected_model": best_name
}, MODEL_DIR / "diabetes_best_model.joblib")

# 8. Save comparisons and metrics
comparison_df = pd.DataFrame(results)

comparison_df.to_csv(
    RESULT_DIR / "diabetes_advanced_comparison.csv",
    index=False
)

with open(RESULT_DIR / "diabetes_advanced_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 9. Save confusion matrix
cm = np.array([[tn, fp], [fn, tp]])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Predicted Negative", "Predicted Positive"],
    yticklabels=["Actual Negative", "Actual Positive"]
)
plt.title("Diabetes - Advanced Model Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(
    PLOT_DIR / "diabetes_advanced_confusion_matrix.png",
    dpi=300
)
plt.close()

# 10. Print results
print("\n" + "=" * 60)
print("ADVANCED DIABETES MODEL COMPARISON")
print("=" * 60)
print(comparison_df.to_string(index=False))

print("\nFINAL UNTOUCHED TEST RESULTS")
for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nSaved advanced diabetes model and results.")
