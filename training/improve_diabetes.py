
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score
)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "datasets/processed/diabetes_cleaned.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "evaluation/results"
PLOT_DIR = ROOT / "evaluation/plots"

for folder in [MODEL_DIR, RESULT_DIR, PLOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_SENSITIVITY = 0.90

# -------------------------------------------------
# 1. Load diabetes dataset
# -------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Outcome"])
y = df["Outcome"].astype(int)

X = X.apply(pd.to_numeric, errors="coerce")

# 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# -------------------------------------------------
# 2. Define models and hyperparameter grids
# -------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

searches = {
    "Logistic Regression": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE
            ))
        ]),
        {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__class_weight": [None, "balanced"]
        }
    ),

    "Random Forest": (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_leaf": [1, 3, 5],
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
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__max_depth": [1, 2, 3],
            "classifier__min_samples_leaf": [3, 5, 10]
        }
    )
}

# -------------------------------------------------
# 3. Hyperparameter tuning using training data only
# -------------------------------------------------

tuned_models = {}
comparison = []

for name, (pipeline, parameters) in searches.items():

    print("\n" + "=" * 55)
    print("Tuning:", name)
    print("=" * 55)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    val_probabilities = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probabilities)

    tuned_models[name] = {
        "model": best_model,
        "validation_probabilities": val_probabilities,
        "validation_auc": val_auc,
        "best_parameters": search.best_params_
    }

    comparison.append({
        "Model": name,
        "CV ROC-AUC": search.best_score_,
        "Validation ROC-AUC": val_auc,
        "Best Parameters": str(search.best_params_)
    })

    print("Best CV ROC-AUC:", round(search.best_score_, 4))
    print("Validation ROC-AUC:", round(val_auc, 4))
    print("Best parameters:", search.best_params_)

# -------------------------------------------------
# 4. Select best model using validation ROC-AUC
# -------------------------------------------------

best_name = max(
    tuned_models,
    key=lambda name: tuned_models[name]["validation_auc"]
)

best_model = tuned_models[best_name]["model"]
val_probabilities = tuned_models[best_name]["validation_probabilities"]

print("\nSelected model:", best_name)

# -------------------------------------------------
# 5. Select threshold using validation data only
# -------------------------------------------------

threshold_candidates = np.unique(
    np.concatenate(([0.0, 1.0], val_probabilities))
)

threshold_results = []

for threshold in threshold_candidates:

    val_predictions = (val_probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_predictions,
        labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    threshold_results.append({
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity
    })

eligible = [
    item for item in threshold_results
    if item["sensitivity"] >= TARGET_SENSITIVITY
]

if eligible:
    selected = max(
        eligible,
        key=lambda item: (item["specificity"], item["threshold"])
    )
else:
    selected = max(
        threshold_results,
        key=lambda item: (item["sensitivity"], item["specificity"])
    )

best_threshold = selected["threshold"]

print("Selected threshold:", round(best_threshold, 4))
print("Validation sensitivity:", round(selected["sensitivity"], 4))
print("Validation specificity:", round(selected["specificity"], 4))

# -------------------------------------------------
# 6. Final evaluation on untouched test data
# -------------------------------------------------

test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

specificity = tn / (tn + fp) if (tn + fp) else 0

test_metrics = {
    "selected_model": best_name,
    "threshold": float(best_threshold),
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "sensitivity_recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "specificity": float(specificity),
    "f1_score": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp)
}

# -------------------------------------------------
# 7. Save tuned model
# -------------------------------------------------

artifact = {
    "model": best_model,
    "threshold": best_threshold,
    "dataset": "Diabetes",
    "target_column": "Outcome",
    "feature_columns": X.columns.tolist(),
    "selected_model": best_name
}

joblib.dump(
    artifact,
    MODEL_DIR / "diabetes_best_model.joblib"
)

# -------------------------------------------------
# 8. Save results
# -------------------------------------------------

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    RESULT_DIR / "diabetes_tuning_comparison.csv",
    index=False
)

with open(
    RESULT_DIR / "diabetes_improved_metrics.json",
    "w"
) as file:
    json.dump(test_metrics, file, indent=4)

# -------------------------------------------------
# 9. Save confusion matrix
# -------------------------------------------------

cm = np.array([[tn, fp], [fn, tp]])

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Negative", "Predicted Positive"],
    yticklabels=["Actual Negative", "Actual Positive"]
)

plt.title("Diabetes - Improved Model Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "diabetes_improved_confusion_matrix.png",
    dpi=300
)

plt.close()

# -------------------------------------------------
# 10. Print final results
# -------------------------------------------------

print("\n" + "=" * 55)
print("DIABETES MODEL IMPROVEMENT RESULTS")
print("=" * 55)

print("\nTuned model comparison:")
print(comparison_df.to_string(index=False))

print("\nFinal test metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value}")

print("\nImproved diabetes model saved successfully.")
