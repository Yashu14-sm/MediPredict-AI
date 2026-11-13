
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "datasets" / "processed"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "evaluation" / "results"
PLOT_DIR = ROOT / "evaluation" / "plots"

for folder in [MODEL_DIR, RESULT_DIR, PLOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_SENSITIVITY = 0.90

DATASETS = {
    "Heart Disease": {
        "file": "heart_disease_cleaned.csv",
        "target": "num"
    },
    "Diabetes": {
        "file": "diabetes_cleaned.csv",
        "target": "Outcome"
    },
    "Breast Cancer": {
        "file": "breast_cancer_cleaned.csv",
        "target": "Diagnosis"
    }
}

# --------------------------------------------------
# 2. Define the three ML algorithms
# --------------------------------------------------

def create_models():

    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=RANDOM_STATE
            ))
        ])
    }


# --------------------------------------------------
# 3. Calculate evaluation metrics
# --------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "sensitivity_recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "specificity": specificity,
        "f1_score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


# --------------------------------------------------
# 4. Select threshold using validation data only
# --------------------------------------------------

def choose_threshold(y_true, probabilities):

    thresholds = np.unique(
        np.concatenate(([0.0, 1.0], probabilities))
    )

    candidates = []

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0

        candidates.append({
            "threshold": float(threshold),
            "sensitivity": sensitivity,
            "specificity": specificity
        })

    eligible = [
        item for item in candidates
        if item["sensitivity"] >= TARGET_SENSITIVITY
    ]

    if eligible:
        # Meet the sensitivity target, then maximize specificity.
        selected = max(
            eligible,
            key=lambda item: (
                item["specificity"],
                item["threshold"]
            )
        )
    else:
        # If the target cannot be met, prioritize sensitivity.
        selected = max(
            candidates,
            key=lambda item: (
                item["sensitivity"],
                item["specificity"]
            )
        )

    return selected["threshold"]


# --------------------------------------------------
# 5. Plot confusion matrix
# --------------------------------------------------

def save_confusion_matrix(y_true, probabilities, threshold,
                          dataset_name):

    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["Actual Negative", "Actual Positive"]
    )

    plt.title(f"{dataset_name} - Confusion Matrix")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")

    filename = dataset_name.lower().replace(" ", "_")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{filename}_confusion_matrix.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 6. Train and compare all models
# --------------------------------------------------

def main():

    validation_results = []
    final_test_results = []
    best_model_summary = {}

    for dataset_name, config in DATASETS.items():

        print("\n" + "=" * 65)
        print(f"DATASET: {dataset_name}")
        print("=" * 65)

        file_path = DATA_DIR / config["file"]
        target_column = config["target"]

        df = pd.read_csv(file_path)

        X = df.drop(columns=[target_column])
        y = df[target_column].astype(int)

        # Ensure all predictors are numeric.
        X = X.apply(pd.to_numeric, errors="coerce")

        # 60% training, 20% validation, 20% testing.
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.40,
            random_state=RANDOM_STATE,
            stratify=y
        )

        X_validation, X_test, y_validation, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )

        print("Training records:", len(X_train))
        print("Validation records:", len(X_validation))
        print("Testing records:", len(X_test))

        models = create_models()
        fitted_models = {}

        # ------------------------------------------
        # Train and compare models on validation set
        # ------------------------------------------

        for model_name, model in models.items():

            print(f"\nTraining {model_name}...")

            model.fit(X_train, y_train)

            validation_probabilities = model.predict_proba(
                X_validation
            )[:, 1]

            metrics = calculate_metrics(
                y_validation,
                validation_probabilities,
                threshold=0.5
            )

            validation_results.append({
                "Dataset": dataset_name,
                "Model": model_name,
                **metrics
            })

            fitted_models[model_name] = {
                "pipeline": model,
                "probabilities": validation_probabilities,
                "roc_auc": metrics["roc_auc"]
            }

            print(
                f"Validation ROC-AUC: {metrics['roc_auc']:.4f}"
            )

        # ------------------------------------------
        # Select the best model using validation AUC
        # ------------------------------------------

        best_model_name = max(
            fitted_models,
            key=lambda name: fitted_models[name]["roc_auc"]
        )

        best_model = fitted_models[best_model_name]["pipeline"]
        best_validation_probabilities = fitted_models[
            best_model_name
        ]["probabilities"]

        # Tune threshold using validation data only.
        best_threshold = choose_threshold(
            y_validation,
            best_validation_probabilities
        )

        # ------------------------------------------
        # Final evaluation on untouched test set
        # ------------------------------------------

        test_probabilities = best_model.predict_proba(X_test)[:, 1]

        test_metrics = calculate_metrics(
            y_test,
            test_probabilities,
            best_threshold
        )

        final_test_results.append({
            "Dataset": dataset_name,
            "Selected Model": best_model_name,
            **test_metrics
        })

        # Save the selected model and its threshold.
        safe_name = dataset_name.lower().replace(" ", "_")

        artifact = {
            "model": best_model,
            "threshold": best_threshold,
            "dataset": dataset_name,
            "target_column": target_column,
            "feature_columns": X.columns.tolist(),
            "selected_model": best_model_name
        }

        joblib.dump(
            artifact,
            MODEL_DIR / f"{safe_name}_best_model.joblib"
        )

        # Save test confusion matrix.
        save_confusion_matrix(
            y_test,
            test_probabilities,
            best_threshold,
            dataset_name
        )

        best_model_summary[dataset_name] = {
            "selected_model": best_model_name,
            "validation_roc_auc": float(
                fitted_models[best_model_name]["roc_auc"]
            ),
            "selected_threshold": float(best_threshold),
            "test_metrics": {
                key: float(value) if isinstance(
                    value, (np.floating, np.integer)
                ) else value
                for key, value in test_metrics.items()
            }
        }

        print("\nSelected model:", best_model_name)
        print(f"Selected threshold: {best_threshold:.4f}")
        print("\nTest results:")

        for metric, value in test_metrics.items():
            print(f"{metric}: {value}")

    # --------------------------------------------------
    # 7. Save all comparison results
    # --------------------------------------------------

    validation_df = pd.DataFrame(validation_results)
    test_df = pd.DataFrame(final_test_results)

    validation_df.to_csv(
        RESULT_DIR / "validation_model_comparison.csv",
        index=False
    )

    test_df.to_csv(
        RESULT_DIR / "final_test_results.csv",
        index=False
    )

    with open(
        RESULT_DIR / "best_models_summary.json",
        "w"
    ) as file:
        json.dump(best_model_summary, file, indent=4)

    print("\n" + "=" * 65)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 65)

    print("\nValidation comparison:")
    print(
        validation_df[
            ["Dataset", "Model", "roc_auc", "accuracy",
             "sensitivity_recall", "specificity"]
        ].to_string(index=False)
    )

    print("\nFinal test results:")
    print(test_df.to_string(index=False))

    print("\nSaved files:")
    print("Models:", MODEL_DIR)
    print("Results:", RESULT_DIR)
    print("Plots:", PLOT_DIR)


if __name__ == "__main__":
    main()
