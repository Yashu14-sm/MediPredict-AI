
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "datasets" / "raw"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. DIABETES PREPROCESSING
# --------------------------------------------------

def preprocess_diabetes():

    print("\nPreprocessing Diabetes Dataset")

    df = pd.read_csv(RAW_DIR / "pima_diabetes.csv")

    # In this dataset, zero is not a plausible measurement
    # for these particular clinical variables.
    invalid_zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in invalid_zero_columns:
        df[column] = df[column].replace(0, np.nan)

    # Target must remain binary.
    df["Outcome"] = df["Outcome"].astype(int)

    # Save cleaned dataset with missing values preserved.
    # Imputation will be fitted only on training data later.
    df.to_csv(
        PROCESSED_DIR / "diabetes_cleaned.csv",
        index=False
    )

    print("Shape:", df.shape)
    print("Missing values:")
    print(df.isnull().sum())
    print("Target distribution:")
    print(df["Outcome"].value_counts())


# --------------------------------------------------
# 2. HEART DISEASE PREPROCESSING
# --------------------------------------------------

def preprocess_heart():

    print("\nPreprocessing Heart Disease Dataset")

    df = pd.read_csv(RAW_DIR / "heart_disease.csv")

    # Normalize column names.
    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    if "num" not in df.columns:
        raise ValueError(
            f"Expected target 'num'. Found: {df.columns.tolist()}"
        )

    # UCI Cleveland target:
    # 0 = no disease
    # 1, 2, 3, 4 = disease present
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    df = df.dropna(subset=["num"])

    df["num"] = (df["num"] > 0).astype(int)

    # Convert feature columns to numeric where possible.
    for column in df.columns:
        if column != "num":
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df.to_csv(
        PROCESSED_DIR / "heart_disease_cleaned.csv",
        index=False
    )

    print("Shape:", df.shape)
    print("Missing values:")
    print(df.isnull().sum())
    print("Target distribution:")
    print(df["num"].value_counts())


# --------------------------------------------------
# 3. BREAST CANCER PREPROCESSING
# --------------------------------------------------

def preprocess_breast_cancer():

    print("\nPreprocessing Breast Cancer Dataset")

    df = pd.read_csv(
        RAW_DIR / "breast_cancer_wisconsin.csv"
    )

    if "Diagnosis" not in df.columns:
        raise ValueError("Diagnosis target column not found.")

    # M = malignant = 1
    # B = benign = 0
    df["Diagnosis"] = (
        df["Diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"M": 1, "B": 0})
    )

    if df["Diagnosis"].isnull().any():
        raise ValueError("Unexpected breast cancer target labels.")

    # Convert predictors to numeric.
    feature_columns = [
        column for column in df.columns
        if column != "Diagnosis"
    ]

    df[feature_columns] = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df.to_csv(
        PROCESSED_DIR / "breast_cancer_cleaned.csv",
        index=False
    )

    print("Shape:", df.shape)
    print("Missing values:")
    print(df.isnull().sum())
    print("Target distribution:")
    print(df["Diagnosis"].value_counts())


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    preprocess_diabetes()
    preprocess_heart()
    preprocess_breast_cancer()

    print("\nAll preprocessing completed!")
