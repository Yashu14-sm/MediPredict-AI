
from pathlib import Path
import requests
import pandas as pd
from ucimlrepo import fetch_ucirepo

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


def save_dataset(dataframe, filename):
    path = RAW_DIR / filename
    dataframe.to_csv(path, index=False)
    print(f"Saved: {path}")
    print(f"Shape: {dataframe.shape}")
    print("-" * 55)


# --------------------------------------------------
# 1. UCI HEART DISEASE
# UCI Dataset ID: 45
# --------------------------------------------------

def download_heart_disease():
    print("\nDownloading UCI Heart Disease...")

    dataset = fetch_ucirepo(id=45)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    targets.columns = [
        str(column).strip().replace(" ", "_")
        for column in targets.columns
    ]

    dataframe = pd.concat(
        [features.reset_index(drop=True),
         targets.reset_index(drop=True)],
        axis=1
    )

    save_dataset(dataframe, "heart_disease.csv")

    print("Heart Disease target information:")
    print(dataset.variables.tail())


# --------------------------------------------------
# 2. PIMA INDIANS DIABETES
# Original 768-row dataset
# --------------------------------------------------

def download_pima_diabetes():
    print("\nDownloading Pima Indians Diabetes...")

    url = (
        "https://raw.githubusercontent.com/"
        "jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv"
    )

    columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome"
    ]

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    from io import StringIO

    dataframe = pd.read_csv(
        StringIO(response.text),
        header=None,
        names=columns
    )

    save_dataset(dataframe, "pima_diabetes.csv")

    print("Target distribution:")
    print(dataframe["Outcome"].value_counts())


# --------------------------------------------------
# 3. BREAST CANCER WISCONSIN (DIAGNOSTIC)
# UCI Dataset ID: 17
# --------------------------------------------------

def download_breast_cancer():
    print("\nDownloading Breast Cancer Wisconsin...")

    dataset = fetch_ucirepo(id=17)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    targets.columns = [
        str(column).strip().replace(" ", "_")
        for column in targets.columns
    ]

    dataframe = pd.concat(
        [features.reset_index(drop=True),
         targets.reset_index(drop=True)],
        axis=1
    )

    save_dataset(dataframe, "breast_cancer_wisconsin.csv")

    print("Target distribution:")
    print(targets.value_counts())


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("MEDIPREDICT AI - DATASET COLLECTION")
    print("=" * 60)

    try:
        download_heart_disease()
        download_pima_diabetes()
        download_breast_cancer()

        print("\nAll three datasets downloaded successfully!")

    except Exception as error:
        print("\nDataset download failed.")
        print("Error:", error)
        print(
            "\nCheck your internet connection and try again."
        )
