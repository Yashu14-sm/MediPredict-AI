
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"
PLOT_DIR = BASE_DIR / "evaluation" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

DATASETS = {
    "Heart Disease": {
        "file": "heart_disease.csv",
        "target": "num"
    },
    "Diabetes": {
        "file": "pima_diabetes.csv",
        "target": "Outcome"
    },
    "Breast Cancer": {
        "file": "breast_cancer_wisconsin.csv",
        "target": "Diagnosis"
    }
}


def analyze_dataset(name, config):

    print("\n" + "=" * 65)
    print(f"EDA REPORT: {name}")
    print("=" * 65)

    path = RAW_DIR / config["file"]

    df = pd.read_csv(path)
    target = config["target"]

    # Basic information
    print("\nDataset shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nStatistical summary:")
    print(df.describe(include="all").T)

    # Target distribution
    print("\nTarget distribution:")
    print(df[target].value_counts(dropna=False))

    # Target distribution chart
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=target)
    plt.title(f"{name} - Target Distribution")
    plt.xlabel("Target class")
    plt.ylabel("Number of records")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{name.lower().replace(' ', '_')}_target.png"
    )
    plt.close()

    # Missing values chart
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        plt.figure(figsize=(10, 5))
        missing.sort_values().plot(kind="barh")
        plt.title(f"{name} - Missing Values")
        plt.tight_layout()
        plt.savefig(
            PLOT_DIR / f"{name.lower().replace(' ', '_')}_missing.png"
        )
        plt.close()

    # Correlation heatmap for numeric columns
    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            numeric_df.corr(),
            cmap="coolwarm",
            center=0
        )
        plt.title(f"{name} - Correlation Heatmap")
        plt.tight_layout()

        plt.savefig(
            PLOT_DIR / f"{name.lower().replace(' ', '_')}_correlation.png"
        )
        plt.close()

    print("\nEDA completed for", name)


if __name__ == "__main__":

    for name, config in DATASETS.items():

        try:
            analyze_dataset(name, config)

        except Exception as error:
            print(f"\nCould not analyze {name}: {error}")

    print("\nAll EDA reports generated.")
