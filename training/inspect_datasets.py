from pathlib import Path
import pandas as pd

raw_dir = Path("datasets/raw")

files = [
    "heart_disease.csv",
    "pima_diabetes.csv",
    "breast_cancer_wisconsin.csv"
]

for filename in files:
    print("\n" + "=" * 60)
    print(filename)

    df = pd.read_csv(raw_dir / filename)

    print("Rows and columns:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst five records:")
    print(df.head())