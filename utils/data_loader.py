"""
=========================================================
Load dataset and split into train/test
=========================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset" / "final"
DATA_FILE = DATASET_DIR / "training_dataset.csv"


def load_dataset(test_size=0.2, random_state=42):

    df = pd.read_csv(DATA_FILE)
    print("=" * 60)
    print("Dataset Loaded Successfully")
    print("=" * 60)
    print(df.shape)

    # -----------------------------
    # Drop columns not useful
    # -----------------------------

    remove_columns = [
        "order_id",
        "product_id",
        "customer_id",
        "order_item_id"
    ]

    remove_columns = [
        col
        for col in remove_columns
        if col in df.columns
    ]
    df.drop(
        columns=remove_columns,
        inplace=True
    )
    # -----------------------------
    # Features
    # -----------------------------
    X = df.drop(
        columns=["price"]
    )
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    print("\nTraining Samples :", X_train.shape)
    print("Testing Samples :", X_test.shape)
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X.columns
    )
    # Fill missing numeric values
    numeric_columns = df.select_dtypes(include=["number"]).columns

    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values
    categorical_columns = df.select_dtypes(include=["object"]).columns

    for col in categorical_columns:
        df[col] = df[col].fillna("Unknown")