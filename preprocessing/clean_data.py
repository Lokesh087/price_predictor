"""
==========================================================
Clean merged dataset
==========================================================
"""
from pathlib import Path
import pandas as pd
import numpy as np
# ======================================================
# Paths
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
INPUT_FILE = PROCESSED_DIR / "merged_dataset.csv"
OUTPUT_FILE = PROCESSED_DIR / "clean_dataset.csv"
# ======================================================
# Load Dataset
# ======================================================

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=[
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date"
    ]
)

print("\nDataset Loaded")
print(df.shape)

# ======================================================
# Remove Duplicate Rows
# ======================================================

duplicates = df.duplicated().sum()

print("\nDuplicate Rows :", duplicates)
df.drop_duplicates(inplace=True)


# ======================================================
# Keep Delivered Orders Only
# ======================================================

df = df[
    df["order_status"] == "delivered"
]
# ======================================================
# Missing Values
# ======================================================

print("\nMissing Values")
print(df.isnull().sum())

df["product_category_name"] = df[
    "product_category_name"
].fillna("Unknown")

numeric_columns = [

    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"

]

for col in numeric_columns:
    df[col] = df[col].fillna(
        df[col].median()
    )


# ======================================================
# Remove Invalid Records
# ======================================================

df = df[df["price"] > 0]
df = df[df["freight_value"] >= 0]
df = df[df["product_weight_g"] > 0]
df = df[df["product_length_cm"] > 0]
df = df[df["product_height_cm"] > 0]
df = df[df["product_width_cm"] > 0]


# ======================================================
# Remove Outliers
# ======================================================

columns = [
    "price",
    "freight_value",
    "product_weight_g"
]

for col in columns:

    q1 = df[col].quantile(0.25)

    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr

    upper = q3 + 1.5 * iqr

    df = df[
        (df[col] >= lower)
        &
        (df[col] <= upper)
    ]


# ======================================================
# Reset Index
# ======================================================

df.reset_index(
    drop=True,
    inplace=True
)


# ======================================================
# Save Dataset
# ======================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)
print("\nCleaning Completed")
print("Final Shape :", df.shape)
print("\nSaved")
print(OUTPUT_FILE)