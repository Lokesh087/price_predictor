"""
==========================================================
Feature Engineering for Price Prediction
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
FINAL_DIR = BASE_DIR / "dataset" / "final"

FINAL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "clean_dataset.csv"
OUTPUT_FILE = FINAL_DIR / "model_dataset.csv"

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

print("Dataset Loaded Successfully")

# ======================================================
# Purchase Date Features
# ======================================================

df["purchase_year"] = df["order_purchase_timestamp"].dt.year
df["purchase_month"] = df["order_purchase_timestamp"].dt.month
df["purchase_day"] = df["order_purchase_timestamp"].dt.day
df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
df["purchase_quarter"] = df["order_purchase_timestamp"].dt.quarter

# ======================================================
# Weekend
# ======================================================

df["is_weekend"] = np.where(
    df["purchase_weekday"] >= 5,
    1,
    0
)

# ======================================================
# Delivery Features
# ======================================================

df["delivery_days"] = (
    df["order_delivered_customer_date"]
    - df["order_purchase_timestamp"]
).dt.days

df["estimated_delivery_days"] = (
    df["order_estimated_delivery_date"]
    - df["order_purchase_timestamp"]
).dt.days

df["delivery_difference"] = (
    df["estimated_delivery_days"]
    - df["delivery_days"]
)

# ======================================================
# Handle Missing Values in Engineered Features
# ======================================================

df["delivery_days"] = df["delivery_days"].fillna(
    df["delivery_days"].median()
)

df["estimated_delivery_days"] = df["estimated_delivery_days"].fillna(
    df["estimated_delivery_days"].median()
)

df["delivery_difference"] = df["delivery_difference"].fillna(
    df["delivery_difference"].median()
)
# ======================================================
# Product Volume
# ======================================================

df["product_volume"] = (

    df["product_length_cm"]

    *

    df["product_height_cm"]

    *

    df["product_width_cm"]

)

# ======================================================
# Product Density
# ======================================================

# df["product_density"] = (df["product_weight_g"] / df["product_volume"])
# df["product_density"] = df["product_density"].replace(
#     [np.inf, -np.inf], np.nan.fillna(0))
# ======================================================
# Product Density
# ======================================================

df["product_density"] = (
    df["product_weight_g"] / df["product_volume"]
)

# Replace infinite values with NaN
df["product_density"] = df["product_density"].replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill NaN values with 0
df["product_density"] = df["product_density"].fillna(0)

# ======================================================
# Remove Unnecessary Columns
# ======================================================
drop_columns = [
    "shipping_limit_date",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]
df.drop(
    columns=drop_columns,
    inplace=True,
    errors="ignore"
)
# ======================================================
# Final Missing Value Check
# ======================================================

numeric_cols = df.select_dtypes(include=["number"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

print("\nRemaining Missing Values:", df.isnull().sum().sum())
# ======================================================
# Save Dataset
# ======================================================
df.to_csv(
    OUTPUT_FILE,
    index=False
)
print("\nFeature Engineering Completed")
print(df.shape)
print("\nSaved Successfully")
print(OUTPUT_FILE)