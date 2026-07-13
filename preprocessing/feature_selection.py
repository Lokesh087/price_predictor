"""
==================================================
Feature Selection
==================================================
"""
from pathlib import Path
import pandas as pd
# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_DIR = BASE_DIR / "dataset" / "final"
INPUT_FILE = FINAL_DIR / "encoded_dataset.csv"
OUTPUT_FILE = FINAL_DIR / "training_dataset.csv"

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("Original Dataset Shape")
print(df.shape)

# ==========================================
# Columns to Remove
# ==========================================

drop_columns = [

    "order_id",
    "order_item_id",
    "product_id",
    "customer_id",
    "seller_id",
    "shipping_limit_date",
    "order_status"
]

drop_columns = [
    col
    for col in drop_columns
    if col in df.columns
]

df.drop(
    columns=drop_columns,
    inplace=True
)

print("\nColumns Removed")
print(drop_columns)
print("\nFinal Shape")
print(df.shape)

# ==========================================
# Save Dataset
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nTraining Dataset Saved")
print(OUTPUT_FILE)
