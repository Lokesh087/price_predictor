"""
==========================================================
Encode categorical variables
==========================================================
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ======================================================
# Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_DIR = BASE_DIR / "dataset" / "final"
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILE = FINAL_DIR / "model_dataset.csv"
OUTPUT_FILE = FINAL_DIR / "encoded_dataset.csv"

# ======================================================
# Load Dataset
# ======================================================

df = pd.read_csv(INPUT_FILE)
print("Dataset Loaded")
print(df.shape)

# ======================================================
# Categorical Columns
# ======================================================

categorical_columns = [
    "product_category_name",
    "seller_id",
    "order_status"
]
encoders = {}
# ======================================================
# Label Encoding
# ======================================================

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(
        df[column].astype(str)
    )
    encoders[column] = encoder
    joblib.dump(
        encoder,
        MODEL_DIR / f"{column}_encoder.pkl"
    )
print("Encoding Completed")
# ======================================================
# Save Dataset
# ======================================================
df.to_csv(
    OUTPUT_FILE,
    index=False
)
print("\nEncoded Dataset Saved")
print(OUTPUT_FILE)