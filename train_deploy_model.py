"""Rebuild the Olist preprocessing pipeline and train the production XGBoost model."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
steps=["preprocessing/merge_data.py","preprocessing/clean_data.py","preprocessing/feature_engineering.py","preprocessing/encoding.py","preprocessing/feature_selection.py","models/xgboost_model.py"]
for step in steps:
 print(f"\n=== {step} ===",flush=True)
 subprocess.run([sys.executable,step],cwd=ROOT,check=True)
required=[ROOT/"saved_models/xgboost.pkl",ROOT/"saved_models/feature_columns.pkl",ROOT/"saved_models/product_category_name_encoder.pkl"]
missing=[str(p) for p in required if not p.is_file()]
if missing: raise SystemExit("Missing model artifacts: "+", ".join(missing))
print("PricePilot model training complete.",flush=True)
