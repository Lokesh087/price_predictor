from pathlib import Path
import joblib
BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_DIR=BASE_DIR/"saved_models"
MODEL_FILE=MODEL_DIR/"xgboost.pkl"
FEATURE_FILE=MODEL_DIR/"feature_columns.pkl"
ENCODER_FILE=MODEL_DIR/"product_category_name_encoder.pkl"
missing=[p.name for p in (MODEL_FILE,FEATURE_FILE,ENCODER_FILE) if not p.exists()]
if missing: raise RuntimeError("Missing trained model artifacts: "+", ".join(missing)+". Run python train_deploy_model.py.")
model=joblib.load(MODEL_FILE)
feature_columns=joblib.load(FEATURE_FILE)
category_encoder=joblib.load(ENCODER_FILE)