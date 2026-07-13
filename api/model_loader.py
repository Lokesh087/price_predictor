from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "saved_models"

MODEL_FILE = MODEL_DIR / "xgboost.pkl"

FEATURE_FILE = MODEL_DIR / "feature_columns.pkl"


model = joblib.load(MODEL_FILE)

feature_columns = joblib.load(FEATURE_FILE)