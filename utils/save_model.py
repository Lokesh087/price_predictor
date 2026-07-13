"""
=========================================================
Save Trained Models
=========================================================
"""
from pathlib import Path
import joblib
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
def save_model(model, model_name):
    """Save trained model."""
    model_path = MODEL_DIR / model_name
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved: {model_path}")
def save_feature_columns(columns):
    """Save feature column names."""
    feature_path = MODEL_DIR / "feature_columns.pkl"
    joblib.dump(list(columns), feature_path)
    print("✅ Feature columns saved")
def load_model(model_name):
    """Load saved model."""
    model_path = MODEL_DIR / model_name
    return joblib.load(model_path)