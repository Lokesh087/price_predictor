"""
==========================================================

Train XGBoost Regressor
==========================================================
"""
from xgboost import XGBRegressor
from utils.data_loader import load_dataset
from utils.metrics import evaluate_model
from utils.save_model import (
    save_model,
    save_feature_columns
)
from utils.plots import (
    actual_vs_predicted,
    residual_plot,
    error_distribution,
    feature_importance,
    prediction_error
)
from config import RANDOM_STATE
from reports.report_generator import save_metrics
# ===============================================
# Train Model
# ===============================================
def train():

    print("=" * 60)
    print("XGBoost Price Prediction Model")
    print("=" * 60)

    X_train, X_test, y_train, y_test, columns = load_dataset()

    model = XGBRegressor(

        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1

    )

    print("\nTraining XGBoost Model...\n")
    model.fit(
        X_train,
        y_train
    )

    print("Training Completed\n")
    predictions = model.predict(
        X_test
    )

    metrics = evaluate_model(
        "XGBoost",

        y_test,

        predictions,

        X_train.shape[1]

    )

    save_metrics(metrics)

    actual_vs_predicted(

        y_test,

        predictions,

        "XGBoost"

    )

    residual_plot(

        y_test,

        predictions,

        "XGBoost"

    )

    error_distribution(

        y_test,

        predictions,

        "XGBoost"

    )

    prediction_error(

        model,

        X_test,

        y_test,

        "XGBoost"

    )

    feature_importance(

        model,

        columns,

        "XGBoost"

    )

    save_model(

        model,

        "xgboost.pkl"

    )

    save_feature_columns(

        columns

    )

    print("\nXGBoost Model Saved Successfully")

    return metrics


# ===============================================
# Main
# ===============================================
if __name__ == "__main__":
    train()