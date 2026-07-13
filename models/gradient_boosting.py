"""
==========================================================
Train Gradient Boosting Regressor
==========================================================
"""
from sklearn.ensemble import GradientBoostingRegressor
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
def train():
    print("="*60)
    print("Gradient Boosting Regressor")
    print("="*60)
    X_train, X_test, y_train, y_test, columns = load_dataset()
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE
    )
    print("\nTraining Model...\n")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_model(
        "Gradient Boosting",
        y_test,
        predictions,
        X_train.shape[1]
    )
    save_metrics(metrics)
    
    actual_vs_predicted(
        y_test,
        predictions,
        "Gradient_Boosting"
    )
    residual_plot(
        y_test,
        predictions,
        "Gradient_Boosting"
    )
    error_distribution(
        y_test,
        predictions,
        "Gradient_Boosting"
    )
    prediction_error(
        model,
        X_test,
        y_test,
        "Gradient_Boosting"
    )
    feature_importance(
        model,
        columns,
        "Gradient_Boosting"

    )
    save_model(
        model,
        "gradient_boosting.pkl"
    )
    save_feature_columns(columns)
    print("\nTraining Completed Successfully")
    return metrics
if __name__ == "__main__":
    train()