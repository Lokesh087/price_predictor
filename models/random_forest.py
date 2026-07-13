"""
==================================================
Random Forest Regressor
==================================================
"""

from sklearn.ensemble import RandomForestRegressor
from reports.report_generator import save_metrics
from utils.data_loader import load_dataset
from utils.metrics import evaluate_model
from utils.save_model import save_model
from utils.plots import *

from config import RF_ESTIMATORS, RANDOM_STATE


def train():
    X_train, X_test, y_train, y_test, columns = load_dataset()
    model = RandomForestRegressor(
        n_estimators=RF_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = evaluate_model(
        "Random Forest",
        y_test,
        pred,
        X_train.shape[1]
    )
    
    save_metrics(metrics)

    actual_vs_predicted(
        y_test,
        pred,
        "Random Forest"
    )

    residual_plot(
        y_test,
        pred,
        "Random Forest"
    )

    error_distribution(
        y_test,
        pred,
        "Random Forest"
    )
    prediction_error(
        model,
        X_test,
        y_test,
        "Random Forest"
    )
    feature_importance(
        model,
        columns,
        "Random Forest"
    )
    save_model(
        model,
        "random_forest.pkl"
    )
if __name__ == "__main__":
    train()