"""
==================================================
Decision Tree Regressor
==================================================
"""

from sklearn.tree import DecisionTreeRegressor
from utils.data_loader import load_dataset
from utils.metrics import evaluate_model
from utils.save_model import save_model
from utils.plots import *
from config import TREE_MAX_DEPTH, RANDOM_STATE
from reports.report_generator import save_metrics
def train():
    X_train, X_test, y_train, y_test, columns = load_dataset()
    model = DecisionTreeRegressor(
        max_depth=TREE_MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = evaluate_model(
        "Decision Tree",
        y_test,
        pred,
        X_train.shape[1]
    )
    save_metrics(metrics)
    
    actual_vs_predicted(
        y_test,
        pred,
        "Decision Tree"
    )
    residual_plot(
        y_test,
        pred,
        "Decision Tree"
    )
    error_distribution(
        y_test,
        pred,
        "Decision Tree"
    )
    prediction_error(
        model,
        X_test,
        y_test,
        "Decision Tree"
    )
    feature_importance(
        model,
        columns,
        "Decision Tree"
    )
    save_model(
        model,
        "decision_tree.pkl"
    )
if __name__ == "__main__":
    train()