"""
===================================================
Main Training Pipeline
====================================================
"""

from models.linear_regression import train as train_linear
from models.decision_tree import train as train_tree
from models.random_forest import train as train_forest
from models.gradient_boosting import train as train_gb
from models.xgboost_model import train as train_xgb

from reports.report_generator import save_metrics


def main():

    print("=" * 70)
    print("        PRICEPILOT AI TRAINING PIPELINE")
    print("=" * 70)

    print("\nTraining Linear Regression...")
    lr_metrics = train_linear()
    save_metrics(lr_metrics)

    print("\nTraining Decision Tree...")
    dt_metrics = train_tree()
    save_metrics(dt_metrics)

    print("\nTraining Random Forest...")
    rf_metrics = train_forest()
    save_metrics(rf_metrics)

    print("\nTraining Gradient Boosting...")
    gb_metrics = train_gb()
    save_metrics(gb_metrics)
    print("\nTraining XGBoost...")
    xgb_metrics = train_xgb()
    save_metrics(xgb_metrics)
    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 70)
if __name__ == "__main__":
    main()