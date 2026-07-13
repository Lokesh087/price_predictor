"""
=========================================================
Evaluate Regression Models
=========================================================
"""

import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
def adjusted_r2(
    r2,
    n,
    p
):
    return (
        1 - ((1-r2) * (n-1))/(n-p-1))
def evaluate_model(
        model_name,
        y_true,
        y_pred,
        feature_count
):

    r2 = r2_score(
        y_true,
        y_pred
    )
    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mape = mean_absolute_percentage_error(
        y_true,
        y_pred
    )

    adj = adjusted_r2(
        r2,
        len(y_true),
        feature_count
    )
    print("\n")
    print("="*60)
    print(model_name)
    print("="*60)
    print(f"R2 Score          : {r2:.4f}")
    print(f"Adjusted R2       : {adj:.4f}")
    print(f"MAE               : {mae:.4f}")
    print(f"RMSE              : {rmse:.4f}")
    print(f"MAPE              : {mape:.4f}")
    print("="*60)
    return {
        "Model":model_name,
        "R2":r2,
        "Adjusted_R2":adj,
        "MAE":mae,
        "RMSE":rmse,
        "MAPE":mape
    }