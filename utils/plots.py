"""
=========================================================
Generate Model Evaluation Graphs
=========================================================
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import PredictionErrorDisplay

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports" / "graphs"
REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)
# =====================================================
# Actual vs Predicted
# =====================================================
def actual_vs_predicted(
        y_true,
        y_pred,
        model_name

):

    plt.figure(figsize=(8,6))
    plt.scatter(
        y_true,
        y_pred,
        alpha=0.5
    )
    plt.plot(
        [
            y_true.min(),
            y_true.max()
        ],
        [
            y_true.min(),
            y_true.max()
        ],
        linestyle="--"
    )
    plt.title(
        f"{model_name} - Actual vs Predicted"
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR /
        f"{model_name}_actual_vs_predicted.png"
    )
    plt.close()
# =====================================================
# Residual Plot
# =====================================================

def residual_plot(
        y_true,
        y_pred,
        model_name
):

    residual = y_true - y_pred
    plt.figure(figsize=(8,6))
    plt.scatter(
        y_pred,
        residual,
        alpha=0.5
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title(
        f"{model_name} Residual Plot"
    )

    plt.tight_layout()

    plt.savefig(

        REPORT_DIR /

        f"{model_name}_residual.png"

    )

    plt.close()


# =====================================================
# Error Distribution
# =====================================================

def error_distribution(

        y_true,

        y_pred,

        model_name

):

    errors = y_true - y_pred

    plt.figure(figsize=(8,6))

    plt.hist(

        errors,

        bins=40

    )

    plt.title(

        f"{model_name} Error Distribution"

    )

    plt.xlabel("Prediction Error")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(

        REPORT_DIR /

        f"{model_name}_error_distribution.png"

    )

    plt.close()


# =====================================================
# Feature Importance
# =====================================================

def feature_importance(

        model,

        columns,

        model_name

):

    if not hasattr(

        model,

        "feature_importances_"

    ):

        return

    importance = pd.DataFrame({

        "Feature":columns,

        "Importance":model.feature_importances_

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False

    )

    plt.figure(figsize=(10,8))

    plt.barh(

        importance["Feature"],

        importance["Importance"]

    )

    plt.title(

        f"{model_name} Feature Importance"

    )

    plt.tight_layout()

    plt.savefig(

        REPORT_DIR /

        f"{model_name}_feature_importance.png"

    )

    plt.close()


# =====================================================
# Prediction Error Plot
# =====================================================

def prediction_error(

        model,

        X_test,

        y_test,

        model_name

):

    fig, ax = plt.subplots(figsize=(8,6))

    PredictionErrorDisplay.from_predictions(

        y_true=y_test,

        y_pred=model.predict(X_test),

        ax=ax

    )

    plt.tight_layout()

    plt.savefig(

        REPORT_DIR /

        f"{model_name}_prediction_error.png"

    )

    plt.close()