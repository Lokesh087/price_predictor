"""
==================================================
Project : PricePilot AI

Linear Regression Model
==================================================
"""

from sklearn.linear_model import LinearRegression

from reports.report_generator import save_metrics

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
    prediction_error
)


def train():

    try:

        # ==========================================
        # Load Dataset
        # ==========================================

        X_train, X_test, y_train, y_test, columns = load_dataset()

        # ==========================================
        # Initialize Model
        # ==========================================

        model = LinearRegression()

        # ==========================================
        # Train Model
        # ==========================================

        print("\nTraining Linear Regression Model...\n")

        model.fit(
            X_train,
            y_train
        )

        # ==========================================
        # Prediction
        # ==========================================

        predictions = model.predict(
            X_test
        )

        # ==========================================
        # Evaluation
        # ==========================================

        metrics = evaluate_model(

            "Linear Regression",

            y_test,

            predictions,

            X_train.shape[1]

        )

        # ==========================================
        # Save Metrics
        # ==========================================

        save_metrics(metrics)

        # ==========================================
        # Graphs
        # ==========================================

        actual_vs_predicted(

            y_test,

            predictions,

            "Linear Regression"

        )

        residual_plot(

            y_test,

            predictions,

            "Linear Regression"

        )

        error_distribution(

            y_test,

            predictions,

            "Linear Regression"

        )

        prediction_error(

            model,

            X_test,

            y_test,

            "Linear Regression"

        )

        # ==========================================
        # Save Model
        # ==========================================

        save_model(

            model,

            "linear_regression.pkl"

        )

        save_feature_columns(columns)

        print("\n✅ Linear Regression Training Completed Successfully")

        return metrics

    except Exception as e:

        print("\n❌ Error while training Linear Regression")

        print(e)

        return None


if __name__ == "__main__":

    train()
# """
# ==================================================
# Linear Regression Model
# ==================================================
# """

# from sklearn.linear_model import LinearRegression
# from reports.report_generator import save_metrics
# from utils.data_loader import load_dataset
# from utils.metrics import evaluate_model
# from utils.save_model import( save_model,save_feature_columns)

# from utils.plots import (
#     actual_vs_predicted,
#     residual_plot,
#     error_distribution,
#     prediction_error
# )


# def train():

#     X_train, X_test, y_train, y_test, columns = load_dataset()
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     metrics = evaluate_model(
#         "Linear Regression",
#         y_test,
#         predictions,
#         X_train.shape[1]
#     )
#     save_metrics(metrics)

#     actual_vs_predicted(
#         y_test,
#         predictions,
#         "Linear Regression"
#     )
#     residual_plot(
#         y_test,
#         predictions,
#         "Linear Regression"
#     )
#     error_distribution(
#         y_test,
#         predictions,
#         "Linear Regression"
#     )
#     prediction_error(
#         model,
#         X_test,
#         y_test,
#         "Linear Regression"
#     )
#     save_model(
#         model,
#         "linear_regression.pkl"
#     )
#     save_feature_columns(columns)
#     print("\n✅ Linear Regression Training Completed Successfully")

#     return metrics
# if __name__ == "__main__":
#     train()