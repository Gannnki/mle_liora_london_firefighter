"""Evaluate the saved best tabular model on train, validation, and test splits."""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

PATH_MODEL = BASE_DIR / "artifacts/best_models/best_model.pkl"

PATH_X_TRAIN = BASE_DIR / "output/scalers/X_train_scaled.csv"
PATH_X_VAL = BASE_DIR / "output/scalers/X_val_scaled.csv"
PATH_X_TEST = BASE_DIR / "output/scalers/X_test_scaled.csv"

PATH_Y_TRAIN = BASE_DIR / "output/data_splits/y_train.csv"
PATH_Y_VAL = BASE_DIR / "output/data_splits/y_val.csv"
PATH_Y_TEST = BASE_DIR / "output/data_splits/y_test.csv"

PATH_OUTPUT_METRICS = BASE_DIR / "output/predictions/model_eval_metrics.csv"
PATH_OUTPUT_TEST_PRED = BASE_DIR / "output/predictions/y_pred_test_eval.csv"


def main():
    """Load the saved model and write final train/validation/test metrics."""
    model = load_model(PATH_MODEL)

    X_train = read_features(PATH_X_TRAIN)
    X_val = read_features(PATH_X_VAL)
    X_test = read_features(PATH_X_TEST)
    validate_feature_count(model, X_train)

    y_train = read_target(PATH_Y_TRAIN)
    y_val = read_target(PATH_Y_VAL)
    y_test = read_target(PATH_Y_TEST)

    model_name = model.__class__.__name__
    model_metrics = {
        "Model": model_name,
        **prefixed_metrics("Train", y_train, predict_original_scale(model, X_train)),
        **prefixed_metrics("Validation", y_val, predict_original_scale(model, X_val)),
        **prefixed_metrics("Test", y_test, predict_original_scale(model, X_test)),
        "Best Params": None,
    }

    naive_train_pred = np.full(len(y_train), y_train.mean())
    naive_val_pred = np.full(len(y_val), y_train.mean())
    naive_test_pred = np.full(len(y_test), y_train.mean())
    naive_metrics = {
        "Model": "Naive Baseline (Mean)",
        **prefixed_metrics("Train", y_train, naive_train_pred),
        **prefixed_metrics("Validation", y_val, naive_val_pred),
        **prefixed_metrics("Test", y_test, naive_test_pred),
        "Best Params": None,
    }

    results = pd.DataFrame([model_metrics, naive_metrics])
    ordered_columns = [
        "Model",
        "Train MAE",
        "Train RMSE",
        "Train R2 Score",
        "Validation MAE",
        "Validation RMSE",
        "Validation R2 Score",
        "Validation RMSLE",
        "Validation P90",
        "Test MAE",
        "Test RMSE",
        "Test R2 Score",
        "Test RMSLE",
        "Test P90",
        "Best Params",
    ]
    results = results.reindex(columns=ordered_columns)

    print("\nBest model evaluation:")
    print(results.to_string(index=False))

    PATH_OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(PATH_OUTPUT_METRICS, index=False)

    test_prediction = predict_original_scale(model, X_test)
    pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "prediction": test_prediction,
            "residual": y_test.to_numpy() - test_prediction,
        }
    ).to_csv(PATH_OUTPUT_TEST_PRED, index=False)

    print(f"\nSaved metrics to: {PATH_OUTPUT_METRICS}")
    print(f"Saved test predictions to: {PATH_OUTPUT_TEST_PRED}")


def load_model(path):
    """Load a saved model artifact and unwrap GridSearchCV-style estimators."""
    model = joblib.load(path)
    return getattr(model, "best_estimator_", model)


def read_features(path):
    """Read feature CSV files as float32 arrays for model prediction."""
    return pd.read_csv(path).to_numpy(dtype=np.float32, copy=False)


def read_target(path):
    """Read a target CSV as a one-dimensional pandas Series."""
    return pd.read_csv(path).squeeze("columns")


def validate_feature_count(model, X):
    """Fail fast when the saved model and current feature matrix do not match."""
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is None:
        return

    actual_features = X.shape[1]
    if expected_features != actual_features:
        raise ValueError(
            "Saved model feature count does not match current scaled data. "
            f"Model expects {expected_features} features, but current data has "
            f"{actual_features}. Rerun src/modeling_main.py after the latest "
            "src/preprocessing_main.py so best_model.pkl is trained on the "
            "same feature set."
        )


def predict_original_scale(model, X):
    """Predict log-scale targets and convert them back to seconds."""
    y_pred_log = model.predict(X)
    return np.expm1(y_pred_log)


def prefixed_metrics(prefix, y_true, y_pred):
    """Return regression metrics with a split-specific column prefix."""
    absolute_error = np.abs(y_true - y_pred)
    return {
        f"{prefix} MAE": round(mean_absolute_error(y_true, y_pred), 2),
        f"{prefix} RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        f"{prefix} R2 Score": round(r2_score(y_true, y_pred), 4),
        f"{prefix} RMSLE": round(
            np.sqrt(mean_squared_log_error(y_true, np.maximum(y_pred, 0))),
            2,
        ),
        f"{prefix} P90": round(np.percentile(absolute_error, 90), 4),
    }


if __name__ == "__main__":
    main()
