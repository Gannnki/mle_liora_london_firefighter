from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent.parent.parent

PATH_MODEL = BASE_DIR / "artifacts/best_models/best_model.pkl"
PATH_X_VAL = BASE_DIR / "output/scalers/X_val_scaled.csv"
PATH_Y_VAL = BASE_DIR / "output/data_splits/y_val.csv"
PATH_OUTPUT = BASE_DIR / "output/predictions/y_pred_validation_eval.csv"

MODEL_WAS_TRAINED_ON_LOG_TARGET = True


def make_safe_feature_names(columns):
    safe_columns = []
    seen = {}

    for column in columns:
        safe_name = re.sub(
            r"[^0-9A-Za-z_]+",
            "_",
            str(column),
        ).strip("_")

        if not safe_name:
            safe_name = "feature"

        count = seen.get(safe_name, 0)
        seen[safe_name] = count + 1

        if count:
            safe_name = f"{safe_name}_{count}"

        safe_columns.append(safe_name)

    return safe_columns


def main():
    model = joblib.load(PATH_MODEL)
    X_val = pd.read_csv(PATH_X_VAL)
    X_val.columns = make_safe_feature_names(X_val.columns)
    y_val = pd.read_csv(PATH_Y_VAL).squeeze()

    y_pred = model.predict(X_val)

    if MODEL_WAS_TRAINED_ON_LOG_TARGET:
        y_pred = np.expm1(y_pred)

    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    print("Validation metrics:")
    print(f"Rows: {len(y_val):,}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

    output_df = pd.DataFrame(
        {
            "actual": y_val.to_numpy(),
            "prediction": y_pred,
            "residual": y_val.to_numpy() - y_pred,
            "absolute_error": np.abs(y_val.to_numpy() - y_pred),
        }
    )

    PATH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(PATH_OUTPUT, index=False)
    print(f"Saved validation predictions to: {PATH_OUTPUT}")


if __name__ == "__main__":
    main()
