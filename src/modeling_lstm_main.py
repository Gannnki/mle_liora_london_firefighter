"""Experimental TensorFlow LSTM training entrypoint, not part of the main pipeline."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Evaluation.eval_training_history import plot_training_history
from modeling_dl import TemporalSequenceBuilder, fit_lstm_model


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "lstm_model_config.yaml"


def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def main():
    config = load_config()

    output_dir = resolve_path(config["output_dir"])
    model_path = resolve_path(config["model_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building LSTM sequence data...")
    sequence_builder = TemporalSequenceBuilder(
        data_path=resolve_path(config["data_path"]),
        config_path=resolve_path(config["pipeline_config_path"]),
        lookback_hours=config["lookback_hours"],
        sequence_features=config.get("sequence_features"),
        max_rows=config.get("max_rows"),
    )
    sequence_data = sequence_builder.build()

    print("Sequence shapes:")
    print(f"X_train: {sequence_data['X_train'].shape}")
    print(f"X_val: {sequence_data['X_val'].shape}")
    print(f"X_test: {sequence_data['X_test'].shape}")

    print("Training LSTM model...")
    model_config = config["model"]
    training_config = config["training"]
    model, model_history, sequence_data = fit_lstm_model(
        sequence_data=sequence_data,
        lstm_units=model_config["lstm_units"],
        dense_units=model_config["dense_units"],
        dropout=model_config["dropout"],
        learning_rate=model_config["learning_rate"],
        epochs=training_config["epochs"],
        batch_size=training_config["batch_size"],
        patience=training_config["patience"],
    )

    model.save(model_path)
    joblib.dump(sequence_builder.scaler, output_dir / "sequence_scaler.pkl")

    history_df = pd.DataFrame(model_history.history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    plot_training_history(
        model_history,
        output_path=output_dir / "training_history.png",
        show=False,
    )

    metrics = {}
    for split_name in ["val", "test"]:
        y_true_log = sequence_data[f"y_{split_name}"]
        y_pred_log = model.predict(
            sequence_data[f"X_{split_name}"],
            batch_size=training_config["batch_size"],
        ).reshape(-1)

        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)

        split_metrics = calculate_metrics(y_true, y_pred)
        metrics[split_name] = split_metrics

        prediction_df = pd.DataFrame(
            {
                "actual": y_true,
                "prediction": y_pred,
                "residual": y_true - y_pred,
            }
        )
        prediction_df.to_csv(
            output_dir / f"y_pred_{split_name}.csv",
            index=False,
        )

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(output_dir / "metrics.csv")

    print("\nLSTM metrics:")
    print(metrics_df.to_string())
    print(f"\nSaved model to: {model_path}")
    print(f"Saved LSTM outputs to: {output_dir}")


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    absolute_error = np.abs(y_true - y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
        "P90_absolute_error": np.percentile(absolute_error, 90),
    }


if __name__ == "__main__":
    main()
