from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from Evaluation.eval_training_history import plot_training_history
from modeling_dl import TemporalSequenceBuilder
from modeling_torch import (
    calculate_regression_metrics,
    make_device,
    predict_torch_model,
    predict_torch_hybrid_model,
    train_torch_lstm,
    train_torch_hybrid_lstm,
)


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "torch_lstm_config.yaml"


class TorchHistory:
    def __init__(self, history_df):
        self.history = {
            "loss": history_df["loss"].tolist(),
            "val_loss": history_df["val_loss"].tolist(),
            "mae": history_df["mae"].tolist(),
            "val_mae": history_df["val_mae"].tolist(),
        }


def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    config = load_config()
    set_random_seed(config.get("random_state", 42))

    device = make_device(config.get("device", "auto"))
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = resolve_path(config["output_dir"])
    model_path = resolve_path(config["model_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building PyTorch LSTM sequence data...")
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

    print("Training PyTorch LSTM model...")
    tabular_data = None
    if config.get("tabular", {}).get("enabled", False):
        print("Loading preprocessed tabular features...")
        tabular_data = load_tabular_data(config["tabular"])
        validate_tabular_alignment(sequence_data, tabular_data)
        print("Training hybrid PyTorch LSTM + tabular model...")
        result = train_torch_hybrid_lstm(
            sequence_data=sequence_data,
            tabular_data=tabular_data,
            model_config=config["model"],
            training_config=config["training"],
            device=device,
        )
        model_type = "hybrid_lstm_tabular"
    else:
        print("Training sequence-only PyTorch LSTM model...")
        result = train_torch_lstm(
            sequence_data=sequence_data,
            model_config=config["model"],
            training_config=config["training"],
            device=device,
        )
        model_type = "sequence_only_lstm"

    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "model_config": config["model"],
            "sequence_features": sequence_data["feature_names"],
            "lookback_hours": config["lookback_hours"],
            "model_type": model_type,
            "tabular_features": (
                tabular_data["feature_names"] if tabular_data is not None else None
            ),
        },
        model_path,
    )
    joblib.dump(sequence_builder.scaler, output_dir / "sequence_scaler.pkl")

    result.history.to_csv(output_dir / "training_history.csv", index=False)
    plot_training_history(
        TorchHistory(result.history),
        output_path=output_dir / "training_history.png",
        show=False,
    )

    metrics = {}
    for split_name in ["val", "test"]:
        y_true_log = sequence_data[f"y_{split_name}"]
        if tabular_data is not None:
            y_pred_log = predict_torch_hybrid_model(
                result.model,
                sequence_data[f"X_{split_name}"],
                tabular_data[f"X_{split_name}"],
                batch_size=config["training"]["batch_size"],
                device=device,
            )
        else:
            y_pred_log = predict_torch_model(
                result.model,
                sequence_data[f"X_{split_name}"],
                batch_size=config["training"]["batch_size"],
                device=device,
            )

        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
        metrics[split_name] = calculate_regression_metrics(y_true, y_pred)

        pd.DataFrame(
            {
                "actual": y_true,
                "prediction": y_pred,
                "residual": y_true - y_pred,
            }
        ).to_csv(
            output_dir / f"y_pred_{split_name}.csv",
            index=False,
        )

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(output_dir / "metrics.csv")

    print("\nPyTorch LSTM metrics:")
    print(metrics_df.to_string())
    print(f"\nSaved model to: {model_path}")
    print(f"Saved outputs to: {output_dir}")


def load_tabular_data(tabular_config):
    X_train = read_tabular_split(resolve_path(tabular_config["X_train_path"]))
    X_val = read_tabular_split(resolve_path(tabular_config["X_val_path"]))
    X_test = read_tabular_split(resolve_path(tabular_config["X_test_path"]))

    return {
        "X_train": X_train.to_numpy(dtype=np.float32),
        "X_val": X_val.to_numpy(dtype=np.float32),
        "X_test": X_test.to_numpy(dtype=np.float32),
        "feature_names": X_train.columns.tolist(),
    }


def read_tabular_split(path):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def validate_tabular_alignment(sequence_data, tabular_data):
    for split_name in ["train", "val", "test"]:
        sequence_rows = sequence_data[f"X_{split_name}"].shape[0]
        tabular_rows = tabular_data[f"X_{split_name}"].shape[0]

        if sequence_rows != tabular_rows:
            raise ValueError(
                f"Hybrid data row mismatch for {split_name}: "
                f"sequence has {sequence_rows:,} rows, "
                f"tabular has {tabular_rows:,} rows. "
                "Rerun src/preprocessing_main.py after generating "
                "data/dataset_with_filtered_distance_dl.csv so the tabular "
                "splits use the same 24h-window filtered dataset."
            )


if __name__ == "__main__":
    main()
