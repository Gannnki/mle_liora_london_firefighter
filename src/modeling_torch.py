"""Experimental PyTorch LSTM utilities, not part of the main XGBoost pipeline."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TorchLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size,
        lstm_units=64,
        dense_units=32,
        num_layers=1,
        dropout=0.2,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_units, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.regressor(last_hidden).squeeze(-1)


class TorchHybridLSTMRegressor(nn.Module):
    def __init__(
        self,
        sequence_input_size,
        tabular_input_size,
        lstm_units=64,
        dense_units=32,
        tabular_hidden_units=128,
        combined_hidden_units=64,
        num_layers=1,
        dropout=0.2,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=sequence_input_size,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_input_size, tabular_hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tabular_hidden_units, dense_units),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_units + dense_units, combined_hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_hidden_units, 1),
        )

    def forward(self, sequence_x, tabular_x):
        sequence_output, _ = self.lstm(sequence_x)
        sequence_embedding = sequence_output[:, -1, :]
        tabular_embedding = self.tabular_branch(tabular_x)
        combined = torch.cat([sequence_embedding, tabular_embedding], dim=1)
        return self.regressor(combined).squeeze(-1)


@dataclass
class TrainingResult:
    model: nn.Module
    history: pd.DataFrame
    best_val_loss: float


def make_device(device_name="auto"):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def make_data_loader(X, y, batch_size, shuffle=False, num_workers=0):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_hybrid_data_loader(
    X_sequence,
    X_tabular,
    y,
    batch_size,
    shuffle=False,
    num_workers=0,
):
    dataset = TensorDataset(
        torch.from_numpy(X_sequence).float(),
        torch.from_numpy(X_tabular).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_torch_lstm(
    sequence_data,
    model_config,
    training_config,
    device,
):
    model = TorchLSTMRegressor(
        input_size=sequence_data["X_train"].shape[-1],
        lstm_units=model_config["lstm_units"],
        dense_units=model_config["dense_units"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(device)

    train_loader = make_data_loader(
        sequence_data["X_train"],
        sequence_data["y_train"],
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config.get("num_workers", 0),
    )
    val_loader = make_data_loader(
        sequence_data["X_val"],
        sequence_data["y_val"],
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config.get("num_workers", 0),
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )

    history_rows = []
    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, training_config["epochs"] + 1):
        train_loss, train_mae = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_mae = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.5f} mae={train_mae:.5f} | "
            f"val_loss={val_loss:.5f} val_mae={val_mae:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= training_config["patience"]:
            print(f"Early stopping after epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(
        model=model,
        history=pd.DataFrame(history_rows),
        best_val_loss=best_val_loss,
    )


def train_torch_hybrid_lstm(
    sequence_data,
    tabular_data,
    model_config,
    training_config,
    device,
):
    model = TorchHybridLSTMRegressor(
        sequence_input_size=sequence_data["X_train"].shape[-1],
        tabular_input_size=tabular_data["X_train"].shape[-1],
        lstm_units=model_config["lstm_units"],
        dense_units=model_config["dense_units"],
        tabular_hidden_units=model_config["tabular_hidden_units"],
        combined_hidden_units=model_config["combined_hidden_units"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
    ).to(device)

    train_loader = make_hybrid_data_loader(
        sequence_data["X_train"],
        tabular_data["X_train"],
        sequence_data["y_train"],
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config.get("num_workers", 0),
    )
    val_loader = make_hybrid_data_loader(
        sequence_data["X_val"],
        tabular_data["X_val"],
        sequence_data["y_val"],
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config.get("num_workers", 0),
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )

    history_rows = []
    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, training_config["epochs"] + 1):
        train_loss, train_mae = run_hybrid_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_mae = run_hybrid_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.5f} mae={train_mae:.5f} | "
            f"val_loss={val_loss:.5f} val_mae={val_mae:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= training_config["patience"]:
            print(f"Early stopping after epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(
        model=model,
        history=pd.DataFrame(history_rows),
        best_val_loss=best_val_loss,
    )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_absolute_error = 0.0
    total_rows = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = y_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_absolute_error += torch.abs(prediction - y_batch).sum().item()
        total_rows += batch_size

    return total_loss / total_rows, total_absolute_error / total_rows


def run_hybrid_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_absolute_error = 0.0
    total_rows = 0

    for X_sequence, X_tabular, y_batch in loader:
        X_sequence = X_sequence.to(device, non_blocking=True)
        X_tabular = X_tabular.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            prediction = model(X_sequence, X_tabular)
            loss = criterion(prediction, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = y_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_absolute_error += torch.abs(prediction - y_batch).sum().item()
        total_rows += batch_size

    return total_loss / total_rows, total_absolute_error / total_rows


def predict_torch_model(model, X, batch_size, device):
    loader = DataLoader(
        torch.from_numpy(X).float(),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    predictions = []

    model.eval()
    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            predictions.append(model(X_batch).detach().cpu().numpy())

    return np.concatenate(predictions)


def predict_torch_hybrid_model(model, X_sequence, X_tabular, batch_size, device):
    dataset = TensorDataset(
        torch.from_numpy(X_sequence).float(),
        torch.from_numpy(X_tabular).float(),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    predictions = []

    model.eval()
    with torch.no_grad():
        for X_sequence_batch, X_tabular_batch in loader:
            X_sequence_batch = X_sequence_batch.to(device, non_blocking=True)
            X_tabular_batch = X_tabular_batch.to(device, non_blocking=True)
            predictions.append(
                model(X_sequence_batch, X_tabular_batch).detach().cpu().numpy()
            )

    return np.concatenate(predictions)


def calculate_regression_metrics(y_true, y_pred):
    absolute_error = np.abs(y_true - y_pred)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "P90_absolute_error": np.percentile(absolute_error, 90),
    }
