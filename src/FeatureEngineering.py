from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from helpers.export_helpers import export_to_csv

class FeatureEncoder:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.feature_config = self.config["feature_encoding"]

        self.one_hot_encoders = {}
        self.top_n_categories = {}
        self.loo_encoders = {}

        self.fitted_columns = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        encoded_parts = []

        for col, cfg in self.feature_config.items():
            if col not in X.columns:
                print(f"Warning: {col} not found in X. Skipping.")
                continue

            print(f"\nEncoding column: {col} | Method: {cfg['encoding']}")
            encoding = cfg["encoding"]

            if encoding == "IGNORE":
                continue

            elif encoding in ["NUMERIC_KEEP", "BINARY_KEEP"]:
                encoded_parts.append(X[[col]].copy())

            elif encoding == "CYCLIC":
                encoded_parts.append(self._cyclic_encode(X, col))

            elif encoding == "ONE_HOT":
                encoded_parts.append(self._fit_transform_one_hot(X, col))

            elif encoding == "TOP_N_PLUS_ONE_HOT":
                encoded_parts.append(self._fit_transform_top_n_one_hot(X, col, cfg))

            elif encoding == "LEAVE_ONE_OUT_TARGET":
                encoded_parts.append(self._fit_transform_loo(X, y, col, cfg))

            else:
                raise ValueError(f"Unknown encoding type: {encoding}")

        X_encoded = pd.concat(encoded_parts, axis=1)
        self.fitted_columns = X_encoded.columns.tolist()
        self.X_train_encoded = X_encoded

        return X_encoded

    def transform(self, X: pd.DataFrame, split_name: str = None) -> pd.DataFrame:
        encoded_parts = []

        for col, cfg in self.feature_config.items():
            if col not in X.columns:
                print(f"Warning: {col} not found in X. Skipping.")
                continue

            encoding = cfg["encoding"]

            if encoding == "IGNORE":
                continue

            elif encoding in ["NUMERIC_KEEP", "BINARY_KEEP"]:
                encoded_parts.append(X[[col]].copy())

            elif encoding == "CYCLIC":
                encoded_parts.append(self._cyclic_encode(X, col))

            elif encoding == "ONE_HOT":
                encoded_parts.append(self._transform_one_hot(X, col))

            elif encoding == "TOP_N_PLUS_ONE_HOT":
                encoded_parts.append(self._transform_top_n_one_hot(X, col))

            elif encoding == "LEAVE_ONE_OUT_TARGET":
                encoded_parts.append(self._transform_loo(X, col))

            else:
                raise ValueError(f"Unknown encoding type: {encoding}")

        X_encoded = pd.concat(encoded_parts, axis=1)
        X_encoded = X_encoded.reindex(columns=self.fitted_columns, fill_value=0)

        if split_name:
            # save the encoded split as an attribute for later use and export
            setattr(self, f"X_{split_name}_encoded", X_encoded)
        return X_encoded

    def _cyclic_encode(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        period_map = {
            "Month": 12,
            "Weekday": 7,
            "Hour": 24,
        }

        if col not in period_map:
            raise ValueError(f"No cyclic period defined for column: {col}")

        period = period_map[col]

        return pd.DataFrame(
            {
                f"{col}_sin": np.sin(2 * np.pi * X[col] / period),
                f"{col}_cos": np.cos(2 * np.pi * X[col] / period),
            },
            index=X.index,
        )

    def _fit_transform_one_hot(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        encoder = ce.OneHotEncoder(
            cols=[col],
            use_cat_names=True,
            handle_unknown="value",
            handle_missing="value",
            return_df=True,
        )

        encoded = encoder.fit_transform(X[[col]])
        self.one_hot_encoders[col] = encoder

        return encoded

    def _transform_one_hot(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        encoder = self.one_hot_encoders[col]
        return encoder.transform(X[[col]])

    def _fit_transform_top_n_one_hot(self, X: pd.DataFrame, col: str, cfg: dict) -> pd.DataFrame:
        top_n = cfg.get("top_n", 10)

        top_categories = X[col].value_counts().head(top_n).index.tolist()
        self.top_n_categories[col] = top_categories

        grouped = X[[col]].copy()
        grouped[col] = grouped[col].where(grouped[col].isin(top_categories), "Other")

        # fit one-hot encoder on the grouped data with lib
        encoder = ce.OneHotEncoder(
            cols=[col],
            use_cat_names=True,
            handle_unknown="value",
            handle_missing="value",
            return_df=True,
        )

        encoded = encoder.fit_transform(grouped)
        self.one_hot_encoders[col] = encoder

        return encoded

    def _transform_top_n_one_hot(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        top_categories = self.top_n_categories[col]

        grouped = X[[col]].copy()
        grouped[col] = grouped[col].where(grouped[col].isin(top_categories), "Other")

        encoder = self.one_hot_encoders[col]
        return encoder.transform(grouped)

    def _fit_transform_loo(self, X: pd.DataFrame, y: pd.Series, col: str, cfg: dict) -> pd.DataFrame:
        sigma = cfg.get("sigma", None)

        encoder = ce.LeaveOneOutEncoder(
            cols=[col],
            sigma=sigma,
            handle_unknown="value",
            handle_missing="value",
            return_df=True,
        )

        encoded = encoder.fit_transform(X[[col]], y)
        encoded.columns = [f"{col}_loo"]

        self.loo_encoders[col] = encoder

        return encoded

    def _transform_loo(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        encoder = self.loo_encoders[col]

        encoded = encoder.transform(X[[col]])
        encoded.columns = [f"{col}_loo"]

        return encoded

    def save_encoder(self, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(self, f)

        print(f"Encoder saved to: {output_path}")

    def export_encoded_splits(self,splitter_target, output_dir: str):
        export_to_csv(
            self.get_encoded_export_objects(),
            target_col=splitter_target,
            output_dir=output_dir)
        
    def get_encoded_export_objects(self):
        return {
            "X_train_encoded": self.X_train_encoded,
            "X_val_encoded": self.X_val_encoded,
            "X_test_encoded": self.X_test_encoded,
            "X_sanity_encoded": self.X_sanity_encoded,
        }
    
class FeatureScaler:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        scaling_config = self.config.get("feature_scaling", {})

        self.scaler_type = scaling_config.get("scaler", "STANDARD")
        self.scale_columns = scaling_config.get("scale_columns", [])

        if self.scaler_type == "STANDARD":
            self.scaler = StandardScaler()
        elif self.scaler_type == "MINMAX":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler must be either STANDARD or MINMAX")

        self.fitted_columns = None

        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
        self.X_sanity_scaled = None

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        X_scaled = X_train.copy()
        self.fitted_columns = X_train.columns.tolist()

        existing_scale_columns = [
            col for col in self.scale_columns if col in X_train.columns
        ]

        missing_scale_columns = [
            col for col in self.scale_columns if col not in X_train.columns
        ]

        if missing_scale_columns:
            print("Warning: scale columns not found and skipped:", missing_scale_columns)

        self.scale_columns = existing_scale_columns

        print("Final columns selected for scaling:")
        for col in self.scale_columns:
            print(" -", col)

        if self.scale_columns:
            X_scaled[self.scale_columns] = self.scaler.fit_transform(
                X_train[self.scale_columns]
            )

        self.X_train_scaled = X_scaled

        print(f"Scaled {len(self.scale_columns)} columns using {self.scaler_type}.")
        print("Scaled columns:", self.scale_columns)

        return X_scaled

    def transform(self, X: pd.DataFrame, split_name: str | None = None) -> pd.DataFrame:
        X = X.reindex(columns=self.fitted_columns, fill_value=0)
        X_scaled = X.copy()

        if self.scale_columns:
            X_scaled[self.scale_columns] = self.scaler.transform(
                X[self.scale_columns]
            )

        if split_name:
            setattr(self, f"X_{split_name}_scaled", X_scaled)

        return X_scaled

    def get_scaled_export_objects(self):
        return {
            "X_train_scaled": self.X_train_scaled,
            "X_val_scaled": self.X_val_scaled,
            "X_test_scaled": self.X_test_scaled,
            "X_sanity_scaled": self.X_sanity_scaled,
        }

    def save_scaler(self, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(self, f)

        print(f"Scaler saved to: {output_path}")

    def export_scaled_splits(self, splitter_target, output_dir: str):
        export_to_csv(
            self.get_scaled_export_objects(),
            target_col=splitter_target,
            output_dir=output_dir)