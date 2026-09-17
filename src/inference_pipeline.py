"""Reusable inference wrapper for raw-input XGBoost predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version
import os
import platform
import sys
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.FeatureEngineering import inference_only_copy
from src.scenario_features import prepare_scenario


@dataclass
class InferencePipeline:
    """Bundle the fitted encoder, scaler, and model for production inference."""

    encoder: Any
    scaler: Any
    model: Any
    target_transform: str = "log1p"
    metadata: dict[str, Any] = field(default_factory=dict)

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Apply the training-time encoder and scaler to raw feature rows."""
        X = self._ensure_dataframe(X_raw)
        required = getattr(self.encoder, "input_columns", None)
        if required is None and hasattr(self.encoder, "feature_config"):
            fitted = set(getattr(self.encoder, "fitted_columns", []) or [])
            categorical = set(getattr(self.encoder, "one_hot_encoders", {}))
            categorical.update(getattr(self.encoder, "loo_encoders", {}))
            required = [name for name, cfg in self.encoder.feature_config.items()
                        if name in fitted or name in categorical
                        or (cfg.get("encoding") == "CYCLIC" and f"{name}_sin" in fitted)]
        missing = sorted(set(required or []) - set(X.columns))
        if missing:
            raise ValueError(f"Missing trained input features: {missing}")
        X_encoded = self.encoder.transform(X)
        return self.scaler.transform(X_encoded)

    def predict_log(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Return raw model predictions on the log-transformed target scale."""
        X_scaled = self.transform(X_raw)
        if self.metadata.get("model_input") == "float32_array":
            X_scaled = X_scaled.to_numpy(dtype=np.float32, copy=False)
        return self.model.predict(X_scaled)

    def predict_scenario(self, template: dict, settings: dict) -> np.ndarray:
        """Apply shared scenario features, then the fitted inference transforms."""
        return self.predict(prepare_scenario(template, settings))

    def validate_artifacts(self) -> None:
        """Reject mismatched feature order/count before serving predictions."""
        encoded = getattr(self.encoder, "fitted_columns", None)
        scaled = getattr(self.scaler, "fitted_columns", None)
        if encoded is not None and scaled is not None and list(encoded) != list(scaled):
            raise ValueError("Encoder and scaler feature order differs")
        expected = getattr(self.model, "n_features_in_", None)
        if expected is not None and scaled is not None and expected != len(scaled):
            raise ValueError(f"Model expects {expected} features; scaler has {len(scaled)}")
        if self.target_transform not in {"log1p", "identity"}:
            raise ValueError(f"Unknown target transform: {self.target_transform}")

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Return predictions on the original response-time scale."""
        y_pred = self.predict_log(X_raw)
        if self.target_transform == "log1p":
            return np.expm1(y_pred)
        return y_pred

    def predict_frame(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Return predictions as a DataFrame ready for API or Streamlit display."""
        predictions = self.predict(X_raw)
        return pd.DataFrame(
            {
                "prediction_seconds": predictions,
                "prediction_minutes": predictions / 60,
            }
        )

    @staticmethod
    def _ensure_dataframe(X_raw: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
        if isinstance(X_raw, pd.DataFrame):
            return X_raw.copy()
        if isinstance(X_raw, dict):
            return pd.DataFrame([X_raw])
        raise TypeError("X_raw must be a pandas DataFrame or a single-row dict.")


def build_inference_pipeline(
    encoder_path: Path,
    scaler_path: Path,
    model_path: Path,
    output_path: Path,
) -> InferencePipeline:
    """Load fitted artifacts, wrap them, and save one inference pipeline pickle."""
    missing_paths = [
        path
        for path in [encoder_path, scaler_path, model_path]
        if not path.exists()
    ]
    if missing_paths:
        missing = "\n".join(f" - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Cannot build inference pipeline because required artifacts are missing:\n"
            f"{missing}"
        )

    _register_legacy_modules()
    inference_pipeline = InferencePipeline(
        encoder=inference_only_copy(joblib.load(encoder_path)),
        scaler=inference_only_copy(joblib.load(scaler_path)),
        model=joblib.load(model_path),
        metadata={
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "encoder_path": str(encoder_path),
            "scaler_path": str(scaler_path),
            "model_path": str(model_path),
            "target_transform": "log1p",
            "model_input": "float32_array",
            "python": platform.python_version(),
            "dependencies": {name: version(name) for name in (
                "numpy", "pandas", "pyarrow", "scikit-learn", "category-encoders", "xgboost", "joblib"
            )},
        },
    )
    inference_pipeline.validate_artifacts()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete file, so a concurrent API never reads half a pickle.
    fd, temporary = tempfile.mkstemp(dir=output_path.parent, suffix=".pkl")
    os.close(fd)
    try:
        joblib.dump(inference_pipeline, temporary, compress=3)
        os.replace(temporary, output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return inference_pipeline


def _register_legacy_modules() -> None:
    """Resolve old top-level pickle imports to the canonical training classes."""
    from src import FeatureEngineering
    sys.modules["FeatureEngineering"] = FeatureEngineering


def load_inference_pipeline(
    pipeline_path: Path,
    repo_root: Path | None = None,
) -> InferencePipeline:
    """Load the production pipeline while supporting legacy src-level pickles."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    src_path = repo_root / "src"
    for import_path in [repo_root, src_path]:
        import_path_text = str(import_path)
        if import_path_text not in sys.path:
            sys.path.insert(0, import_path_text)

    _register_legacy_modules()
    pipeline = joblib.load(pipeline_path)
    pipeline.validate_artifacts()
    return pipeline
