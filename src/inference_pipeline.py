"""Reusable inference wrapper for raw-input XGBoost predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


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
        X_encoded = self.encoder.transform(X)
        return self.scaler.transform(X_encoded)

    def predict_log(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Return raw model predictions on the log-transformed target scale."""
        X_scaled = self.transform(X_raw)
        return self.model.predict(X_scaled)

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

    inference_pipeline = InferencePipeline(
        encoder=joblib.load(encoder_path),
        scaler=joblib.load(scaler_path),
        model=joblib.load(model_path),
        metadata={
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "encoder_path": str(encoder_path),
            "scaler_path": str(scaler_path),
            "model_path": str(model_path),
            "target_transform": "log1p",
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(inference_pipeline, output_path, compress=3)
    return inference_pipeline


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

    return joblib.load(pipeline_path)
