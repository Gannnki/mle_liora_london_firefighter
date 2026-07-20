"""Model-loading and prediction logic shared by the FastAPI application."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]

# The persisted encoder/scaler reference ``FeatureEngineering`` as a top-level
# module, so that module must be importable before joblib restores the objects.
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from FeatureEngineering import FeatureEncoder, FeatureScaler  # noqa: E402,F401


class ArtifactError(RuntimeError):
    """Raised when production artifacts cannot be loaded consistently."""


def _first_existing(paths: list[Path]) -> Path:
    path = next((candidate for candidate in paths if candidate.exists()), None)
    if path is None:
        raise ArtifactError("Artifact not found; checked: " + ", ".join(map(str, paths)))
    return path


class PredictionService:
    """Load the trained pipeline once and execute predictions."""

    def __init__(self) -> None:
        self.model = joblib.load(_first_existing([
            APP_DIR / "models_streamlit" / "best_model.pkl",
            REPO_ROOT / "artifacts" / "best_models" / "best_model.pkl",
        ]))
        self.scaler = joblib.load(_first_existing([
            APP_DIR / "models_streamlit" / "feature_scaler.pkl",
            REPO_ROOT / "artifacts" / "scalers" / "feature_scaler.pkl",
        ]))
        self.encoder = joblib.load(_first_existing([
            APP_DIR / "models_streamlit" / "feature_encoder.pkl",
            REPO_ROOT / "artifacts" / "encoders" / "feature_encoder.pkl",
        ]))

        expected = self._model_feature_count()
        scaler_count = len(getattr(self.scaler, "fitted_columns", []) or [])
        if expected is not None and scaler_count and expected != scaler_count:
            raise ArtifactError(
                f"Incompatible artifacts: model expects {expected} features, "
                f"but scaler contains {scaler_count}."
            )

    @staticmethod
    def _calls_ordinal(series: pd.Series) -> pd.Series:
        mapping = {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0,
                   "4-5": 4.5, "6-10": 8.0, "10+": 12.0}
        numeric = pd.to_numeric(series, errors="coerce")
        mapped = series.astype(str).str.strip().map(mapping)
        return mapped.fillna(numeric).fillna(0.0)

    def _model_feature_count(self) -> int | None:
        for attr in ("n_features_in_", "n_features_"):
            value = getattr(self.model, attr, None)
            if value is not None:
                return int(value)
        if hasattr(self.model, "get_booster"):
            try:
                return int(self.model.get_booster().num_features())
            except Exception:
                return None
        return None

    def _prepare(self, template: dict[str, Any], settings: dict[str, Any]) -> pd.DataFrame:
        x = pd.DataFrame([template])
        x = x.drop(columns=["Selector_Label"], errors="ignore")
        x["Month"] = settings["month"]
        x["Weekday"] = settings["weekday"]
        x["Hour"] = settings["hour"]
        x["IncidentGroup"] = settings["incident_group"]
        x["SpecialServiceType"] = settings["special_service_type"]
        x["PropertyCategory"] = settings["property_category"]
        x["PropertyType"] = settings["property_type"]

        hour, weekday = settings["hour"], settings["weekday"]
        x["Is_Nightshift"] = int(hour >= 23 or hour < 6)
        x["Is_Rush_Hour"] = int(((7 <= hour <= 9) or (16 <= hour <= 19)) and weekday < 5)
        x["Is_Weekend"] = int(weekday >= 5)
        x["Is_SpecialService"] = int(settings["incident_group"] == "Special Service")
        x["property_access_complexity"] = x["PropertyType"].str.contains(
            "Flat|Maisonette|Care|Hospital|School|Sheltered|Estate", case=False, na=False
        ).astype(int)

        x["risk_property_outdoor"] = x["PropertyCategory"].eq("Outdoor").astype(int)
        x["risk_property_road_vehicle"] = x["PropertyCategory"].eq("Road Vehicle").astype(int)
        x["risk_property_outdoor_structure"] = x["PropertyCategory"].eq("Outdoor Structure").astype(int)
        calls = self._calls_ordinal(x["NumOfCalls_bucket"])
        x["NumOfCalls_ord"] = calls
        x["NumOfCalls_log"] = np.log1p(calls)
        x["risk_many_calls"] = (calls >= 3).astype(int)
        x["risk_very_many_calls"] = (calls >= 12).astype(int)
        x["risk_special_service"] = ((x["Is_SpecialService"] == 1) | x["IncidentGroup"].eq("Special Service")).astype(int)
        x["risk_fire"] = x["IncidentGroup"].eq("Fire").astype(int)
        x["risk_noncentral"] = (pd.to_numeric(x["Is_central_London"]) == 0).astype(int)
        x["risk_repeated_call"] = (pd.to_numeric(x["Is_RepeatedCall"]) == 1).astype(int)
        x["risk_weekday_4"] = (x["Weekday"] == 4).astype(int)
        x["risk_weekday_2"] = (x["Weekday"] == 2).astype(int)
        x["risk_month_3_5_6"] = x["Month"].isin([3, 5, 6]).astype(int)
        x["risk_not_nightshift"] = (x["Is_Nightshift"] == 0).astype(int)
        x["risk_not_weekend"] = (x["Is_Weekend"] == 0).astype(int)

        risk_columns = [column for column in x.columns if column.startswith("risk_")]
        x["high_residual_risk_score"] = x[risk_columns].sum(axis=1)
        for output, left, right in [
            ("many_calls_x_outdoor", "risk_many_calls", "risk_property_outdoor"),
            ("many_calls_x_road_vehicle", "risk_many_calls", "risk_property_road_vehicle"),
            ("many_calls_x_special", "risk_many_calls", "risk_special_service"),
            ("many_calls_x_noncentral", "risk_many_calls", "risk_noncentral"),
            ("road_vehicle_x_noncentral", "risk_property_road_vehicle", "risk_noncentral"),
            ("outdoor_x_noncentral", "risk_property_outdoor", "risk_noncentral"),
            ("repeated_x_many_calls", "risk_repeated_call", "risk_many_calls"),
            ("fire_x_many_calls", "risk_fire", "risk_many_calls"),
        ]:
            x[output] = x[left] * x[right]

        for column, config in self.encoder.feature_config.items():
            if column not in x:
                x[column] = 0 if config.get("encoding") in {"NUMERIC_KEEP", "BINARY_KEEP", "CYCLIC"} else "Unknown"
        return x.reset_index(drop=True)

    def predict(self, template: dict[str, Any], settings: dict[str, Any]) -> dict[str, float | int]:
        prepared = self._prepare(template, settings)
        encoded = self.encoder.transform(prepared)
        scaled = self.scaler.transform(encoded)
        expected_features = getattr(self.scaler, "fitted_columns", None)
        if expected_features and hasattr(scaled, "reindex"):
            scaled = scaled.reindex(columns=expected_features, fill_value=0)
        matrix = scaled.to_numpy(dtype=np.float32, copy=False)
        expected_count = self._model_feature_count()
        if expected_count is not None and matrix.shape[1] != expected_count:
            raise ValueError(f"Model expects {expected_count} features; pipeline produced {matrix.shape[1]}.")

        started = time.perf_counter()
        raw = float(np.asarray(self.model.predict(matrix)).reshape(-1)[0])
        inference_ms = (time.perf_counter() - started) * 1000
        seconds = max(0.0, float(np.expm1(raw)))
        return {
            "attendance_time_seconds": seconds,
            "minutes": int(seconds // 60),
            "remaining_seconds": int(seconds % 60),
            "inference_ms": inference_ms,
        }
