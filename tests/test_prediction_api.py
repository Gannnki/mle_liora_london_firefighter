"""Exercise the real encoder/scaler/XGBoost bundle through the HTTP contract."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from xgboost import XGBRegressor

from src.FeatureEngineering import FeatureEncoder, FeatureScaler
from src.inference_pipeline import build_inference_pipeline, load_inference_pipeline
from src.scenario_features import prepare_scenario
from src.display_streamlit.api import app, get_prediction_service


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    frame = pd.read_csv("src/display_streamlit/models_streamlit/demo_scenarios.csv", index_col=0).head(8)
    template = json.loads(frame.iloc[[0]].to_json(orient="records"))[0]
    settings = dict(month=7, weekday=6, hour=8, incident_group="Fire",
                    special_service_type="NoSpecialService", property_category="Outdoor",
                    property_type="Roadside")
    config = {
        "feature_encoding": {
            "Hour": {"encoding": "CYCLIC"},
            "IncidentGroup": {"encoding": "ONE_HOT"},
            "distance_fire_to_station": {"encoding": "NUMERIC_KEEP"},
            "NumOfCalls_bucket": {"encoding": "NUMERIC_KEEP"},
            "Is_Rush_Hour": {"encoding": "BINARY_KEEP"},
            "high_residual_risk_score": {"encoding": "NUMERIC_KEEP"},
        },
        "feature_scaling": {"scaler": "STANDARD", "scale_columns": ["distance_fire_to_station"]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    encoder = FeatureEncoder(config_path)
    scaler = FeatureScaler(config_path)
    y = np.log1p(np.arange(len(frame)) * 30 + 150)
    matrix = scaler.fit_transform(encoder.fit_transform(frame, pd.Series(y)))
    model = XGBRegressor(n_estimators=5, max_depth=2, n_jobs=1, random_state=42)
    model.fit(matrix.to_numpy(dtype=np.float32), y)
    encoder.save_encoder(tmp_path / "encoder.pkl")
    scaler.save_scaler(tmp_path / "scaler.pkl")
    joblib.dump(model, tmp_path / "model.pkl")
    path = tmp_path / "pipeline.pkl"
    built = build_inference_pipeline(tmp_path / "encoder.pkl", tmp_path / "scaler.pkl",
                                     tmp_path / "model.pkl", path)
    # Saving must not discard the in-memory splits needed by preprocessing exports.
    assert hasattr(encoder, "X_train_encoded")
    assert hasattr(scaler, "X_train_scaled")
    assert not hasattr(built.encoder, "X_train_encoded")
    assert not hasattr(built.scaler, "X_train_scaled")
    monkeypatch.setenv("LFB_MODEL_PATH", str(path))
    get_prediction_service.cache_clear()
    yield path, template, settings, model, scaler, encoder
    get_prediction_service.cache_clear()


def test_api_matches_offline_and_training_transforms(bundle):
    path, template, settings, model, scaler, encoder = bundle
    prepared = prepare_scenario(template, settings)
    expected = np.expm1(model.predict(scaler.transform(encoder.transform(prepared)).to_numpy(dtype=np.float32)))[0]
    offline = load_inference_pipeline(path).predict_scenario(template, settings)[0]
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json={"template": template, **settings})
    assert response.status_code == 200, response.text
    assert response.json()["attendance_time_seconds"] == pytest.approx(float(expected))
    assert offline == pytest.approx(expected)


def test_missing_trained_feature_is_rejected(bundle):
    _, template, settings, *_ = bundle
    del template["distance_fire_to_station"]
    with TestClient(app) as client:
        response = client.post("/predict", json={"template": template, **settings})
    assert response.status_code == 422
    assert "distance_fire_to_station" in response.json()["detail"]


def test_invalid_scenario_control_is_rejected(bundle):
    _, template, settings, *_ = bundle
    with TestClient(app) as client:
        response = client.post("/predict", json={"template": template, **settings, "month": 13})
    assert response.status_code == 422


def test_unavailable_model_is_unhealthy(tmp_path, monkeypatch):
    monkeypatch.setenv("LFB_MODEL_PATH", str(tmp_path / "missing.pkl"))
    get_prediction_service.cache_clear()
    try:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 503
    finally:
        get_prediction_service.cache_clear()


def test_mismatched_bundle_is_rejected(bundle):
    path, *_ = bundle
    pipeline = load_inference_pipeline(path)
    pipeline.scaler.fitted_columns = list(reversed(pipeline.scaler.fitted_columns))
    with pytest.raises(ValueError, match="feature order"):
        pipeline.validate_artifacts()
