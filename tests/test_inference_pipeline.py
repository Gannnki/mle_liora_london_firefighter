import numpy as np
import pandas as pd
import joblib

from inference_pipeline import InferencePipeline, load_inference_pipeline


class DummyEncoder:
    def transform(self, X):
        return X.assign(encoded_feature=X["raw_feature"] + 1)


class DummyScaler:
    def transform(self, X):
        X_scaled = X.copy()
        X_scaled["encoded_feature"] = X_scaled["encoded_feature"] * 2
        return X_scaled


class DummyModel:
    def predict(self, X):
        return np.log1p(X["encoded_feature"].to_numpy())


def test_inference_pipeline_predicts_on_original_scale():
    pipeline = InferencePipeline(
        encoder=DummyEncoder(),
        scaler=DummyScaler(),
        model=DummyModel(),
    )

    predictions = pipeline.predict(pd.DataFrame({"raw_feature": [2]}))

    np.testing.assert_allclose(predictions, [6])


def test_inference_pipeline_accepts_single_row_dict():
    pipeline = InferencePipeline(
        encoder=DummyEncoder(),
        scaler=DummyScaler(),
        model=DummyModel(),
    )

    prediction_frame = pipeline.predict_frame({"raw_feature": 4})

    np.testing.assert_allclose(prediction_frame.loc[0, "prediction_seconds"], 10)
    np.testing.assert_allclose(prediction_frame.loc[0, "prediction_minutes"], 10 / 60)


def test_load_inference_pipeline_round_trip(tmp_path):
    pipeline = InferencePipeline(
        encoder=DummyEncoder(),
        scaler=DummyScaler(),
        model=DummyModel(),
    )
    pipeline_path = tmp_path / "inference_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)

    loaded = load_inference_pipeline(pipeline_path)

    np.testing.assert_allclose(loaded.predict({"raw_feature": 1}), [4])
