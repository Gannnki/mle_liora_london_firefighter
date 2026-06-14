import pandas as pd
import yaml

from FeatureEngineering import FeatureScaler, should_suppress_missing_feature_warning


def test_optional_station_24h_warning_is_suppressed():
    assert should_suppress_missing_feature_warning("station_prev_24h_incident_count_sum")
    assert not should_suppress_missing_feature_warning("distance_fire_to_station")


def test_feature_scaler_scales_only_configured_columns(tmp_path):
    config_path = tmp_path / "pipeline_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "feature_scaling": {
                    "scaler": "STANDARD",
                    "scale_columns": ["numeric_feature"],
                }
            }
        )
    )

    X_train = pd.DataFrame(
        {
            "numeric_feature": [10.0, 20.0, 30.0],
            "one_hot_feature": [0, 1, 0],
        }
    )

    scaler = FeatureScaler(config_path)
    X_scaled = scaler.fit_transform(X_train)

    assert round(float(X_scaled["numeric_feature"].mean()), 7) == 0.0
    assert X_scaled["one_hot_feature"].tolist() == [0, 1, 0]
