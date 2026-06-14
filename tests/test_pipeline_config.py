from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_config_has_required_core_fields():
    config = yaml.safe_load((PROJECT_ROOT / "config" / "pipeline_config.yaml").read_text())

    assert config["target_column"] == "AttendanceTimeSeconds"
    assert config["date_column"] == "DateOfCall"
    assert config["date_splits"]["train_start"] < config["date_splits"]["test_end"]
    assert "feature_encoding" in config
    assert "feature_scaling" in config


def test_removed_loo_features_are_not_scaled():
    config = yaml.safe_load((PROJECT_ROOT / "config" / "pipeline_config.yaml").read_text())
    scale_columns = config["feature_scaling"]["scale_columns"]

    assert not any(column.endswith("_loo") for column in scale_columns)


def test_optional_24h_features_remain_configured_but_optional():
    config = yaml.safe_load((PROJECT_ROOT / "config" / "pipeline_config.yaml").read_text())
    feature_columns = config["feature_encoding"].keys()

    station_24h_features = [
        column for column in feature_columns if column.startswith("station_prev_24h_")
    ]

    assert len(station_24h_features) > 0
