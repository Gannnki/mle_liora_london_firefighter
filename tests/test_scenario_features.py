import pandas as pd
import pytest

from DataSplitter import DataSplitter
from src.scenario_features import add_time_features, calls_ordinal, prepare_scenario


def test_weekend_rush_hour_matches_training_definition():
    frame = pd.DataFrame({"Hour": [5, 8, 12, 18, 23], "Weekday": [6] * 5})
    result = add_time_features(frame)
    assert result["Is_Rush_Hour"].tolist() == [0, 1, 0, 1, 0]
    assert result["Is_Nightshift"].tolist() == [1, 0, 0, 0, 1]
    assert result["Is_Weekend"].tolist() == [1] * 5


def test_training_and_scenario_risks_match_and_ignore_stale_extra_flags():
    template = dict(PropertyCategory="Outdoor", PropertyType="Flat", NumOfCalls_bucket="4-5",
                    Is_central_London=0, Is_RepeatedCall=1, risk_unused=999)
    settings = dict(month=3, weekday=2, hour=8, incident_group="Fire",
                    special_service_type="NoSpecialService", property_category="Outdoor",
                    property_type="Flat")
    prepared = prepare_scenario(template, settings)
    splitter = DataSplitter.__new__(DataSplitter)
    splitter.df = prepared.drop(columns=[c for c in prepared if c.startswith("risk_")])
    splitter.add_property_access_complexity_feature()
    splitter.add_risk_features()
    splitter.add_risk_interaction_features()
    splitter.sort_out_numcalls()
    for column in ["high_residual_risk_score", "many_calls_x_outdoor", "NumOfCalls_bucket"]:
        assert splitter.df[column].iloc[0] == prepared[column].iloc[0]
    assert prepared["NumOfCalls_bucket"].iloc[0] == 4.5
    assert prepared["many_calls_x_outdoor"].iloc[0] == 1
    assert prepared["high_residual_risk_score"].iloc[0] < 20


def test_invalid_call_bucket_is_rejected():
    with pytest.raises(ValueError, match="NumOfCalls_bucket"):
        calls_ordinal(pd.Series(["not-a-bucket"]))
