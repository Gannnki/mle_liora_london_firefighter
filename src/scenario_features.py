"""Stateless features shared by training and historical scenario inference.

Geography and historical workload are supplied by the input snapshot. Changing
scenario controls does not manufacture new road distances or historical data.
"""

import numpy as np
import pandas as pd


def calls_ordinal(series: pd.Series) -> pd.Series:
    mapping = {"0": 0.0, "1": 1.0, "2": 2.0, "3": 3.0,
               "4-5": 4.5, "6-10": 8.0, "10+": 12.0}
    values = series.astype(str).str.strip().map(mapping)
    values = values.fillna(pd.to_numeric(series, errors="coerce"))
    if (series.notna() & values.isna()).any() or (values.dropna() < 0).any():
        raise ValueError("NumOfCalls_bucket must contain known buckets or nonnegative numbers")
    return values.fillna(0.0)


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["Is_Nightshift"] = ((x["Hour"] >= 23) | (x["Hour"] < 6)).astype(int)
    # Preserve the training definition: these hour ranges also apply on weekends.
    x["Is_Rush_Hour"] = (x["Hour"].between(7, 9) | x["Hour"].between(16, 19)).astype(int)
    x["Is_Weekend"] = (x["Weekday"] >= 5).astype(int)
    return x


def add_property_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["property_access_complexity"] = x["PropertyType"].str.contains(
        "Flat|Maisonette|Care|Hospital|School|Sheltered|Estate", case=False, na=False
    ).astype(int)
    return x


def add_risk_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    calls = calls_ordinal(x["NumOfCalls_bucket"])
    x["NumOfCalls_ord"] = calls
    x["NumOfCalls_log"] = np.log1p(calls)
    flags = {
        "risk_property_outdoor": x["PropertyCategory"].eq("Outdoor"),
        "risk_property_road_vehicle": x["PropertyCategory"].eq("Road Vehicle"),
        "risk_property_outdoor_structure": x["PropertyCategory"].eq("Outdoor Structure"),
        "risk_many_calls": calls >= 3,
        "risk_very_many_calls": calls >= 12,
        "risk_special_service": (x["Is_SpecialService"] == 1) | x["IncidentGroup"].eq("Special Service"),
        "risk_fire": x["IncidentGroup"].eq("Fire"),
        "risk_noncentral": pd.to_numeric(x["Is_central_London"]) == 0,
        "risk_repeated_call": pd.to_numeric(x["Is_RepeatedCall"]) == 1,
        "risk_weekday_4": x["Weekday"] == 4,
        "risk_weekday_2": x["Weekday"] == 2,
        "risk_month_3_5_6": x["Month"].isin([3, 5, 6]),
        "risk_not_nightshift": x["Is_Nightshift"] == 0,
        "risk_not_weekend": x["Is_Weekend"] == 0,
    }
    for column, values in flags.items():
        x[column] = values.astype(int)
    x["high_residual_risk_score"] = x[list(flags)].sum(axis=1)
    return x


def add_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
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
    return x


def prepare_scenario(template: dict, settings: dict) -> pd.DataFrame:
    x = pd.DataFrame([template]).drop(columns=["Selector_Label"], errors="ignore")
    columns = {"month": "Month", "weekday": "Weekday", "hour": "Hour",
               "incident_group": "IncidentGroup", "special_service_type": "SpecialServiceType",
               "property_category": "PropertyCategory", "property_type": "PropertyType"}
    for key, column in columns.items():
        x[column] = settings[key]
    x = add_time_features(x)
    x["Is_SpecialService"] = x["IncidentGroup"].eq("Special Service").astype(int)
    x = add_interaction_features(add_risk_features(add_property_features(x)))
    x["NumOfCalls_bucket"] = calls_ordinal(x["NumOfCalls_bucket"])
    return x.reset_index(drop=True)
