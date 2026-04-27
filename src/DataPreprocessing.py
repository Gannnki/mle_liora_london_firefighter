"""Preprocessing pipeline for London Fire Brigade incident and mobilisation data."""

from calendar import month
import os
import holidays
from dateutil.rrule import weekday
import pandas as pd
from pyproj import Transformer
import matplotlib.pyplot as plt

class DataPreprocesser:
    """Merge raw datasets and prepare engineered features for modelling."""

    def __init__(
        self,
        incident_df: pd.DataFrame,
        mobilisation_df: pd.DataFrame,
        path_firestation_coor: str | os.PathLike,
        rules_path: str | os.PathLike,
        merged_path: str | os.PathLike,
        path_intermediate_output: str | os.PathLike,
        path_distance_data: str | os.PathLike
    ):
        """Initialize the preprocessor with input data and output paths."""
        self.incident_df = incident_df
        self.mobilisation_df = mobilisation_df
        self.transformer = Transformer.from_crs(
            "EPSG:27700",
            "EPSG:4326",
            always_xy=True,
        )
        self.path_firestation_coor = path_firestation_coor
        self.rules_path = rules_path
        self.merged_path = merged_path
        self.intermediate_path = path_intermediate_output
        self.distance_data_path = path_distance_data

    def run(self, export2csv: bool) -> None:
        """Run the full preprocessing pipeline and optionally export the result."""
        merged_df = self.build_merged_dataset()
        df_final = self.post_merge_feature_engineering(merged_df)
        self.df_intermediate = df_final.copy()

        if export2csv:
            df_final.to_csv(self.merged_path, index=False)

    def build_merged_dataset(
        self,
        incident_sheet: str = "Incident",
        mobilisation_sheet: str = "Mobilisation",
        merge_key: str = "IncidentNumber",
    ) -> pd.DataFrame:
        """Apply column-drop rules and merge mobilisation rows with incidents."""
        incident_df = self.incident_df
        mobilisation_df = self.mobilisation_df

        def _get_drop_cols(rule_df: pd.DataFrame, keep_for_merge: list[str]) -> list[str]:
            """Return rule-defined columns to drop, excluding required merge keys."""
            if "Column" not in rule_df.columns or "Action" not in rule_df.columns:
                raise ValueError("Rule sheet must contain 'Column' and 'Action' columns.")

            action_series = rule_df["Action"].astype(str).str.strip().str.lower()
            drop_cols = (
                rule_df.loc[action_series.str.contains("drop", na=False), "Column"]
                .astype(str)
                .tolist()
            )
            drop_cols = [c for c in drop_cols if c not in keep_for_merge]
            return drop_cols

        # Step 1: drop roles based on rules before merging
        incident_rules = pd.read_excel(self.rules_path, sheet_name=incident_sheet)
        mobilisation_rules = pd.read_excel(self.rules_path, sheet_name=mobilisation_sheet)
        keep_for_merge = [merge_key]
        incident_drop_cols = _get_drop_cols(incident_rules, keep_for_merge=keep_for_merge)
        mobilisation_drop_cols = _get_drop_cols(
            mobilisation_rules,
            keep_for_merge=keep_for_merge,
        )
        incident_df = incident_df.drop(
            columns=[c for c in incident_drop_cols if c in incident_df.columns],
            errors="ignore",
        )
        mobilisation_df = mobilisation_df.drop(
            columns=[c for c in mobilisation_drop_cols if c in mobilisation_df.columns],
            errors="ignore",
        )

        # Step 2: Merge
        merged_df = mobilisation_df.merge(
            incident_df,
            on=merge_key,
            how="left",
            validate="many_to_one",
        )

        # Step 3: delete the file source column
        merged_df.drop("source_file_x", axis=1, inplace=True)
        merged_df = merged_df.iloc[:, :-1]

        os.makedirs("output", exist_ok=True)
        merged_df.to_csv(self.merged_path, index=False)
        return merged_df

    def convert_en_to_latlong(
        self,
        easting: float,
        northing: float,
    ) -> tuple[float, float]:
        """Convert British National Grid easting/northing to latitude/longitude."""
        lon, lat = self.transformer.transform(easting, northing)
        return lat, lon

    def convert_EN_to_latlong(
        self,
        Easting: float,
        Northing: float,
    ) -> tuple[float, float]:
        """Convert easting/northing coordinates to latitude/longitude."""
        return self.convert_en_to_latlong(Easting, Northing)

    def post_merge_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean merged data and create placeholder engineered feature columns."""
        # drop duplicates if any
        df = df.drop_duplicates()

        # step: process coordinates
        # target cols: Easting_m, Northing_m, Easting_rounded, Northing_rounded
        df = self.process_coordinates(df)

        # deal with missing values in SpecialServiceType
        df = self.process_special_service_type(df)

        # Missing station names represent unknown stations and cannot be mapped.
        before = len(df)
        df = df.dropna(subset=["DeployedFromStation_Name"])
        after = len(df)
        print(f"Dropped {before - after} rows due to missing station info")

        # Keep only stations that exist in the fire station coordinate lookup.
        station_coords_df = pd.read_csv(self.path_firestation_coor)
        valid_stations = set(station_coords_df["Station Name"].unique())
        valid_stations = set(s.strip().upper() for s in valid_stations if isinstance(s, str))
        df["DeployedFromStation_Name"] = (
            df["DeployedFromStation_Name"].str.strip().str.upper()
        )
        df = df[df["DeployedFromStation_Name"].isin(valid_stations)]

        # deal with trivial missing values in other columns
        trivial_cols = [
            "DateOfCall",
            "IncidentGroup",
            "SpecialServiceType",
            "PropertyCategory",
            "PropertyType",
            "IncGeo_BoroughName",
            "NumCalls",
        ]
        df = df.dropna(subset=trivial_cols)

        # Add placeholders for downstream distance and speed features.
        df["distance_fire_to_station"] = None
        df["avg_speed"] = 50 # assume average speed of 50 km/h for now, to be replaced with actual speed calculated from distance and time later

        # deal with time features - remove and extract
        df = self.process_mobilised_datetime(df)

        # add new time feature columns: Is_Nightshift, Is_Rush_Hour, Is_weekend, is_public_holiday
        df = self.add_new_time_features(df)

        # keep the entries with PumpOrder == 1
        df = df[df["PumpOrder"] == 1].copy()

        # merge distance dataset
        df = self.merge_distance_data(df)

        # deal with outliers in AttendanceTimeSeconds
        df = self.handle_attendance_time_outliers(df)

        # deal with outliers in NumOfCalls and create new features
        df = self.handle_numcall_outliers(df)
        return df

    def process_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create latitude/longitude columns from available easting/northing fields."""
        df["Easting_processed"] = df["Easting_m"].fillna(df["Easting_rounded"])
        df["Northing_processed"] = df["Northing_m"].fillna(df["Northing_rounded"])
        df["Latitude"], df["Longitude"] = zip(
            *df.apply(
                lambda row: self.convert_EN_to_latlong(
                    row["Easting_processed"],
                    row["Northing_processed"],
                ),
                axis=1,
            )
        )

        # drop the Easting and Northing columns after processing
        df.drop(
            columns=[
                "Easting_m",
                "Northing_m",
                "Easting_rounded",
                "Northing_rounded",
                "Easting_processed",
                "Northing_processed",
            ],
            inplace=True,
        )
        return df

    def process_special_service_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a special-service flag and fill missing service categories."""
        # Binary indicator for whether a row has a special service.
        pos_incidentgrp =  df.columns.get_loc("IncidentGroup")
        is_special_service = df["SpecialServiceType"].notna().astype(int)
        df.insert(pos_incidentgrp + 1, "Is_SpecialService", is_special_service)

        mask_inconsistent = (
        (df["IncidentGroup"] == "Special Service") &
        (df["SpecialServiceType"].isna())
        )
        print("Inconsistent rows in SpecialService:", mask_inconsistent.sum())    
        df = df[~mask_inconsistent].copy()

        # fill missing values (semantic: not special service)
        df["SpecialServiceType"] = df["SpecialServiceType"].fillna("NoSpecialService")
        return df
    
    def process_mobilised_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop HourOfCall and DateOfCall,
        then extract month and hour from DateAndTimeMobilised.
        """

        # convert to datetime
        df["DateAndTimeMobilised"] = pd.to_datetime(
            df["DateAndTimeMobilised"],
            format="%d/%m/%Y %H:%M",
            errors="coerce"
        )

        # extract new time features
        hour = df["DateAndTimeMobilised"].dt.hour
        # find position after CalYear
        calyear_pos = df.columns.get_loc("CalYear")

        df["DateOfCall"] = pd.to_datetime(df["DateOfCall"], errors="coerce")
        month = df["DateOfCall"].dt.month
        weekday = df["DateOfCall"].dt.weekday

        # insert new columns
        df.insert(calyear_pos + 1, "Month", month)
        df.insert(calyear_pos + 3, "Hour", hour)
        df.insert(calyear_pos + 2, "Weekday", weekday)

        # drop old redundant columns
        df = df.drop(columns=["HourOfCall", "DateAndTimeMobilised"], errors="ignore")
        return df
    
    def add_new_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based indicator features:
        - is_nightshift: 23:00–06:00
        - is_rush_hour: 07:00–09:00 and 16:00–19:00
        - is_weekend: Saturday/Sunday
        """
        # find position after Hour
        hour_pos = df.columns.get_loc("Hour")
        is_nightshift = ((df["Hour"] >= 23) | (df["Hour"] < 6)).astype(int)

        is_rush_hour = (
            ((df["Hour"] >= 7) & (df["Hour"] <= 9)) |
            ((df["Hour"] >= 16) & (df["Hour"] <= 19))
        ).astype(int)

        is_weekend = (df["Weekday"] >= 5).astype(int)

        uk_holidays = holidays.UnitedKingdom(years=df["CalYear"].unique())
        is_public_holiday = df["DateOfCall"].dt.date.apply(
        lambda x: 1 if x in uk_holidays else 0
    )

        # insert new columns
        df.insert(hour_pos + 1, "Is_Nightshift", is_nightshift)
        df.insert(hour_pos + 2, "Is_Rush_Hour", is_rush_hour)
        df.insert(hour_pos + 3, "Is_Weekend", is_weekend)
        df.insert(hour_pos + 4, "Is_Public_Holiday", is_public_holiday)
        return df
    
    def remove_attendance_time_outliers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unrealistic outliers in AttendanceTimeSeconds.
        """

        # remove impossible low values
        df = df[df["AttendanceTimeSeconds"] >= 10].copy()

        # remove extreme high values using 99.5 percentile
        upper_bound = df["AttendanceTimeSeconds"].quantile(0.995)
        print(f"Upper bound for AttendanceTimeSeconds: {upper_bound:.2f}")

        df = df[df["AttendanceTimeSeconds"] <= upper_bound].copy()
        return df
    
    def merge_distance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        distance_df = pd.read_csv(self.distance_data_path)

        df["IncidentNumber"] = df["IncidentNumber"].astype(str)
        distance_df["IncidentNumber"] = distance_df["IncidentNumber"].astype(str)

        distance_df = distance_df.drop_duplicates(subset=["IncidentNumber"])

        distance_map = distance_df.set_index("IncidentNumber")["Distance_from_First_Station"]
        df["distance_fire_to_station"] = df["IncidentNumber"].map(distance_map)

        print("Missing distance:", df["distance_fire_to_station"].isna().sum())

        # if value missing we add 20 m as a small buffer distance to avoid zero distance which can cause issues in log transformation later
        df["distance_fire_to_station"] = df["distance_fire_to_station"].fillna(20)
        df.loc[df["distance_fire_to_station"] == 0, "distance_fire_to_station"] = 20
        return df
    
    def handle_attendance_time_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unrealistic outliers in AttendanceTimeSeconds. we keep 99 and cap the rest at tail
        """
        df2 = df[df['AttendanceTimeSeconds'] >= 20].copy()

        upper_cap = df2['AttendanceTimeSeconds'].quantile(0.99)

        df2['AttendanceTimeSeconds'] = df2['AttendanceTimeSeconds'].clip(upper=upper_cap)
        return df2
    
    def handle_numcall_outliers(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Preprocess NumOfCalls feature:
        1. Create business-oriented buckets
        2. Create repeated call flag
        3. Create log transformed feature
        
        Parameters
        ----------
        df : pandas.DataFrame
        col : str
            column name of number of calls
            
        Returns
        -------
        df : pandas.DataFrame
            dataframe with new processed columns
        """
        
        df = df.copy()
        col = "NumCalls"
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1)

        def bucket_num_calls(x):
            if x <= 1:
                return "1"
            elif x == 2:
                return "2"
            elif x == 3:
                return "3"
            elif x <= 5:
                return "4-5"
            elif x <= 10:
                return "6-10"
            else:
                return "10+"
        
        df["NumOfCalls_bucket"] = df[col].apply(bucket_num_calls)
        is_repeatedCall = (df[col] > 1).astype(int)

        pos_numcall = df.columns.get_loc(col)
        # insert new columns
        df.insert(pos_numcall + 1, "Is_RepeatedCall", is_repeatedCall)
        return df