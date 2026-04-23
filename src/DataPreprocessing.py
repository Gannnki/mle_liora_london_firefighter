"""Preprocessing pipeline for London Fire Brigade incident and mobilisation data."""

import os

import pandas as pd
from pyproj import Transformer


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
        df["distance_fire_to_city_center"] = None
        df["avg_speed"] = None
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
        df["is_special_service"] = df["SpecialServiceType"].notna().astype(int)

        # fill missing values (semantic: not special service)
        df["SpecialServiceType"] = df["SpecialServiceType"].fillna("NoSpecialService")
        return df
