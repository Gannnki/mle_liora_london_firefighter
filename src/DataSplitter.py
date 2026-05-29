import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from helpers.export_helpers import export_to_csv
import yaml

import numpy as np
import pandas as pd
import yaml


class DataSplitter:
    def __init__(self, df: pd.DataFrame, config_path: str, export_path: str, flag_export: bool = False):
        self.df = df.copy()
        self.export_path = export_path
        self.flag_export = flag_export

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.target_col = self.config["target_column"]
        self.date_col = self.config["date_column"]
        self.year_col = self.config["year_column"]
        self.month_col = self.config["month_column"]
        self.drop_columns = self.config.get("drop_columns", [])
        self.station_coordinates_path = self._resolve_config_path(
            config_path,
            self.config.get(
                "station_coordinates_path",
                "utils/station_addresses_with_latlong_corrected.csv",
            ),
        )
        self.borough_intersection_features_path = self._resolve_config_path(
            config_path,
            self.config.get(
                "borough_intersection_features_path",
                "utils/borough_intersection_features.csv",
            ),
        )
        self.intersections_path = self._resolve_config_path(
            config_path,
            self.config.get(
                "intersections_path",
                "utils/london_intersections_27700.gpkg",
            ),
        )

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self.X_train = None
        self.X_val = None
        self.X_test = None

        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.y_train_log = None
        self.y_val_log = None
        self.y_test_log = None

    @staticmethod
    def _resolve_config_path(config_path, configured_path):
        path = Path(configured_path)

        if path.is_absolute():
            return path

        project_root = Path(config_path).resolve().parent.parent
        return project_root / path

    def run(self):
        self.add_new_features()
        self.drop_cols_before_split()
        self.split_by_time()
        # we do log transformation because it has shown a right skewed distribution, and log transform can help reduce the skewness and make the distribution more normal-like, which can improve model performance
        self.log_transform_target()

        print("\nExporting split datasets to CSV...")
        print(self.export_path)

        if self.flag_export:
            export_objects = self.get_split_export_objects()
            export_to_csv(export_objects, self.export_path)

    def split_by_time(self):
        splits = self.config["date_splits"]

        # build datetime column for filtering
        # zfill means to pad the month with leading zeros if it's a single digit, to ensure the format is consistent (e.g., "2021-01" instead of "2021-1")
        self.df["split_date"] = pd.to_datetime(
            self.df[self.year_col].astype(str) + "-" +
            self.df[self.month_col].astype(str).str.zfill(2)
        )

        train_start = pd.to_datetime(splits["train_start"])
        train_end = pd.to_datetime(splits["train_end"])

        val_start = pd.to_datetime(splits["val_start"])
        val_end = pd.to_datetime(splits["val_end"])

        test_start = pd.to_datetime(splits["test_start"])
        test_end = pd.to_datetime(splits["test_end"])

        self.train_df = self.df[
            (self.df["split_date"] >= train_start) &
            (self.df["split_date"] <= train_end)
        ].copy()

        self.val_df = self.df[
            (self.df["split_date"] >= val_start) &
            (self.df["split_date"] <= val_end)
        ].copy()

        self.test_df = self.df[
            (self.df["split_date"] >= test_start) &
            (self.df["split_date"] <= test_end)
        ].copy()

        print("Train raw shape:", self.train_df.shape)
        print("Validation raw shape:", self.val_df.shape)
        print("Test raw shape:", self.test_df.shape)

        self.X_train = self.train_df.drop(columns=[self.target_col, "split_date"])
        self.X_val = self.val_df.drop(columns=[self.target_col, "split_date"])
        self.X_test = self.test_df.drop(columns=[self.target_col, "split_date"])

        self.y_train = self.train_df[self.target_col]
        self.y_val = self.val_df[self.target_col]
        self.y_test = self.test_df[self.target_col]

    def log_transform_target(self):
        # we use log transform to reduce the skewness of the target variable, which can help improve model performance
        self.y_train_log = np.log1p(self.y_train)
        self.y_val_log = np.log1p(self.y_val)
        self.y_test_log = np.log1p(self.y_test)

    def drop_cols_before_split(self):
        existing_cols = [col for col in self.drop_columns if col in self.df.columns]
        self.df.drop(columns=existing_cols, inplace=True)
        print("Dropped columns:", existing_cols)

    def get_split_export_objects(self):
        return {
            "X_train": self.X_train,
            "X_val": self.X_val,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_val": self.y_val,
            "y_test": self.y_test,
            "y_train_log": self.y_train_log,
            "y_val_log": self.y_val_log,
            "y_test_log": self.y_test_log,
        }
    
    def export_encoded_splits(self, splitter_target: str, output_dir: str):
        export_to_csv(self.get_split_export_objects(),
            target_col=splitter_target,
            output_dir=output_dir)

    def add_new_features(self):
        self.add_inner_london_feature()
        self.add_rolling_inertia_features()
        self.add_borough_complexity_features()
        self.add_incident_intersection_density_features()
        self.add_concurrent_same_borough_feature()
        self.add_more_distance_features()
        self.add_property_access_complexity_feature()
        self.add_risk_features()
        self.add_risk_interaction_features()
        self.sort_out_numcalls()
    
    def add_inner_london_feature(self):
        inner_london = [
    'CITY OF LONDON', 'WESTMINSTER', 'CAMDEN', 'ISLINGTON', 'HACKNEY', 
    'TOWER HAMLETS', 'SOUTHWARK', 'LAMBETH', 'KENSINGTON AND CHELSEA', 
    'HAMMERSMITH AND FULHAM', 'WANDSWORTH', 'LEWISHAM', 'NEWHAM', 'HARINGEY'
]
        self.df['Is_central_London'] = self.df['IncGeo_BoroughName'].isin(inner_london).astype(int)

    def add_rolling_inertia_features(self):
        required_columns = [
            "DeployedFromStation_Name",
            "IncGeo_BoroughName",
            self.target_col,
        ]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Cannot add rolling inertia features. Missing columns: "
                f"{missing_columns}"
            )

        incident_time = self._get_incident_time_for_concurrency()
        if incident_time.isna().any():
            missing_count = int(incident_time.isna().sum())
            raise ValueError(
                "Cannot add rolling inertia features because incident time "
                f"has {missing_count:,} missing values."
            )

        print("Adding rolling inertia features...")

        feature_columns = [
            "station_rolling_mean_last_3",
            "borough_rolling_mean_last_3",
            "borough_rolling_max_last_3",
            "station_inertia_rolling_mean_last_3",
            "borough_inertia_rolling_mean_last_3",
            "borough_inertia_rolling_max_last_3",
            "expected_time_by_station_inertia",
            "expected_time_by_borough_inertia",
        ]
        self.df = self.df.drop(columns=feature_columns, errors="ignore")

        working_df = (
            self.df[
                [
                    "DeployedFromStation_Name",
                    "IncGeo_BoroughName",
                    self.target_col,
                ]
            ]
            .assign(
                incident_time=incident_time,
                original_index=self.df.index,
                target_value=pd.to_numeric(
                    self.df[self.target_col],
                    errors="coerce",
                ),
            )
            .sort_values(
                [
                    "DeployedFromStation_Name",
                    "IncGeo_BoroughName",
                    "incident_time",
                    "original_index",
                ]
            )
        )

        time_sorted_target = (
            working_df
            .sort_values(["incident_time", "original_index"])["target_value"]
        )
        global_prior = (
            time_sorted_target
            .shift(1)
            .expanding()
            .mean()
            .reindex(working_df.index)
        )
        global_fallback = working_df["target_value"].median()
        global_prior = global_prior.fillna(global_fallback)

        self._add_group_rolling_features(
            working_df,
            group_col="DeployedFromStation_Name",
            mean_col="station_rolling_mean_last_3",
            inertia_mean_col="station_inertia_rolling_mean_last_3",
            global_prior=global_prior,
        )
        self._add_group_rolling_features(
            working_df,
            group_col="IncGeo_BoroughName",
            mean_col="borough_rolling_mean_last_3",
            max_col="borough_rolling_max_last_3",
            inertia_mean_col="borough_inertia_rolling_mean_last_3",
            inertia_max_col="borough_inertia_rolling_max_last_3",
            global_prior=global_prior,
        )

        working_df["expected_time_by_station_inertia"] = (
            working_df["station_rolling_mean_last_3"]
            + working_df["station_inertia_rolling_mean_last_3"]
        )
        working_df["expected_time_by_borough_inertia"] = (
            working_df["borough_rolling_mean_last_3"]
            + working_df["borough_inertia_rolling_mean_last_3"]
        )

        feature_values = (
            working_df
            .set_index("original_index")[feature_columns]
            .reindex(self.df.index)
        )

        for col in feature_columns:
            if "inertia" in col and not col.startswith("expected_time"):
                feature_values[col] = feature_values[col].fillna(0.0)
            else:
                feature_values[col] = feature_values[col].fillna(global_fallback)

        self.df = self.df.join(feature_values)

    def _add_group_rolling_features(
        self,
        working_df,
        group_col,
        mean_col,
        global_prior,
        max_col=None,
        inertia_mean_col=None,
        inertia_max_col=None,
    ):
        sorted_df = working_df.sort_values(
            [group_col, "incident_time", "original_index"]
        )
        grouped_target = sorted_df.groupby(group_col)["target_value"]
        previous_target = grouped_target.shift(1)
        previous_previous_target = grouped_target.shift(2)
        historical_inertia = previous_target - previous_previous_target

        working_df[mean_col] = (
            previous_target
            .groupby(sorted_df[group_col])
            .transform(lambda series: series.rolling(3, min_periods=1).mean())
            .reindex(working_df.index)
            .fillna(global_prior)
        )

        if max_col is not None:
            working_df[max_col] = (
                previous_target
                .groupby(sorted_df[group_col])
                .transform(lambda series: series.rolling(3, min_periods=1).max())
                .reindex(working_df.index)
                .fillna(global_prior)
            )

        if inertia_mean_col is not None:
            working_df[inertia_mean_col] = (
                historical_inertia
                .groupby(sorted_df[group_col])
                .transform(lambda series: series.rolling(3, min_periods=1).mean())
                .reindex(working_df.index)
                .fillna(0.0)
            )

        if inertia_max_col is not None:
            working_df[inertia_max_col] = (
                historical_inertia
                .groupby(sorted_df[group_col])
                .transform(lambda series: series.rolling(3, min_periods=1).max())
                .reindex(working_df.index)
                .fillna(0.0)
            )

    def add_borough_complexity_features(self):
        if "IncGeo_BoroughName" not in self.df.columns:
            raise ValueError(
                "Cannot add borough complexity features. Missing column: "
                "IncGeo_BoroughName"
            )

        print("Adding borough intersection complexity features...")

        feature_path = self.borough_intersection_features_path
        if not feature_path.exists():
            print(
                "Borough intersection feature file not found; generating it at "
                f"{feature_path}"
            )
            from geo_feature.borough_complexity import build_borough_intersection_features

            build_borough_intersection_features(output_path=feature_path)

        borough_features = pd.read_csv(feature_path)
        required_columns = [
            "IncGeo_BoroughName",
            "borough_intersection_count",
            "borough_intersection_density",
        ]
        missing_columns = [
            col for col in required_columns if col not in borough_features.columns
        ]

        if missing_columns:
            raise ValueError(
                "Borough intersection feature file is missing columns: "
                f"{missing_columns}"
            )

        borough_features = borough_features[required_columns].copy()
        borough_features["IncGeo_BoroughName"] = (
            borough_features["IncGeo_BoroughName"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        original_borough = self.df["IncGeo_BoroughName"].to_numpy()
        self.df = self.df.drop(
            columns=[
                "borough_intersection_count",
                "borough_intersection_density",
            ],
            errors="ignore",
        )
        self.df["IncGeo_BoroughName"] = (
            self.df["IncGeo_BoroughName"].astype(str).str.strip().str.upper()
        )

        self.df = self.df.merge(
            borough_features,
            on="IncGeo_BoroughName",
            how="left",
        )
        self.df["IncGeo_BoroughName"] = original_borough

        self.df["borough_intersection_count"] = (
            self.df["borough_intersection_count"].fillna(0)
        )
        self.df["borough_intersection_density"] = (
            self.df["borough_intersection_density"].fillna(0)
        )

    def add_incident_intersection_density_features(self):
        required_columns = ["Latitude", "Longitude"]
        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Cannot add incident intersection density features. "
                f"Missing columns: {missing_columns}"
            )

        print("Adding incident intersection density features...")

        intersections = self._load_or_build_intersections()
        if intersections.empty:
            for radius_meters in [500, 1000]:
                self.df[f"incident_intersection_count_{radius_meters}m"] = 0
                self.df[f"incident_intersection_density_{radius_meters}m"] = 0.0
            return

        coordinate_features = self._calculate_intersection_counts_by_coordinate(
            intersections=intersections,
            radii_meters=[500, 1000],
        )
        incident_feature_columns = [
            "incident_intersection_count_500m",
            "incident_intersection_density_500m",
            "incident_intersection_count_1000m",
            "incident_intersection_density_1000m",
        ]
        self.df = self.df.drop(columns=incident_feature_columns, errors="ignore")

        latitude = pd.to_numeric(self.df["Latitude"], errors="coerce").round(6)
        longitude = pd.to_numeric(self.df["Longitude"], errors="coerce").round(6)
        row_coordinates = pd.DataFrame(
            {
                "Latitude": latitude,
                "Longitude": longitude,
            },
            index=self.df.index,
        )

        self.df = self.df.join(
            row_coordinates.merge(
                coordinate_features,
                on=["Latitude", "Longitude"],
                how="left",
            )
            .drop(columns=["Latitude", "Longitude"])
            .set_index(self.df.index)
        )

        for radius_meters in [500, 1000]:
            count_col = f"incident_intersection_count_{radius_meters}m"
            density_col = f"incident_intersection_density_{radius_meters}m"
            self.df[count_col] = self.df[count_col].fillna(0).astype(int)
            self.df[density_col] = self.df[density_col].fillna(0.0)

    def _load_or_build_intersections(self):
        import geopandas as gpd

        if self.intersections_path.exists():
            intersections = gpd.read_file(self.intersections_path)
        else:
            print(
                "Intersection cache not found; generating it at "
                f"{self.intersections_path}"
            )
            from geo_feature.borough_complexity import load_intersections

            intersections = load_intersections()
            self.intersections_path.parent.mkdir(parents=True, exist_ok=True)
            intersections.to_file(self.intersections_path, driver="GPKG")

        return intersections.to_crs("EPSG:27700")

    def _calculate_intersection_counts_by_coordinate(self, intersections, radii_meters):
        import geopandas as gpd

        coordinate_df = (
            self.df[["Latitude", "Longitude"]]
            .assign(
                Latitude=pd.to_numeric(self.df["Latitude"], errors="coerce").round(6),
                Longitude=pd.to_numeric(self.df["Longitude"], errors="coerce").round(6),
            )
            .dropna(subset=["Latitude", "Longitude"])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if coordinate_df.empty:
            for radius_meters in radii_meters:
                coordinate_df[f"incident_intersection_count_{radius_meters}m"] = []
                coordinate_df[f"incident_intersection_density_{radius_meters}m"] = []
            return coordinate_df

        points = gpd.GeoDataFrame(
            coordinate_df,
            geometry=gpd.points_from_xy(
                coordinate_df["Longitude"],
                coordinate_df["Latitude"],
            ),
            crs="EPSG:4326",
        ).to_crs("EPSG:27700")

        features = coordinate_df.copy()
        for radius_meters in radii_meters:
            buffers = points.geometry.buffer(radius_meters)
            matches = intersections.sindex.query(buffers, predicate="intersects")
            counts = pd.Series(matches[0]).value_counts(sort=False)

            count_col = f"incident_intersection_count_{radius_meters}m"
            density_col = f"incident_intersection_density_{radius_meters}m"
            features[count_col] = (
                pd.Series(0, index=features.index)
                .add(counts, fill_value=0)
                .astype(int)
            )
            buffer_area_km2 = np.pi * (radius_meters / 1000) ** 2
            features[density_col] = features[count_col] / buffer_area_km2

        return features

    def add_concurrent_same_borough_feature(self):
        required_columns = [
            "IncGeo_BoroughName",
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Cannot add concurrent_same_borough. Missing columns: "
                f"{missing_columns}"
            )

        incident_time = self._get_incident_time_for_concurrency()

        if incident_time.isna().any():
            missing_count = int(incident_time.isna().sum())
            raise ValueError(
                "Cannot add concurrent_same_borough because incident time "
                f"has {missing_count:,} missing values."
            )

        print("Adding concurrent_same_borough feature...")

        working_df = (
            self.df[["IncGeo_BoroughName"]]
            .assign(
                incident_time=incident_time,
                original_index=self.df.index,
                row_count=1,
            )
            .sort_values(["IncGeo_BoroughName", "incident_time"])
        )

        rolling_counts = (
            working_df
            .set_index("incident_time")
            .groupby("IncGeo_BoroughName")["row_count"]
            .rolling("30min")
            .count()
            .reset_index(level=0, drop=True)
            - 1
        )

        working_df["concurrent_same_borough"] = rolling_counts.to_numpy()

        self.df["concurrent_same_borough"] = (
            working_df
            .set_index("original_index")["concurrent_same_borough"]
            .reindex(self.df.index)
            .fillna(0)
            .astype(int)
        )

    def _get_incident_time_for_concurrency(self):
        if "DateAndTimeMobilised" in self.df.columns:
            return pd.to_datetime(
                self.df["DateAndTimeMobilised"],
                errors="coerce",
            )

        if "DateOfCall" in self.df.columns and "Hour" in self.df.columns:
            date_of_call = pd.to_datetime(
                self.df["DateOfCall"],
                errors="coerce",
            )
            hour_offset = pd.to_timedelta(
                pd.to_numeric(
                    self.df["Hour"],
                    errors="coerce",
                ),
                unit="h",
            )

            return date_of_call + hour_offset

        if "IncidentNumber" in self.df.columns and "Hour" in self.df.columns:
            incident_date = pd.to_datetime(
                self.df["IncidentNumber"].astype(str).str.extract(
                    r"-(\d{8})$",
                    expand=False,
                ),
                format="%d%m%Y",
                errors="coerce",
            )
            hour_offset = pd.to_timedelta(
                pd.to_numeric(
                    self.df["Hour"],
                    errors="coerce",
                ),
                unit="h",
            )

            return incident_date + hour_offset

        if "DateOfCall" not in self.df.columns or "Hour" not in self.df.columns:
            raise ValueError(
                "Cannot add concurrent_same_borough. Need either "
                "DateAndTimeMobilised, DateOfCall and Hour, or "
                "IncidentNumber and Hour."
            )
    
    def add_more_distance_features(self):
        # add distance to london center (Charing Cross) as a feature, using the Haversine formula
        # coordinates of Charing Cross  
        print("Adding distance to city center feature...")      
        charing_cross_lat = 51.5074
        charing_cross_lon = -0.1278

        self.df["distance_to_city_center_km"] = haversine_distance(
        self.df["Latitude"],
        self.df["Longitude"],
        charing_cross_lat,
        charing_cross_lon
                            )
        
        print("add distance transformtion features ...")
        self.df["distance_sqrt"] = np.sqrt(self.df["distance_fire_to_station"])
        self.df["distance_squared"] = self.df["distance_fire_to_station"] ** 2
        self.df["distance_bin"] = pd.cut(
            self.df["distance_fire_to_station"],
            bins=[0, 500, 1000, 2000, 3000, 5000, np.inf],
            labels=[
                "0-500m",
                "500m-1km",
                "1-2km",
                "2-3km",
                "3-5km",
                "5km+",
            ],
            include_lowest=True,
        )
        self.add_area_distance_bin_feature()
        self.add_detour_ratio_feature()

    def add_area_distance_bin_feature(self):
        print("Adding area distance bin feature...")

        distance = self.df["distance_fire_to_station"]
        is_center = self.df["Is_central_London"].eq(1)

        conditions = [
            is_center & distance.between(0, 500, inclusive="both"),
            is_center & distance.gt(500) & distance.le(1000),
            is_center & distance.gt(1000) & distance.le(2000),
            ~is_center & distance.gt(3000) & distance.le(4000),
            ~is_center & distance.gt(4000) & distance.le(6000),
            ~is_center & distance.gt(6000),
        ]
        labels = [
            "center_0_500",
            "center_500_1000",
            "center_1000_2000",
            "outer_3000_4000",
            "outer_4000_6000",
            "outer_6000_plus",
        ]

        self.df["area_distance_bin"] = np.select(
            conditions,
            labels,
            default="area_distance_other",
        )

    def add_detour_ratio_feature(self):
        required_columns = [
            "DeployedFromStation_Name",
            "Latitude",
            "Longitude",
            "distance_fire_to_station",
        ]
        missing_columns = [
            col
            for col in required_columns
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Cannot add detour_ratio. Missing columns: "
                f"{missing_columns}"
            )

        print("Adding detour ratio feature...")

        station_coords = pd.read_csv(self.station_coordinates_path)
        station_coords["station_key"] = (
            station_coords["Station Name"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        station_lat_map = station_coords.set_index("station_key")["lat"]
        station_lon_map = station_coords.set_index("station_key")["long"]
        station_key = (
            self.df["DeployedFromStation_Name"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        station_latitude = station_key.map(station_lat_map)
        station_longitude = station_key.map(station_lon_map)

        missing_station_coords = station_latitude.isna() | station_longitude.isna()
        if missing_station_coords.any():
            unknown_stations = sorted(
                self.df.loc[
                    missing_station_coords,
                    "DeployedFromStation_Name",
                ].astype(str).unique()
            )
            raise ValueError(
                "Cannot add detour_ratio. Missing station coordinates for: "
                f"{unknown_stations}"
            )

        straight_line_distance_m = haversine_distance(
            station_latitude,
            station_longitude,
            self.df["Latitude"],
            self.df["Longitude"],
        ) * 1000

        self.df["detour_ratio"] = (
            self.df["distance_fire_to_station"]
            / straight_line_distance_m.replace(0, np.nan)
        )
        self.df["detour_ratio"] = self.df["detour_ratio"].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        self.df["detour_ratio"] = self.df["detour_ratio"].fillna(1.0)

    def add_property_access_complexity_feature(self):
        if "PropertyType" not in self.df.columns:
            raise ValueError(
                "Cannot add property_access_complexity. Missing column: "
                "PropertyType"
            )

        print("Adding property access complexity feature...")

        self.df["property_access_complexity"] = (
            self.df["PropertyType"]
            .str.contains(
                "Flat|Maisonette|Care|Hospital|School|Sheltered|Estate",
                case=False,
                na=False,
            )
            .astype(int)
        )

    def add_risk_features(self):
        base_features = [
            "PropertyCategory",
            "NumOfCalls_bucket",
            "Is_SpecialService",
            "IncidentGroup",
            "Is_central_London",
            "Weekday",
            "Is_RepeatedCall",
            "Month",
            "Is_Nightshift",
            "Is_Weekend",
        ]

        missing_features = [
            col
            for col in base_features
            if col not in self.df.columns
        ]

        if missing_features:
            raise ValueError(
                "Cannot add risk features. Missing columns: "
                f"{missing_features}"
            )

        print("Adding high residual risk features...")

        self.df["risk_property_outdoor"] = (
            self.df["PropertyCategory"].eq("Outdoor")
        ).astype(int)

        self.df["risk_property_road_vehicle"] = (
            self.df["PropertyCategory"].eq("Road Vehicle")
        ).astype(int)

        self.df["risk_property_outdoor_structure"] = (
            self.df["PropertyCategory"].eq("Outdoor Structure")
        ).astype(int)

        numcalls_ord = self._numcalls_bucket_to_ordinal(
            self.df["NumOfCalls_bucket"]
        )

        self.df["NumOfCalls_ord"] = numcalls_ord
        self.df["NumOfCalls_log"] = np.log1p(numcalls_ord)

        self.df["risk_many_calls"] = (
            self.df["NumOfCalls_ord"] >= 3
        ).astype(int)

        self.df["risk_very_many_calls"] = (
            self.df["NumOfCalls_ord"] >= 12
        ).astype(int)

        self.df["risk_special_service"] = (
            (self.df["Is_SpecialService"] == 1)
            | (self.df["IncidentGroup"].eq("Special Service"))
        ).astype(int)

        self.df["risk_fire"] = (
            self.df["IncidentGroup"].eq("Fire")
        ).astype(int)

        self.df["risk_noncentral"] = (
            self.df["Is_central_London"] == 0
        ).astype(int)

        self.df["risk_repeated_call"] = (
            self.df["Is_RepeatedCall"] == 1
        ).astype(int)

        self.df["risk_weekday_4"] = (
            self.df["Weekday"] == 4
        ).astype(int)

        self.df["risk_weekday_2"] = (
            self.df["Weekday"] == 2
        ).astype(int)

        self.df["risk_month_3_5_6"] = (
            self.df["Month"].isin([3, 5, 6])
        ).astype(int)

        self.df["risk_not_nightshift"] = (
            self.df["Is_Nightshift"] == 0
        ).astype(int)

        self.df["risk_not_weekend"] = (
            self.df["Is_Weekend"] == 0
        ).astype(int)

        risk_cols = [
            "risk_property_outdoor",
            "risk_property_road_vehicle",
            "risk_property_outdoor_structure",
            "risk_many_calls",
            "risk_very_many_calls",
            "risk_special_service",
            "risk_fire",
            "risk_noncentral",
            "risk_repeated_call",
            "risk_weekday_4",
            "risk_weekday_2",
            "risk_month_3_5_6",
            "risk_not_nightshift",
            "risk_not_weekend",
        ]

        self.df["high_residual_risk_score"] = self.df[risk_cols].sum(axis=1)

    def add_risk_interaction_features(self):
        interaction_base_features = [
            "risk_many_calls",
            "risk_property_outdoor",
            "risk_property_road_vehicle",
            "risk_special_service",
            "risk_noncentral",
            "risk_repeated_call",
            "risk_fire",
        ]

        missing_features = [
            col
            for col in interaction_base_features
            if col not in self.df.columns
        ]

        if missing_features:
            raise ValueError(
                "Cannot add risk interaction features. Missing columns: "
                f"{missing_features}"
            )

        print("Adding risk interaction features...")

        self.df["many_calls_x_outdoor"] = (
            self.df["risk_many_calls"] * self.df["risk_property_outdoor"]
        )

        self.df["many_calls_x_road_vehicle"] = (
            self.df["risk_many_calls"] * self.df["risk_property_road_vehicle"]
        )

        self.df["many_calls_x_special"] = (
            self.df["risk_many_calls"] * self.df["risk_special_service"]
        )

        self.df["many_calls_x_noncentral"] = (
            self.df["risk_many_calls"] * self.df["risk_noncentral"]
        )

        self.df["road_vehicle_x_noncentral"] = (
            self.df["risk_property_road_vehicle"] * self.df["risk_noncentral"]
        )

        self.df["outdoor_x_noncentral"] = (
            self.df["risk_property_outdoor"] * self.df["risk_noncentral"]
        )

        self.df["repeated_x_many_calls"] = (
            self.df["risk_repeated_call"] * self.df["risk_many_calls"]
        )

        self.df["fire_x_many_calls"] = (
            self.df["risk_fire"] * self.df["risk_many_calls"]
        )

    def _numcalls_bucket_to_ordinal(self, series):
        bucket_map = {
            "0": 0.0,
            "1": 1.0,
            "2": 2.0,
            "3": 3.0,
            "4-5": 4.5,
            "6-10": 8.0,
            "10+": 12.0,
        }

        numeric_values = pd.to_numeric(
            series,
            errors="coerce",
        )

        mapped_values = (
            series
            .astype(str)
            .str.strip()
            .map(bucket_map)
        )

        return mapped_values.fillna(numeric_values).fillna(0.0)

    def sort_out_numcalls(self):
        if "NumOfCalls_bucket" not in self.df.columns:
            print("NumOfCalls_bucket column not found; skipping mapping.")
            return

        numcalls_mapping = {
            "0": 0.0,
            "1": 1.0,
            "2": 2.0,
            "3": 3.0,
            "4-5": 4.5,
            "6-10": 8.0,
            "10+": 12.0,
            1: 1.0,
            2: 2.0,
            3: 3.0,
        }

        cleaned_bucket = self.df["NumOfCalls_bucket"].astype(str).str.strip()
        mapped_bucket = cleaned_bucket.map(numcalls_mapping)

        missing_mapping = (
            self.df["NumOfCalls_bucket"].notna()
            & mapped_bucket.isna()
        )

        if missing_mapping.any():
            unknown_values = sorted(
                self.df.loc[
                    missing_mapping,
                    "NumOfCalls_bucket",
                ].astype(str).unique()
            )
            raise ValueError(
                "Unknown NumOfCalls_bucket values: "
                f"{unknown_values}"
            )

        self.df["NumOfCalls_bucket"] = mapped_bucket
        print("Mapped NumOfCalls_bucket to numeric order values.")


        def add_inner_london_feature(self):
            pass

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two lat/lon points in kilometers.
    Works with scalars, pandas Series, or numpy arrays.
    """
    R = 6371  # Earth radius in kilometers

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c
