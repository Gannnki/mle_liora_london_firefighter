from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


DEFAULT_SEQUENCE_FEATURES = [
    "incident_count",
    "mean_attendance_seconds",
    "mean_distance_fire_to_station",
    "repeated_call_rate",
    "special_service_rate",
    "false_alarm_count",
    "fire_count",
    "special_service_count",
    "rush_hour_rate",
    "nightshift_rate",
]


class TemporalSequenceBuilder:
    def __init__(
        self,
        data_path="data/dataset_with_filtered_distance_speed.csv",
        config_path="config/pipeline_config.yaml",
        lookback_hours=24,
        sequence_features=None,
        max_rows=None,
    ):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.lookback_hours = lookback_hours
        self.sequence_features = sequence_features or DEFAULT_SEQUENCE_FEATURES
        self.max_rows = max_rows
        self.scaler = StandardScaler()

        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.year_col = self.config["year_column"]
        self.month_col = self.config["month_column"]

    def build(self):
        df = self._load_data()
        hourly_features = self._build_station_hour_features(df)
        sequence_tensor, station_to_index, hour_to_index = self._build_station_tensor(
            hourly_features
        )

        X_seq, valid_rows = self._make_incident_sequences(
            df,
            sequence_tensor,
            station_to_index,
            hour_to_index,
        )
        y = np.log1p(
            valid_rows[self.config["target_column"]].to_numpy(dtype=np.float32)
        )

        split_dates = self._split_dates(valid_rows)
        X_train, X_val, X_test = self._split_array(X_seq, split_dates)
        y_train, y_val, y_test = self._split_array(y, split_dates)

        X_train, X_val, X_test = self._scale_sequence_features(
            X_train,
            X_val,
            X_test,
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": self.sequence_features,
            "valid_rows": valid_rows,
        }

    def build_temporal_feature_frame(self):
        df = self._load_data()
        hourly_features = self._build_station_hour_features(df)
        sequence_tensor, station_to_index, hour_to_index = self._build_station_tensor(
            hourly_features
        )

        X_seq, valid_rows = self._make_incident_sequences(
            df,
            sequence_tensor,
            station_to_index,
            hour_to_index,
        )
        temporal_features = self._summarize_sequences(X_seq)

        return pd.concat(
            [
                valid_rows.reset_index(drop=True),
                temporal_features,
            ],
            axis=1,
        )

    def _load_data(self):
        df = pd.read_csv(self.data_path, nrows=self.max_rows)
        df["incident_hour"] = self._parse_incident_hour(df)
        df = df.dropna(
            subset=[
                "incident_hour",
                "DeployedFromStation_Name",
                self.config["target_column"],
            ]
        ).copy()
        df["DeployedFromStation_Name"] = (
            df["DeployedFromStation_Name"].astype(str).str.strip().str.upper()
        )
        return df.sort_values("incident_hour").reset_index(drop=True)

    def _summarize_sequences(self, X_seq):
        summary_frames = []

        aggregations = {
            "sum": np.sum(X_seq, axis=1),
            "mean": np.mean(X_seq, axis=1),
            "max": np.max(X_seq, axis=1),
            "std": np.std(X_seq, axis=1),
            "last": X_seq[:, -1, :],
        }

        for aggregation_name, values in aggregations.items():
            columns = [
                f"station_prev_{self.lookback_hours}h_{feature}_{aggregation_name}"
                for feature in self.sequence_features
            ]
            summary_frames.append(
                pd.DataFrame(
                    values,
                    columns=columns,
                )
            )

        return pd.concat(summary_frames, axis=1)

    def _parse_incident_hour(self, df):
        if "DateAndTimeMobilised" in df.columns:
            incident_time = pd.to_datetime(
                df["DateAndTimeMobilised"],
                errors="coerce",
            )
            return incident_time.dt.floor("h")

        incident_date = pd.to_datetime(
            df["IncidentNumber"]
            .astype(str)
            .str.extract(r"-(\d{8})$", expand=False),
            format="%d%m%Y",
            errors="coerce",
        )
        hour_offset = pd.to_timedelta(
            pd.to_numeric(df["Hour"], errors="coerce"),
            unit="h",
        )
        return incident_date + hour_offset

    def _build_station_hour_features(self, df):
        working = df.assign(
            is_false_alarm=df["IncidentGroup"].eq("False Alarm").astype(int),
            is_fire=df["IncidentGroup"].eq("Fire").astype(int),
            is_special_service=df["IncidentGroup"].eq("Special Service").astype(int),
        )

        hourly = (
            working
            .groupby(["DeployedFromStation_Name", "incident_hour"])
            .agg(
                incident_count=(self.config["target_column"], "size"),
                mean_attendance_seconds=(self.config["target_column"], "mean"),
                mean_distance_fire_to_station=("distance_fire_to_station", "mean"),
                repeated_call_rate=("Is_RepeatedCall", "mean"),
                special_service_rate=("Is_SpecialService", "mean"),
                false_alarm_count=("is_false_alarm", "sum"),
                fire_count=("is_fire", "sum"),
                special_service_count=("is_special_service", "sum"),
                rush_hour_rate=("Is_Rush_Hour", "mean"),
                nightshift_rate=("Is_Nightshift", "mean"),
            )
            .reset_index()
        )

        return hourly

    def _build_station_tensor(self, hourly_features):
        stations = sorted(hourly_features["DeployedFromStation_Name"].unique())
        hours = pd.date_range(
            hourly_features["incident_hour"].min(),
            hourly_features["incident_hour"].max(),
            freq="h",
        )

        station_to_index = {station: i for i, station in enumerate(stations)}
        hour_to_index = {hour: i for i, hour in enumerate(hours)}

        tensor = np.zeros(
            (
                len(stations),
                len(hours),
                len(self.sequence_features),
            ),
            dtype=np.float32,
        )

        station_idx = hourly_features["DeployedFromStation_Name"].map(station_to_index)
        hour_idx = hourly_features["incident_hour"].map(hour_to_index)
        feature_values = hourly_features[self.sequence_features].fillna(0).to_numpy(
            dtype=np.float32
        )

        tensor[
            station_idx.to_numpy(),
            hour_idx.to_numpy(),
            :,
        ] = feature_values

        return tensor, station_to_index, hour_to_index

    def _make_incident_sequences(
        self,
        df,
        sequence_tensor,
        station_to_index,
        hour_to_index,
    ):
        station_idx = df["DeployedFromStation_Name"].map(station_to_index)
        hour_idx = df["incident_hour"].map(hour_to_index)
        valid_mask = (
            station_idx.notna()
            & hour_idx.notna()
            & (hour_idx >= self.lookback_hours)
        )

        valid_rows = df.loc[valid_mask].copy()
        station_idx = station_idx.loc[valid_mask].astype(int).to_numpy()
        hour_idx = hour_idx.loc[valid_mask].astype(int).to_numpy()

        X_seq = np.empty(
            (
                len(valid_rows),
                self.lookback_hours,
                len(self.sequence_features),
            ),
            dtype=np.float32,
        )

        for row_idx, (station, hour) in enumerate(zip(station_idx, hour_idx)):
            X_seq[row_idx] = sequence_tensor[
                station,
                hour - self.lookback_hours:hour,
                :,
            ]

        return X_seq, valid_rows

    def _split_dates(self, valid_rows):
        if self.year_col in valid_rows.columns and self.month_col in valid_rows.columns:
            split_date = pd.to_datetime(
                valid_rows[self.year_col].astype(str)
                + "-"
                + valid_rows[self.month_col].astype(str).str.zfill(2)
            )
        else:
            split_date = valid_rows["incident_hour"].dt.to_period("M").dt.to_timestamp()
        splits = self.config["date_splits"]

        return {
            "train": split_date.between(
                pd.to_datetime(splits["train_start"]),
                pd.to_datetime(splits["train_end"]),
                inclusive="both",
            ),
            "val": split_date.between(
                pd.to_datetime(splits["val_start"]),
                pd.to_datetime(splits["val_end"]),
                inclusive="both",
            ),
            "test": split_date.between(
                pd.to_datetime(splits["test_start"]),
                pd.to_datetime(splits["test_end"]),
                inclusive="both",
            ),
        }

    @staticmethod
    def _split_array(array, split_dates):
        return (
            array[split_dates["train"].to_numpy()],
            array[split_dates["val"].to_numpy()],
            array[split_dates["test"].to_numpy()],
        )

    def _scale_sequence_features(self, X_train, X_val, X_test):
        n_features = X_train.shape[-1]
        train_2d = X_train.reshape(-1, n_features)
        self.scaler.fit(train_2d)

        return (
            self.scaler.transform(train_2d).reshape(X_train.shape),
            self._transform_sequence_features(X_val, n_features),
            self._transform_sequence_features(X_test, n_features),
        )

    def _transform_sequence_features(self, X, n_features):
        if X.shape[0] == 0:
            return X

        return self.scaler.transform(X.reshape(-1, n_features)).reshape(X.shape)


def build_lstm_model(
    input_shape,
    lstm_units=64,
    dense_units=32,
    dropout=0.2,
    learning_rate=0.001,
):
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(lstm_units, dropout=dropout),
            keras.layers.Dense(dense_units, activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def fit_lstm_model(
    sequence_data,
    lstm_units=64,
    dense_units=32,
    dropout=0.2,
    learning_rate=0.001,
    epochs=30,
    batch_size=512,
    patience=5,
):
    from tensorflow import keras

    model = build_lstm_model(
        input_shape=sequence_data["X_train"].shape[1:],
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]

    model_history = model.fit(
        sequence_data["X_train"],
        sequence_data["y_train"],
        validation_data=(
            sequence_data["X_val"],
            sequence_data["y_val"],
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    return model, model_history, sequence_data


def train_lstm_from_csv(
    data_path="data/dataset_with_filtered_distance_speed.csv",
    config_path="config/pipeline_config.yaml",
    lookback_hours=24,
    epochs=30,
    batch_size=512,
    max_rows=None,
    lstm_units=64,
    dense_units=32,
    dropout=0.2,
    learning_rate=0.001,
    patience=5,
):
    sequence_data = TemporalSequenceBuilder(
        data_path=data_path,
        config_path=config_path,
        lookback_hours=lookback_hours,
        max_rows=max_rows,
    ).build()

    return fit_lstm_model(
        sequence_data=sequence_data,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout=dropout,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )
