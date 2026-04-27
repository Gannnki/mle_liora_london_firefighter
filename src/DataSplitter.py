import pandas as pd
import numpy as np
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
        self.drop_columns = self.config.get("drop_columns", [])

        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.sanity_df = None

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.X_sanity = None

        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.y_sanity = None

        self.y_train_log = None
        self.y_val_log = None
        self.y_test_log = None
        self.y_sanity_log = None

    def run(self):
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
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        splits = self.config["date_splits"]

        self.train_df = self.df[
            (self.df[self.date_col] >= splits["train_start"]) &
            (self.df[self.date_col] <= splits["train_end"])
        ].copy()

        self.val_df = self.df[
            (self.df[self.date_col] >= splits["val_start"]) &
            (self.df[self.date_col] <= splits["val_end"])
        ].copy()

        self.test_df = self.df[
            (self.df[self.date_col] >= splits["test_start"]) &
            (self.df[self.date_col] <= splits["test_end"])
        ].copy()

        self.sanity_df = self.df[
            (self.df[self.date_col] >= splits["sanity_start"]) &
            (self.df[self.date_col] <= splits["sanity_end"])
        ].copy()

        print("Train raw shape:", self.train_df.shape)
        print("Validation raw shape:", self.val_df.shape)
        print("Test raw shape:", self.test_df.shape)
        print("Sanity raw shape:", self.sanity_df.shape)

        self.X_train = self.train_df.drop(columns=[self.target_col, self.date_col])
        self.X_val = self.val_df.drop(columns=[self.target_col, self.date_col])
        self.X_test = self.test_df.drop(columns=[self.target_col, self.date_col])
        self.X_sanity = self.sanity_df.drop(columns=[self.target_col, self.date_col])

        self.y_train = self.train_df[self.target_col]
        self.y_val = self.val_df[self.target_col]
        self.y_test = self.test_df[self.target_col]
        self.y_sanity = self.sanity_df[self.target_col]

    def log_transform_target(self):
        # we use log transform to reduce the skewness of the target variable, which can help improve model performance
        self.y_train_log = np.log1p(self.y_train)
        self.y_val_log = np.log1p(self.y_val)
        self.y_test_log = np.log1p(self.y_test)
        self.y_sanity_log = np.log1p(self.y_sanity)

    def drop_cols_before_split(self):
        existing_cols = [col for col in self.drop_columns if col in self.df.columns]
        self.df.drop(columns=existing_cols, inplace=True)
        print("Dropped columns:", existing_cols)

    def get_split_export_objects(self):
        return {
            "X_train": self.X_train,
            "X_val": self.X_val,
            "X_test": self.X_test,
            "X_sanity": self.X_sanity,
            "y_train": self.y_train,
            "y_val": self.y_val,
            "y_test": self.y_test,
            "y_sanity": self.y_sanity,
            "y_train_log": self.y_train_log,
            "y_val_log": self.y_val_log,
            "y_test_log": self.y_test_log,
            "y_sanity_log": self.y_sanity_log,
        }