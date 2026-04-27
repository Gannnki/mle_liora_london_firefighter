"""Run the preprocessing pipeline on truncated LFB datasets."""

import warnings
from pathlib import Path

import pandas as pd
from pyproj import Transformer

from DataPreprocessing import DataPreprocesser
from DataSplitter import DataSplitter
from FeatureEngineering import FeatureEncoder, FeatureScaler
# time tracking for preprocessing
import time

start_time = time.perf_counter()

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR_TRUNCATED = BASE_DIR / "lfb_truncated"
RULES_FILE = BASE_DIR / "utils/rules.xlsx"
DISTANCE_FILE = BASE_DIR / "utils/computed_distance.csv"
FIRESTATION_COORDS_FILE = BASE_DIR / "utils/station_addresses_with_latlong_corrected.csv"
OUTPUT_PATH_MERGED = BASE_DIR / "output/merged_dataset.csv"
OUTPUT_PATH_INTERMEDIATE = BASE_DIR / "output/intermediate_processed_dataset.csv"

# config for encoder
CONFIG_PATH = BASE_DIR / "config/pipeline_config.yaml"
SPLIT_EXPORT_PATH =  BASE_DIR / "output/data_splits"
ENCODER_EXPORT_PATH = BASE_DIR / "output/encoders"
SCALER_EXPORT_PATH = BASE_DIR / "output/scalers"

transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# set pandas display options for better readability
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)

if __name__ == "__main__":
    # read the truncated data from CSVs
    # to generate the truncated data, please run scripts/truncate_dataset_timebase.py
    # load the merged dataset
    merged_df = pd.read_csv(OUTPUT_PATH_MERGED)

    # perform test train validation split and sanity check split
    splitter = DataSplitter(df=merged_df, config_path=CONFIG_PATH, export_path=SPLIT_EXPORT_PATH, flag_export=False)
    splitter.run()

    encoder = FeatureEncoder(config_path="config/pipeline_config.yaml")
    X_train_encoded = encoder.fit_transform(
        splitter.X_train,
        splitter.y_train_log
    )

    # split_name is used to save the encoded splits as attributes in the encoder for later export
    X_val_encoded = encoder.transform(splitter.X_val, split_name="val")
    X_test_encoded = encoder.transform(splitter.X_test, split_name="test")
    X_sanity_encoded = encoder.transform(splitter.X_sanity, split_name="sanity")

    encoder.export_encoded_splits(splitter_target=splitter.target_col, output_dir=ENCODER_EXPORT_PATH)
    encoder.save_encoder("artifacts/encoders/feature_encoder.pkl")

    # scaler
    scaler = FeatureScaler(config_path=CONFIG_PATH)

    scaler.fit_transform(encoder.X_train_encoded)
    scaler.transform(encoder.X_val_encoded, split_name="val")
    scaler.transform(encoder.X_test_encoded, split_name="test")
    scaler.transform(encoder.X_sanity_encoded, split_name="sanity")

    scaler.save_scaler("artifacts/scalers/feature_scaler.pkl")

    scaler.export_scaled_splits(
        splitter_target=splitter.target_col,
        output_dir=SCALER_EXPORT_PATH
    )

    # track time end of preprocessing
    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

    """
    mobilisation_df = pd.read_csv(DATA_DIR_TRUNCATED / "mobilisation_truncated.csv")
    incident_df = pd.read_csv(DATA_DIR_TRUNCATED / "incidents_truncated.csv")

    DataPreprocesser_instance = DataPreprocesser(
        incident_df=incident_df,
        mobilisation_df=mobilisation_df,
        path_firestation_coor=FIRESTATION_COORDS_FILE,
        rules_path=RULES_FILE,
        merged_path=OUTPUT_PATH_MERGED,
        path_intermediate_output=OUTPUT_PATH_INTERMEDIATE,
        path_distance_data=DISTANCE_FILE)

    # target output path : OUTPUT_PATH_MERGED = BASE_DIR / "output/merged_dataset.csv"
    DataPreprocesser_instance.run(export2csv=True)
    # feature engineering and encoding
"""
