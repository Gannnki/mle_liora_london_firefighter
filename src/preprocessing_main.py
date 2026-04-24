"""Run the preprocessing pipeline on truncated LFB datasets."""

import warnings
from pathlib import Path

import pandas as pd
from pyproj import Transformer

from DataPreprocessing import DataPreprocesser

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR_TRUNCATED = BASE_DIR / "lfb_truncated"
RULES_FILE = BASE_DIR / "utils/rules.xlsx"
FIRESTATION_COORDS_FILE = BASE_DIR / "utils/station_addresses_with_latlong_corrected.csv"
OUTPUT_PATH_MERGED = BASE_DIR / "output/merged_dataset.csv"
OUTPUT_PATH_INTERMEDIATE = BASE_DIR / "output/intermediate_processed_dataset.csv"

transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

# set pandas display options for better readability
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_rows", None)


if __name__ == "__main__":
    # read the truncated data from CSVs
    # to generate the truncated data, please run scripts/truncate_dataset_timebase.py
    mobilisation_df = pd.read_csv(DATA_DIR_TRUNCATED / "mobilisation_truncated.csv")
    incident_df = pd.read_csv(DATA_DIR_TRUNCATED / "incidents_truncated.csv")

    DataPreprocesser_instance = DataPreprocesser(
        incident_df=incident_df,
        mobilisation_df=mobilisation_df,
        path_firestation_coor=FIRESTATION_COORDS_FILE,
        rules_path=RULES_FILE,
        merged_path=OUTPUT_PATH_MERGED,
        path_intermediate_output=OUTPUT_PATH_INTERMEDIATE,
    )

    DataPreprocesser_instance.run(export2csv=True)
