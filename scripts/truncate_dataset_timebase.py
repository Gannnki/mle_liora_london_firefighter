
import warnings
from pathlib import Path
import os
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.DataLoader import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR_INCIDENTS = BASE_DIR / "lfb_converted/incident"
DATA_DIR_MOBILISATION = BASE_DIR / "lfb_converted/mobilisation"
DATA_DIR_TRUNCATED = BASE_DIR / "lfb_truncated"
YEAR_THRESHOLD_LOWER = 2021
YEAR_THRESHOLD_UPPER = 2025


def load_data():
    try:
        incident_loader = DataLoader(DATA_DIR_INCIDENTS, loader_config=True)
        incident_df = incident_loader.load_all_csv_in_folder()

        mobilisation_loader = DataLoader(DATA_DIR_MOBILISATION, loader_config=True)
        mobilisation_df = mobilisation_loader.load_all_csv_in_folder()
        return incident_df, mobilisation_df
    except Exception as e:
        print(f"Error while loading data: {e}")
        raise UnboundLocalError
    
def data_truncation(incident_df, mobilisation_df):
    incident_df, mobilisation_df = load_data()
    print("Data loaded successfully.")
    incident_truncated = incident_df[incident_df['CalYear'] >= YEAR_THRESHOLD_LOWER].copy()
    mobilisation_truncated = mobilisation_df[mobilisation_df['CalYear'] >= YEAR_THRESHOLD_LOWER].copy()
    
    print('Export both as csv for preprocessing step')
    # make new dir if not exist
    os.makedirs("lfb_truncated", exist_ok=True)
    mobilisation_truncated.to_csv(BASE_DIR / "lfb_truncated/mobilisation_truncated.csv", index=False)
    incident_truncated.to_csv(BASE_DIR / "lfb_truncated/incidents_truncated.csv", index=False)


if __name__ == "__main__":
    # please convert your xlsx into csvs 
    # otherwise u can convert with the convert2csv.py
    incident_df, mobilisation_df = load_data()
    data_truncation(incident_df=incident_df, mobilisation_df=mobilisation_df)
