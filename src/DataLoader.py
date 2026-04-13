# load the csvs from data folder
import os
import pandas as pd

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self):
        # Implement logic to load CSV files from the data directory
        pass

    def load_csv(self, relative_path: str) -> pd.DataFrame:
        file_path = self.base_dir / relative_path
        return pd.read_csv(file_path)

    def load_excel(self, relative_path: str, sheet_name=0) -> pd.DataFrame:
        file_path = self.base_dir / relative_path
        return pd.read_excel(file_path, sheet_name=sheet_name)