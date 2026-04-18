# load the csvs from data folder
import os
import pandas as pd

class DataLoader:
    def __init__(self, data_dir, loader_config=None):
        self.data_dir = data_dir
        self.loader_config = loader_config

    def load_data(self):
        # Implement logic to load CSV files from the data directory
        pass

    def load_csv(self, relative_path: str) -> pd.DataFrame:
        file_path = self.base_dir / relative_path
        return pd.read_csv(file_path)

    def load_excel(self, relative_path: str, sheet_name=0) -> pd.DataFrame:
        file_path = self.base_dir / relative_path
        return pd.read_excel(file_path, sheet_name=sheet_name)
    
    def load_all_csv_in_folder(
        self,
        folder= False,
        recursive: bool = False,
        add_source: bool = True
    ) -> pd.DataFrame:
        """
        Load all CSV files from a folder and concatenate into one DataFrame.

        Args:
            folder (str): subfolder under base_dir (e.g. 'incidents')
            recursive (bool): whether to search subdirectories
            add_source (bool): whether to add source file column

        Returns:
            pd.DataFrame
        """
        if not folder:
            folder_path = self.data_dir
        else:
            folder_path = self.data_dir / folder

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # choose glob method
        pattern = "**/*.csv" if recursive else "*.csv"
        files = sorted(folder_path.glob(pattern))

        if not files:
            raise ValueError(f"No CSV files found in {folder_path}")

        dfs = []

        for file in files:
            try:
                df = pd.read_csv(file)

                if add_source:
                    df["source_file"] = file.name

                dfs.append(df)
                if not self.loader_config:
                    print(f"Loaded: {file.name} | shape={df.shape}")

            except Exception as e:
                print(f"Failed to load {file.name}: {e}")

        # concat
        combined_df = pd.concat(dfs, ignore_index=True)
        if not self.loader_config:
            print("\nFinal combined shape:", combined_df.shape)

        return combined_df