"""Data loading helpers for CSV and Excel files."""

from pathlib import Path

import pandas as pd


class DataLoader:
    """Load tabular files from a configured base data directory."""

    def __init__(self, data_dir: str | Path, loader_config: bool | None = None):
        """Initialize the loader with a data directory and optional quiet mode."""
        self.data_dir = Path(data_dir)
        self.base_dir = self.data_dir
        self.loader_config = loader_config

    def load_data(self):
        """Placeholder for project-specific data loading logic."""
        pass

    def load_csv(self, relative_path: str) -> pd.DataFrame:
        """Load a CSV file relative to the base data directory."""
        file_path = self.base_dir / relative_path
        return pd.read_csv(file_path, low_memory=False)

    def load_excel(self, relative_path: str, sheet_name: str | int = 0) -> pd.DataFrame:
        """Load an Excel sheet relative to the base data directory."""
        file_path = self.base_dir / relative_path
        return pd.read_excel(file_path, sheet_name=sheet_name)

    def load_all_csv_in_folder(
        self,
        folder: str | Path | bool = False,
        recursive: bool = False,
        add_source: bool = True,
    ) -> pd.DataFrame:
        """Load all CSV files in a folder and concatenate them into one DataFrame."""
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
                df = pd.read_csv(file, low_memory=False)

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
