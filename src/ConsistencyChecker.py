"""Utilities for inspecting consistency across converted CSV datasets."""

from pathlib import Path
from typing import Any

import pandas as pd


class CSVConsistencyChecker:
    """Check schemas, dtypes, null counts, and summary statistics for CSV files."""

    def __init__(self, folder_path: str | Path):
        """Initialize the checker with the folder containing CSV files."""
        self.folder_path = Path(folder_path)
        print("Current working dir:", Path.cwd())

    def get_csv_files(self) -> list[Path]:
        """Return all CSV files directly inside the configured folder."""
        csv_files = sorted(self.folder_path.glob("*.csv"))
        print("Found files:", csv_files)
        return csv_files

    def inspect_file(self, file_path: Path) -> dict[str, Any]:
        """Read a CSV file and return its core structure and quality metadata."""
        df = pd.read_csv(file_path)

        return {
            "file_name": file_path.name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
        }

    def check_schema_consistency(self) -> None:
        """Print whether each CSV file has the same column order as the first file."""
        files = self.get_csv_files()
        if not files:
            print("No CSV files found.")
            return

        base_info = self.inspect_file(files[0])
        base_columns = base_info["columns"]

        print(f"Base file: {files[0].name}")
        print(f"Base columns: {base_columns}\n")

        for file_path in files[1:]:
            info = self.inspect_file(file_path)

            if info["columns"] != base_columns:
                print(f"[SCHEMA MISMATCH] {file_path.name}")
                print(f"Columns: {info['columns']}\n")
            else:
                print(f"[OK] Schema matches: {file_path.name}")

    def check_dtype_consistency(self) -> None:
        """Print dtype mismatches compared with the first CSV file."""
        files = self.get_csv_files()
        if not files:
            print("No CSV files found.")
            return

        base_info = self.inspect_file(files[0])
        base_dtypes = base_info["dtypes"]

        print("\nChecking dtypes...\n")

        for file_path in files[1:]:
            info = self.inspect_file(file_path)

            mismatches = []
            for col, dtype in info["dtypes"].items():
                base_dtype = base_dtypes.get(col)
                if base_dtype != dtype:
                    mismatches.append((col, base_dtype, dtype))

            if mismatches:
                print(f"[DTYPE MISMATCH] {file_path.name}")
                for col, base_dtype, dtype in mismatches:
                    print(f"  - Column: {col}, base={base_dtype}, current={dtype}")
                print()
            else:
                print(f"[OK] Dtypes match: {file_path.name}")

    def summarize_files(self) -> pd.DataFrame:
        """Return a one-row-per-file summary with row and column counts."""
        rows = []
        for file_path in self.get_csv_files():
            info = self.inspect_file(file_path)
            rows.append(
                {
                    "file_name": info["file_name"],
                    "row_count": info["row_count"],
                    "column_count": info["column_count"],
                }
            )
        return pd.DataFrame(rows)

    def check_null_ratio(self) -> None:
        """Print the percentage of missing values for columns with any nulls."""
        print("\nChecking null ratios...\n")
        for file_path in self.get_csv_files():
            df = pd.read_csv(file_path)
            null_ratio = (df.isnull().mean() * 100).round(2)

            print(f"{file_path.name}")
            print(null_ratio[null_ratio > 0])
            print("-" * 40)

    def write_report(self, output_dir: str | Path = "output") -> None:
        """Write a text report with schema, dtype, and null-count details."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # use the input folder name as part of the report name
        report_name = f"{self.folder_path.name}.txt"
        report_path = output_dir / report_name

        files = self.get_csv_files()

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("CSV Consistency Report\n")
            f.write(f"Folder: {self.folder_path.resolve()}\n")
            f.write(f"Total files: {len(files)}\n")
            f.write("=" * 50 + "\n\n")

            for file_path in files:
                try:
                    info = self.inspect_file(file_path)

                    f.write(f"File: {info['file_name']}\n")
                    f.write(f"Rows: {info['row_count']}\n")
                    f.write(f"Columns: {info['column_count']}\n")

                    f.write("Column Names:\n")
                    for col in info["columns"]:
                        f.write(f"  - {col}\n")

                    f.write("Dtypes:\n")
                    for col, dtype in info["dtypes"].items():
                        f.write(f"  - {col}: {dtype}\n")

                    f.write("Null Counts (non-zero only):\n")
                    for col, null_count in info["null_counts"].items():
                        if null_count > 0:
                            f.write(f"  - {col}: {null_count}\n")

                    f.write("-" * 50 + "\n\n")

                except Exception as e:
                    f.write(f"Failed to process {file_path.name}: {e}\n")
                    f.write("-" * 50 + "\n\n")

        print(f"Report written to: {report_path.resolve()}")
