# This script will convert xlsx files to csv files, and save them in the same directory as the original xlsx file.
# target dir called lfb_converted

import shutil
from pathlib import Path

import pandas as pd

# Source and target directories
SRC_DIR = Path("lfb_downloads")
DST_DIR = Path("lfb_converted")


def convert_excel_to_csv(src_file: Path, dst_file: Path):
    """Convert Excel file to CSV."""
    try:
        df = pd.read_excel(src_file)
        df.to_csv(dst_file, index=False)
        print(f"Converted: {src_file} -> {dst_file}")
    except Exception as e:
        print(f"Failed to convert {src_file}: {e}")


def copy_csv(src_file: Path, dst_file: Path):
    """Copy CSV file."""
    shutil.copy2(src_file, dst_file)
    print(f"Copied: {src_file} -> {dst_file}")


def process_files():
    for file_path in SRC_DIR.rglob("*"):
        if file_path.is_file():
            # Keep relative path structure
            relative_path = file_path.relative_to(SRC_DIR)
            target_path = DST_DIR / relative_path

            # Ensure target folder exists
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.suffix.lower() == ".csv":
                copy_csv(file_path, target_path)

            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                # Change suffix to .csv
                target_path = target_path.with_suffix(".csv")
                convert_excel_to_csv(file_path, target_path)

            else:
                print(f"Skipped (unsupported): {file_path}")


def main():
    print("Starting conversion...\n")
    process_files()
    print("\nDone.")


if __name__ == "__main__":
    main()