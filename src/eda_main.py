"""Run EDA and consistency checks for the London Fire Brigade datasets."""

from io import StringIO
import os
from pathlib import Path
from typing import Literal
import warnings

from ConsistencyChecker import CSVConsistencyChecker
from DataLoader import DataLoader
from DataVizPlotter import EDAVisualizer

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR_INCIDENTS = BASE_DIR / "lfb_converted/incidents"
DATA_DIR_MOBILISATION = BASE_DIR / "lfb_converted/mobilisation"
OUTPUT_DIR_FIGS = BASE_DIR / "output/figures"


def check_consistency(data_dir: Path) -> None:
    """Print and save consistency checks for all CSV files in a directory."""
    checker = CSVConsistencyChecker(data_dir)
    checker.summarize_files()
    checker.check_schema_consistency()
    checker.check_dtype_consistency()
    checker.check_null_ratio()

    checker.write_report()


def load_data() -> tuple:
    """Load incident and mobilisation CSV files into DataFrames."""
    try:
        incident_loader = DataLoader(DATA_DIR_INCIDENTS, loader_config=True)
        incident_df = incident_loader.load_all_csv_in_folder()

        mobilisation_loader = DataLoader(DATA_DIR_MOBILISATION, loader_config=True)
        mobilisation_df = mobilisation_loader.load_all_csv_in_folder()
        return incident_df, mobilisation_df
    except Exception as e:
        print(f"Error while loading data: {e}")
        raise


def eda_overview(
    df,
    name: str = "DataFrame",
    mode: Literal["print_and_save", "print_only", "save_only"] = "print_and_save",
) -> None:
    """Print and/or save a compact exploratory summary for a DataFrame."""
    if mode not in ["print_and_save", "print_only", "save_only"]:
        raise ValueError("Invalid mode. Choose from 'print_and_save', 'print_only', 'save_only'.")

    if mode in ["print_and_save", "print_only"]:
        print(f"--- EDA Overview: {name} ---")
        print("Shape:", df.shape)

        print("\nInfo:")
        df.info()

        print("\nMissing Values:")
        print(df.isnull().sum())

        print("\nDescriptive Statistics:")
        print(df.describe(include="all"))

        print("\nSample Data:")
        print(df.head())

    if mode in ["print_and_save", "save_only"]:
        os.makedirs("output", exist_ok=True)

        with open(f"output/{name}_eda_overview.txt", "w", encoding="utf-8") as f:
            f.write(f"--- EDA Overview: {name} ---\n")
            f.write(f"Shape: {df.shape}\n\n")

            f.write("Info:\n")
            buffer = StringIO()
            df.info(buf=buffer)
            f.write(buffer.getvalue() + "\n")

            # Missing values
            f.write("Missing Values:\n")
            f.write(df.isnull().sum().to_string() + "\n\n")

            # descriptive statistics
            f.write("Descriptive Statistics:\n")
            f.write(df.describe(include="all").to_string() + "\n\n")

            # sample data
            f.write("Sample Data:\n")
            f.write(df.head().to_string() + "\n")


if __name__ == "__main__":
    # check consistency incidents
    # check_consistency(DATA_DIR_INCIDENTS)

    # check consistency mobilisation
    # check_consistency(DATA_DIR_MOBILISATION)

    # load data
    incident_df, mobilisation_df = load_data()

    # EDA overview
    # print and save results to txt files
    print("Performing EDA overview...")
    eda_overview(incident_df, name="Incidents", mode="print_only")
    eda_overview(mobilisation_df, name="Mobilisation", mode="print_only")

    visualizer = EDAVisualizer(OUTPUT_DIR_FIGS)
    visualizer.plot_top_boroughs(
        incident_df,
        "incident_top_boroughs.png",
    )

    visualizer.plot_top_incident_types(
        incident_df,
        "incident_top_incident_types.png",
    )

    visualizer.plot_distribution(
        mobilisation_df,
        "AttendanceTimeSeconds",
        "response_time_hist_mobilisation.png",
    )

    visualizer.plot_distribution(
        incident_df,
        "FirstPumpArriving_AttendanceTime",
        "response_time_hist_incidents.png",
    )

    visualizer.plot_boxplot(
        incident_df,
        "FirstPumpArriving_AttendanceTime",
        "response_time_boxplot_incidents.png",
    )

    core_columns = [
        "AttendanceTimeSeconds",
        "TurnoutTimeSeconds",
        "TravelTimeSeconds",
        "PumpOrder",
        "DelayCodeId",
        "HourOfCall",
    ]
    visualizer.plot_correlation(mobilisation_df[core_columns], "correlation_matrix_mobi.png")

    core_columns_incidents = [
        "FirstPumpArriving_AttendanceTime",
        "NumPumpsAttending",
        "NumStationsWithPumpsAttending",
        "PumpCount",
        "NumCalls",
        "HourOfCall",
        "Easting_rounded",
        "Northing_rounded",
    ]
    visualizer.plot_correlation(
        incident_df[core_columns_incidents],
        "correlation_matrix_incidents.png",
    )
