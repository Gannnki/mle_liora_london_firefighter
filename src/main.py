# main program to run the project
from DataLoader import DataLoader
from ConsistencyChecker import CSVConsistencyChecker
from DataVizPlotter import EDAVisualizer
from pathlib import Path
import warnings
from io import StringIO
import os

# ignore warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR_INCIDENTS = BASE_DIR / "lfb_converted/incidents"
DATA_DIR_MOBILISATION = BASE_DIR / "lfb_converted/mobilisation"
OUTPUT_DIR_FIGS = BASE_DIR / "output/figures"


def check_consistency(data_dir: Path):
    checker = CSVConsistencyChecker(data_dir)
    # can be printed out to debug
    summary_df = checker.summarize_files()
    checker.check_schema_consistency()
    checker.check_dtype_consistency()
    checker.check_null_ratio()

    # dump results to txt report 
    checker.write_report()

def load_data():
    try:
        incident_loader = DataLoader(DATA_DIR_INCIDENTS, loader_config=True)
        incident_df = incident_loader.load_all_csv_in_folder()

        mobilisation_loader = DataLoader(DATA_DIR_MOBILISATION, loader_config=True)
        mobilisation_df = mobilisation_loader.load_all_csv_in_folder()
        return incident_df, mobilisation_df
    except Exception as e:
        print(f"Error while loading data: {e}")
        raise


def eda_overview(df, name="DataFrame", mode="print_and_save"):
    if mode not in ["print_and_save", "print_only", "save_only"]:
        raise ValueError("Invalid mode. Choose from 'print_and_save', 'print_only', 'save_only'.")

    # ---------- PRINT ----------
    if mode in ["print_and_save", "print_only"]:
        print(f"--- EDA Overview: {name} ---")
        print("Shape:", df.shape)

        print("\nInfo:")
        df.info()  

        print("\nMissing Values:")
        print(df.isnull().sum())

        print("\nDescriptive Statistics:")
        print(df.describe(include='all'))

        print("\nSample Data:")
        print(df.head())

    # ---------- SAVE ----------
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
            f.write(df.describe(include='all').to_string() + "\n\n")

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
    "incident_top_boroughs.png"
)
    
    visualizer.plot_top_incident_types(
    incident_df,
    "incident_top_incident_types.png"
)
    
    visualizer.plot_distribution(
    mobilisation_df,
    "AttendanceTimeSeconds",
    "response_time_hist_mobilisation.png"
)
    
    visualizer.plot_distribution(
    incident_df,
    "FirstPumpArriving_AttendanceTime",
    "response_time_hist_incidents.png"
)
    
    visualizer.plot_boxplot(
    incident_df,
    "FirstPumpArriving_AttendanceTime",
    "response_time_boxplot_incidents.png"
)
    
    #  "AttendanceTimeSeconds" in mobilisation dataset
    core_columns = ["AttendanceTimeSeconds", "TurnoutTimeSeconds", "TravelTimeSeconds", "PumpOrder", "DelayCodeId", "HourOfCall"]
    visualizer.plot_correlation(mobilisation_df[core_columns], "correlation_matrix_mobi.png")

    #  "AttendanceTimeSeconds" in mobilisation dataset
    # avoid categorical cols
    core_columns_incidents = [
    "FirstPumpArriving_AttendanceTime",
    "NumPumpsAttending",
    "NumStationsWithPumpsAttending",
    "PumpCount",
    "NumCalls",
    "HourOfCall",
    "Easting_rounded",
    "Northing_rounded"
]
    visualizer.plot_correlation(incident_df[core_columns_incidents], "correlation_matrix_incidents.png")
    