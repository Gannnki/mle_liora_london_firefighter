# main program to run the project
from DataLoader import DataLoader
from ConsistencyChecker import CSVConsistencyChecker

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR_INCIDENTS = BASE_DIR / "lfb_converted/incidents"
DATA_DIR_MOBILISATION = BASE_DIR / "lfb_converted/mobilisation"


def check_consistency(data_dir: Path):
    checker = CSVConsistencyChecker(data_dir)
    # can be printed out to debug
    summary_df = checker.summarize_files()
    checker.check_schema_consistency()
    checker.check_dtype_consistency()
    checker.check_null_ratio()

    # dump results to txt report 
    checker.write_report()

if __name__ == "__main__":
    # data_loader = DataLoader("lfb_converted")
    # data_loader.load_data()
    # check consistency incidents
    check_consistency(DATA_DIR_INCIDENTS)

    # check consistency mobilisation
    check_consistency(DATA_DIR_MOBILISATION)