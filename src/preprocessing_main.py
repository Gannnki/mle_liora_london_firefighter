"""Run the preprocessing pipeline on truncated LFB datasets."""

import warnings
from pathlib import Path

import pandas as pd
from pyproj import Transformer

from DataPreprocessing import DataPreprocesser
# time tracking for preprocessing
import time

start_time = time.perf_counter()

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR_TRUNCATED = BASE_DIR / "lfb_truncated"
RULES_FILE = BASE_DIR / "utils/rules.xlsx"
DISTANCE_FILE = BASE_DIR / "utils/computed_distance.csv"
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
        path_distance_data=DISTANCE_FILE)

    # target output path : OUTPUT_PATH_MERGED = BASE_DIR / "output/merged_dataset.csv"
    DataPreprocesser_instance.run(export2csv=True)
    end_time = time.perf_counter()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

    """
    # test code : 
    df = pd.read_csv(OUTPUT_PATH_MERGED)
    print(df["NumCalls"].value_counts(normalize=True))

    call_dist = pd.crosstab(df["NumCalls"], df["IncidentGroup"], normalize='columns')
    print(call_dist.head(10))


    print(df["IncidentGroup"].value_counts(normalize=True))
    df.groupby("IncidentGroup")["AttendanceTimeSeconds"].describe()

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # cap just for visualization
    viz_df = df.copy()
    upper = viz_df["AttendanceTimeSeconds"].quantile(0.99)
    viz_df["AttendanceTimeViz"] = viz_df["AttendanceTimeSeconds"].clip(upper=upper)

    order = ["False Alarm", "Special Service", "Fire"]

    plt.figure(figsize=(11,6))

    sns.violinplot(
        data=viz_df,
        x="IncidentGroup",
        y="AttendanceTimeViz",
        order=order,
        inner=None,
        cut=0
    )

    sns.boxplot(
        data=viz_df,
        x="IncidentGroup",
        y="AttendanceTimeViz",
        order=order,
        width=0.15,
        showcaps=True,
        boxprops={'facecolor':'none'},
        showfliers=False,
        whiskerprops={'linewidth':1.5}
    )

    # annotate medians
    medians = viz_df.groupby("IncidentGroup")["AttendanceTimeViz"].median().reindex(order)

    for i, med in enumerate(medians):
        plt.text(i, med+15, f"Median={med:.0f}s", ha='center', fontsize=11, fontweight='bold')

    plt.title("Attendance Time Distribution by Incident Group (Capped at 99th Percentile)")
    plt.xlabel("")
    plt.ylabel("Attendance Time (Seconds)")
    plt.show()
    #high_1000 = df[df['AttendanceTimeSeconds'] > 1000]

    #print(df[df['AttendanceTimeSeconds'] > 1000]['AttendanceTimeSeconds'].value_counts().sort_index().tail(20))
    """
