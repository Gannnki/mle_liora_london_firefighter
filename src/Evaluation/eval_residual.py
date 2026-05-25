# residual analysis


# PATH of important files
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PATH_y_pred = BASE_DIR / "output/predictions/y_pred_test.csv"
PATH_X_test = BASE_DIR / "output/data_splits/X_test.csv"

PATH_error_output = BASE_DIR / "output/predictions/residual_analysis.csv"

def analysis_on_station(error_df):
        # analyse on stations 
    station_summary = (
        error_df
        .groupby("DeployedFromStation_Name")
        .agg(
            residual_mean=("residual", "mean"),
            residual_median=("residual", "median"),
            abs_error_mean=("abs_error", "mean"),
            abs_error_median=("abs_error", "median"),
            residual_std=("residual", "std"),
            count=("residual", "size")
        )
        .reset_index()
    )

    print("\n=== Station Count Distribution ===")
    print(f"Total stations: {len(station_summary)}")
    print(f"Max count: {station_summary['count'].max()}")
    print(f"Mean count: {station_summary['count'].mean():.0f}")
    print(f"Median count: {station_summary['count'].median():.0f}")

    # Filter by a more reasonable threshold
    min_count_threshold = max(100, station_summary['count'].median() * 0.5)
    station_summary_filtered = station_summary[station_summary["count"] >= min_count_threshold]

    print(f"\n=== Filtering by count >= {min_count_threshold:.0f} ===")
    print(f"Filtered stations: {len(station_summary_filtered)}")

    print("\n=== Top 20 stations by Absolute Error (Mean) ===")
    print(station_summary_filtered.sort_values("abs_error_mean", ascending=False).head(20))

    print("\n=== Top 20 stations with POSITIVE residual bias (Overestimation) ===")
    print(station_summary_filtered.sort_values("residual_mean", ascending=False).head(20))

    print("\n=== Top 20 stations with NEGATIVE residual bias (Underestimation) ===")
    print(station_summary_filtered.sort_values("residual_mean", ascending=True).head(20))

def analysis_on_borough(error_df):
    pass

def analysis_on_target_bins(error_df):
    print("bin analysis:")
    min_actual = error_df["actual"].min()
    # Use 0 as the lower bound for the bins if actual values are non-negative,
    # otherwise use the true minimum value.
    lower_bound = 0 if min_actual >= 0 else min_actual
    upper_bound = error_df["actual"].max()
    bins = np.linspace(lower_bound, upper_bound, 11)
    error_df["actual_bin"] = pd.cut(error_df["actual"], bins=bins, include_lowest=True)

    bin_summary = (
        error_df
        .groupby("actual_bin")
        .agg(
            actual_mean=("actual", "mean"),
            predicted_mean=("prediction", "mean"),
            residual_mean=("residual", "mean"),
            residual_median=("residual", "median"),
            abs_error_mean=("absolute_error", "mean"),
            abs_error_median=("absolute_error", "median"),
            count=("actual", "size")
        )
        .reset_index()
    )

    print(bin_summary)

def output_residual_analysis():

    PATH_error_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    error_df.to_csv(
        PATH_error_output,
        index=False,
    )

    print(f"\nSaved residual analysis to: {PATH_error_output}")


def visualize_residuals(error_df):
    
    # visualization: residual distribution (positive vs negative)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of residuals
    axes[0, 0].hist(error_df["residual"], bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero residual")
    axes[0, 0].set_xlabel("Residual (Actual - Predicted)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Residuals")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Count of positive vs negative residuals
    positive_count = (error_df["residual"] > 0).sum()
    negative_count = (error_df["residual"] < 0).sum()
    zero_count = (error_df["residual"] == 0).sum()
    counts = [positive_count, negative_count, zero_count]
    labels = [f"Positive\n({positive_count})", f"Negative\n({negative_count})", f"Zero\n({zero_count})"]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    axes[0, 1].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0, 1].set_title("Residual Sign Distribution")

    # 3. Box plot of residuals
    axes[1, 0].boxplot(error_df["residual"], vert=True)
    axes[1, 0].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[1, 0].set_ylabel("Residual (Actual - Predicted)")
    axes[1, 0].set_title("Box Plot of Residuals")
    axes[1, 0].grid(alpha=0.3)

    # 4. Residuals vs Predicted values
    axes[1, 1].scatter(error_df["prediction"], error_df["residual"], alpha=0.5, s=20)
    axes[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[1, 1].set_xlabel("Predicted Values")
    axes[1, 1].set_ylabel("Residual (Actual - Predicted)")
    axes[1, 1].set_title("Residuals vs Predicted Values")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PATH_error_output.parent / "residual_visualization.png", dpi=300, bbox_inches="tight")
    print(f"Saved residual visualization to: {PATH_error_output.parent / 'residual_visualization.png'}")
    plt.close()

    # Additional diagnostic plots
    from scipy import stats

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Q-Q plot (test for normality)
    stats.probplot(error_df["residual"], dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title("Q-Q Plot (Normality Test)")
    axes[0, 0].grid(alpha=0.3)

    # 2. Histogram of absolute errors
    axes[0, 1].hist(error_df["absolute_error"], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 1].axvline(error_df["absolute_error"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {error_df['absolute_error'].mean():.2f}")
    axes[0, 1].axvline(error_df["absolute_error"].median(), color="green", linestyle="--", linewidth=2, label=f"Median: {error_df['absolute_error'].median():.2f}")
    axes[0, 1].set_xlabel("Absolute Error")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Absolute Errors")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # 3. Residuals vs actual values
    axes[1, 0].scatter(error_df["actual"], error_df["residual"], alpha=0.5, s=20, color="purple")
    axes[1, 0].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[1, 0].set_xlabel("Actual Values")
    axes[1, 0].set_ylabel("Residual (Actual - Predicted)")
    axes[1, 0].set_title("Residuals vs Actual Values")
    axes[1, 0].grid(alpha=0.3)

    # 4. Histogram of relative errors (where actual != 0)
    valid_relative_errors = error_df["relative_error"].dropna()
    if len(valid_relative_errors) > 0:
        axes[1, 1].hist(valid_relative_errors, bins=50, edgecolor="black", alpha=0.7, color="coral")
        axes[1, 1].set_xlabel("Relative Error (% of actual)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Distribution of Relative Errors")
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "No relative errors to plot", ha="center", va="center")
        axes[1, 1].set_title("Relative Error Distribution")

    plt.tight_layout()
    plt.savefig(PATH_error_output.parent / "residual_diagnostics.png", dpi=300, bbox_inches="tight")
    print(f"Saved diagnostic plots to: {PATH_error_output.parent / 'residual_diagnostics.png'}")
    plt.close()

    # Additional plot: Residuals by borough (if borough data exists)
    if "IncGeo_BoroughName" in error_df.columns:
        boroughs = error_df["IncGeo_BoroughName"].unique()
        if len(boroughs) > 0:
            fig, ax = plt.subplots(figsize=(14, 6))
            error_df.boxplot(column="residual", by="IncGeo_BoroughName", ax=ax)
            ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
            ax.set_xlabel("Borough")
            ax.set_ylabel("Residual (Actual - Predicted)")
            ax.set_title("Residuals Distribution by Borough")
            plt.suptitle("")  # Remove the automatic title
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(PATH_error_output.parent / "residual_by_borough.png", dpi=300, bbox_inches="tight")
            print(f"Saved borough residuals plot to: {PATH_error_output.parent / 'residual_by_borough.png'}")
            plt.close()

    # Additional plot: Residuals by station (if station data exists)
    if "DeployedFromStation_Name" in error_df.columns:
        stations = error_df["DeployedFromStation_Name"].unique()
        if len(stations) > 0:
            fig, ax = plt.subplots(figsize=(14, 6))
            error_df.boxplot(column="residual", by="DeployedFromStation_Name", ax=ax)
            ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
            ax.set_xlabel("Deployed From Station")
            ax.set_ylabel("Residual (Actual - Predicted)")
            ax.set_title("Residuals Distribution by Deployed From Station")
            plt.suptitle("")  # Remove the automatic title
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(PATH_error_output.parent / "residual_by_station.png", dpi=300, bbox_inches="tight")
            print(f"Saved station residuals plot to: {PATH_error_output.parent / 'residual_by_station.png'}")
            plt.close()

    # Additional plots: Y_true vs Y_pred analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Histogram of y_true (actual values)
    axes[0].hist(error_df["actual"], bins=50, edgecolor="black", alpha=0.7, color="teal")
    axes[0].axvline(error_df["actual"].mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {error_df['actual'].mean():.2f}")
    axes[0].axvline(error_df["actual"].median(), color="green", linestyle="--", linewidth=2, label=f"Median: {error_df['actual'].median():.2f}")
    axes[0].set_xlabel("Actual Values (Y_true)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Actual Values (Y_true)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Y_true vs Y_pred scatter plot with perfect prediction line
    axes[1].scatter(error_df["actual"], error_df["prediction"], alpha=0.5, s=20, color="steelblue")
    # Add perfect prediction line (y=x)
    min_val = min(0, error_df["actual"].min(), error_df["prediction"].min())
    max_val = max(error_df["actual"].max(), error_df["prediction"].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Perfect prediction (y=x)")
    axes[1].set_xlim(min_val, max_val)
    axes[1].set_ylim(min_val, max_val)
    axes[1].set_xlabel("Actual Values (Y_true)")
    axes[1].set_ylabel("Predicted Values (Y_pred)")
    axes[1].set_title("Actual vs Predicted Values")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PATH_error_output.parent / "y_true_vs_y_pred.png", dpi=300, bbox_inches="tight")
    print(f"Saved Y_true vs Y_pred plot to: {PATH_error_output.parent / 'y_true_vs_y_pred.png'}")
    plt.close()

    # Print residual statistics
    print(f"\n=== Residual Statistics ===")
    print(f"Positive residuals: {positive_count} ({100*positive_count/len(error_df):.2f}%)")
    print(f"Negative residuals: {negative_count} ({100*negative_count/len(error_df):.2f}%)")
    print(f"Zero residuals: {zero_count} ({100*zero_count/len(error_df):.2f}%)")
    print(f"\nMean residual: {error_df['residual'].mean():.4f}")
    print(f"Median residual: {error_df['residual'].median():.4f}")
    print(f"Std of residuals: {error_df['residual'].std():.4f}")
    print(f"Min residual: {error_df['residual'].min():.4f}")
    print(f"Max residual: {error_df['residual'].max():.4f}")

    output_residual_analysis()

# load files
X_test = pd.read_csv(PATH_X_test)
y_pred = pd.read_csv(PATH_y_pred)

required_columns = {"actual", "prediction"}
missing_columns = required_columns - set(y_pred.columns)

if missing_columns:
    raise ValueError(
        f"{PATH_y_pred} is missing columns: {sorted(missing_columns)}"
    )

if len(X_test) != len(y_pred):
    raise ValueError(
        "X_test and prediction CSV have different row counts: "
        f"{len(X_test)} != {len(y_pred)}"
    )

# distribution of absolute errors
error_df = X_test.reset_index(drop=True).copy()
error_df["actual"] = y_pred["actual"].to_numpy()
error_df["prediction"] = y_pred["prediction"].to_numpy()
error_df["absolute_error"] = (error_df["actual"] - error_df["prediction"]).abs()
error_df["relative_error"] = np.where(
    error_df["actual"] != 0,
    error_df["absolute_error"] / error_df["actual"],
    np.nan,
)

# calculate residuals (not absolute)
error_df["residual"] = error_df["actual"] - error_df["prediction"]

# add plot to visualize residual distribution and error distribution
# Additional diagnostic plots will be created after error statistics

# check percentage of positive vs negative residuals
positive_residuals_per = (error_df["residual"] > 0).sum() / len(error_df) * 100
negative_residuals_per = (error_df["residual"] < 0).sum() / len(error_df) * 100
zero_residuals_per = (error_df["residual"] == 0).sum() / len(error_df) * 100

print("statistics of residuals:")
print(f"Positive residuals: {positive_residuals_per:.2f}%")
print(f"Negative residuals: {negative_residuals_per:.2f}%")
print(f"Zero residuals: {zero_residuals_per:.2f}%")

visualize_residuals(error_df)


error_df["abs_error"] = abs(error_df["actual"] - error_df["prediction"])
top_100_errors = error_df.sort_values("abs_error", ascending=False).head(100)
top_300_errors = error_df.sort_values("abs_error", ascending=False).head(10000)
# analyze top 100 errors
print("\nTop 300 errors:")
print("underestimation count:", (top_300_errors["residual"] < 0).sum())
print("overestimation count:", (top_300_errors["residual"] > 0).sum())

# print(top_300_errors[["actual", "prediction", "residual", "CalYear", "Is_Rush_Hour", "IncidentGroup"]].to_string(index=False))
# top_300_errors.to_csv(PATH_error_output.parent / "top_300_errors.csv", index=False)
analysis_on_target_bins(error_df)
