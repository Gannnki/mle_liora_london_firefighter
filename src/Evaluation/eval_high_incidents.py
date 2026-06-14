"""Post-training analysis of high-response-time and high-residual incidents."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def visualization():
        # draw histogram for high response times
    plt.figure(figsize=(10, 6))
    sns.histplot(df_high_response_times[RESPONSE_TIME_COLUMN], bins=50, kde=True)
    plt.axvline(
        df_high_response_times[RESPONSE_TIME_COLUMN].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df_high_response_times[RESPONSE_TIME_COLUMN].mean():.2f}",
    )
    plt.axvline(
        df_high_response_times[RESPONSE_TIME_COLUMN].median(),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {df_high_response_times[RESPONSE_TIME_COLUMN].median():.2f}",
    )
    plt.title("Histogram of High Response Times")
    plt.xlabel("Response Time (seconds)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    histogram_path = PATH_OUTPUT_DIR / "high_response_time_histogram.png"
    plt.savefig(histogram_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram to: {histogram_path}")

    # draw a correlation heatmap for the high response times
    numeric_high_response_times = df_high_response_times.select_dtypes(include="number")
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_high_response_times.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap for High Response Times")
    plt.tight_layout()
    heatmap_path = PATH_OUTPUT_DIR / "high_response_time_correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to: {heatmap_path}")

def visual_residual():

    # Dense residual plot: hexbin shows point concentration instead of overplotting.
    fig, ax = plt.subplots(figsize=(10, 6))
    hexbin = ax.hexbin(
        df_high_response_times["predicted_response_time"],
        df_high_response_times["residual"],
        gridsize=45,
        mincnt=1,
        cmap="viridis",
        bins="log",
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_title("Residuals vs Predicted Response Time (High Incidents)")
    ax.set_xlabel("Predicted Response Time (seconds)")
    ax.set_ylabel("Residual (Actual - Predicted, seconds)")
    ax.grid(alpha=0.3)
    colorbar = fig.colorbar(hexbin, ax=ax)
    colorbar.set_label("Log count")
    plt.tight_layout()
    residual_hexbin_path = PATH_OUTPUT_DIR / "high_response_time_residuals_hexbin.png"
    plt.savefig(residual_hexbin_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved dense residual plot to: {residual_hexbin_path}")

    # Lighter scatter view: sample points and use high transparency.
    scatter_data = df_high_response_times.sample(
        min(MAX_SCATTER_POINTS, len(df_high_response_times)),
        random_state=42,
    )

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=scatter_data,
        x="predicted_response_time",
        y="residual",
        s=14,
        alpha=0.25,
        edgecolor=None,
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.title(f"Sampled Residuals (n={len(scatter_data):,})")
    plt.xlabel("Predicted Response Time (seconds)")
    plt.ylabel("Residual (Actual - Predicted, seconds)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    residual_sample_path = PATH_OUTPUT_DIR / "high_response_time_residuals_sample.png"
    plt.savefig(residual_sample_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved sampled residual plot to: {residual_sample_path}")

    # Distribution view: easier to read the residual bias and spread.
    plt.figure(figsize=(10, 6))
    sns.histplot(
        df_high_response_times["residual"],
        bins=60,
        kde=True,
        color="steelblue",
    )
    plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero residual")
    plt.axvline(
        df_high_response_times["residual"].mean(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df_high_response_times['residual'].mean():.2f}",
    )
    plt.title("Residual Distribution (High Incidents)")
    plt.xlabel("Residual (Actual - Predicted, seconds)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    residual_hist_path = PATH_OUTPUT_DIR / "high_response_time_residual_histogram.png"
    plt.savefig(residual_hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residual histogram to: {residual_hist_path}")


def analyze_lift_for_high_response_and_residuals(df):
    actual_threshold = df[RESPONSE_TIME_COLUMN].quantile(HIGH_RESPONSE_QUANTILE)
    residual_threshold = df["residual"].quantile(HIGH_RESPONSE_QUANTILE)
    target_mask = (
        (df[RESPONSE_TIME_COLUMN] > actual_threshold)
        & (df["residual"] > residual_threshold)
    )
    baseline_rate = target_mask.mean()

    print("\nLift target definition:")
    print(f"{RESPONSE_TIME_COLUMN} > {actual_threshold:.2f}")
    print(f"residual > {residual_threshold:.2f}")
    print(
        "Target rows: "
        f"{target_mask.sum():,} / {len(df):,} "
        f"({baseline_rate * 100:.2f}%)"
    )

    categorical_lift = _calculate_categorical_lift(
        df,
        target_mask,
        baseline_rate,
    )

    numeric_differences = _calculate_numeric_differences(
        df,
        target_mask,
    )

    categorical_lift_path = PATH_OUTPUT_DIR / "high_response_residual_categorical_lift.csv"
    numeric_differences_path = PATH_OUTPUT_DIR / "high_response_residual_numeric_differences.csv"

    categorical_lift.to_csv(
        categorical_lift_path,
        index=False,
    )
    numeric_differences.to_csv(
        numeric_differences_path,
        index=False,
    )

    print(f"Saved categorical lift table to: {categorical_lift_path}")
    print(f"Saved numeric difference table to: {numeric_differences_path}")

    print("\nTop categorical lift values:")
    print(categorical_lift.head(20).to_string(index=False))

    print("\nTop numeric differences:")
    print(numeric_differences.head(15).to_string(index=False))

    _plot_top_categorical_lift(categorical_lift)
    _plot_top_numeric_differences(numeric_differences)

    return categorical_lift, numeric_differences


def _calculate_categorical_lift(
    df,
    target_mask,
    baseline_rate,
):
    excluded_columns = _excluded_analysis_columns()
    categorical_columns = [
        col
        for col in df.columns
        if col not in excluded_columns
        and (
            df[col].dtype == "object"
            or df[col].nunique(dropna=False) <= 20
        )
    ]

    lift_tables = []

    for col in categorical_columns:
        grouped = (
            df.groupby(col, dropna=False)
            .agg(
                count=(RESPONSE_TIME_COLUMN, "size"),
                target_rate=(
                    RESPONSE_TIME_COLUMN,
                    lambda series: target_mask.loc[series.index].mean(),
                ),
                actual_mean=(RESPONSE_TIME_COLUMN, "mean"),
                residual_mean=("residual", "mean"),
            )
            .reset_index()
        )

        grouped = grouped[grouped["count"] >= MIN_LIFT_GROUP_COUNT].copy()

        if grouped.empty:
            continue

        grouped["lift"] = grouped["target_rate"] / baseline_rate
        grouped["column"] = col
        grouped = grouped.rename(columns={col: "value"})
        lift_tables.append(
            grouped[
                [
                    "column",
                    "value",
                    "count",
                    "target_rate",
                    "lift",
                    "actual_mean",
                    "residual_mean",
                ]
            ]
        )

    if not lift_tables:
        return pd.DataFrame(
            columns=[
                "column",
                "value",
                "count",
                "target_rate",
                "lift",
                "actual_mean",
                "residual_mean",
            ]
        )

    return (
        pd.concat(lift_tables, ignore_index=True)
        .sort_values(["lift", "count"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _calculate_numeric_differences(
    df,
    target_mask,
):
    excluded_columns = _excluded_analysis_columns()
    numeric_columns = [
        col
        for col in df.columns
        if col not in excluded_columns
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique(dropna=True) > 20
    ]

    rows = []

    for col in numeric_columns:
        target_mean = df.loc[target_mask, col].mean()
        other_mean = df.loc[~target_mask, col].mean()
        diff = target_mean - other_mean
        std = df[col].std()
        standardized_diff = diff / std if std else np.nan

        rows.append(
            {
                "column": col,
                "target_mean": target_mean,
                "other_mean": other_mean,
                "diff": diff,
                "standardized_diff": standardized_diff,
                "target_median": df.loc[target_mask, col].median(),
                "other_median": df.loc[~target_mask, col].median(),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "column",
                "target_mean",
                "other_mean",
                "diff",
                "standardized_diff",
                "target_median",
                "other_median",
            ]
        )

    numeric_differences = pd.DataFrame(rows)
    numeric_differences["abs_standardized_diff"] = (
        numeric_differences["standardized_diff"].abs()
    )

    return (
        numeric_differences
        .sort_values("abs_standardized_diff", ascending=False)
        .drop(columns="abs_standardized_diff")
        .reset_index(drop=True)
    )


def _plot_top_categorical_lift(categorical_lift):
    if categorical_lift.empty:
        return

    plot_data = categorical_lift.head(MAX_LIFT_PLOT_ROWS).copy()
    plot_data["label"] = (
        plot_data["column"].astype(str)
        + " = "
        + plot_data["value"].astype(str)
    )
    plot_data = plot_data.sort_values("lift", ascending=True)

    plt.figure(figsize=(11, 8))
    plt.barh(
        plot_data["label"],
        plot_data["lift"],
        color="steelblue",
    )
    plt.axvline(1, color="red", linestyle="--", linewidth=2, label="Average")
    plt.xlabel("Lift vs overall target rate")
    plt.title("Top Categorical Lift: High Response + High Residual")
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    lift_plot_path = PATH_OUTPUT_DIR / "high_response_residual_categorical_lift.png"
    plt.savefig(lift_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved categorical lift plot to: {lift_plot_path}")


def _plot_top_numeric_differences(numeric_differences):
    if numeric_differences.empty:
        return

    plot_data = numeric_differences.head(MAX_NUMERIC_PLOT_ROWS).copy()
    plot_data = plot_data.sort_values("standardized_diff", ascending=True)

    colors = np.where(
        plot_data["standardized_diff"] >= 0,
        "steelblue",
        "coral",
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_data["column"],
        plot_data["standardized_diff"],
        color=colors,
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Standardized mean difference")
    plt.title("Numeric Differences: High Response + High Residual")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    numeric_plot_path = PATH_OUTPUT_DIR / "high_response_residual_numeric_differences.png"
    plt.savefig(numeric_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved numeric difference plot to: {numeric_plot_path}")


def _excluded_analysis_columns():
    return {
        "index",
        RESPONSE_TIME_COLUMN,
        "predicted_response_time",
        "residual",
    }


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PATH_X_train = BASE_DIR / "output/data_splits/X_train.csv"
PATH_y_train = BASE_DIR / "output/data_splits/y_train.csv"

PATH_X_validation = BASE_DIR / "output/data_splits/X_val.csv"
PATH_y_validation = BASE_DIR / "output/data_splits/y_val.csv"
PATH_Y_validation_predictions = BASE_DIR / "output/predictions/y_pred_validation.csv"

PATH_OUTPUT_DIR = BASE_DIR / "output/Analysis/high_incidents"

RESPONSE_TIME_COLUMN = "response_time"
HIGH_RESPONSE_QUANTILE = 0.75
MAX_SCATTER_POINTS = 5000
MIN_LIFT_GROUP_COUNT = 100
MAX_LIFT_PLOT_ROWS = 20
MAX_NUMERIC_PLOT_ROWS = 15

# load dataset
df_high_incidents = pd.read_csv(PATH_X_validation)
y_validation = pd.read_csv(PATH_y_validation)
y_validation_predictions = pd.read_csv(PATH_Y_validation_predictions)
df_high_incidents[RESPONSE_TIME_COLUMN] = y_validation.iloc[:, 0]

if "prediction" in y_validation_predictions.columns:
    df_high_incidents["predicted_response_time"] = y_validation_predictions["prediction"]
else:
    df_high_incidents["predicted_response_time"] = y_validation_predictions.iloc[:, -1]

df_high_incidents["residual"] = (
    df_high_incidents[RESPONSE_TIME_COLUMN]
    - df_high_incidents["predicted_response_time"]
)

PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# filter for high response times
threshold = df_high_incidents[RESPONSE_TIME_COLUMN].quantile(HIGH_RESPONSE_QUANTILE)
df_high_response_times = df_high_incidents[
    df_high_incidents[RESPONSE_TIME_COLUMN] > threshold
].copy()


print(f"High response time threshold (q{HIGH_RESPONSE_QUANTILE:.2f}): {threshold:.2f}")
print(f"High response incidents: {len(df_high_response_times):,} / {len(df_high_incidents):,}")
print(df_high_response_times.describe())

# visualization()

print("\nResidual statistics for high response incidents:")
print(df_high_response_times["residual"].describe())

visual_residual()
analyze_lift_for_high_response_and_residuals(df_high_incidents)
