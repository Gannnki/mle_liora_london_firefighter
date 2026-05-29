from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent.parent

PATH_RESIDUAL_ANALYSIS = BASE_DIR / "output/predictions/residual_analysis.csv"
PATH_OUTPUT_DIR = BASE_DIR / "output/Analysis/incident_spatial"


def main():
    PATH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PATH_RESIDUAL_ANALYSIS)
    df = _prepare_spatial_df(df)

    plot_incident_density(df)
    plot_mean_residual_hexbin(df)
    plot_kde_density(df)


def _prepare_spatial_df(df):
    required_columns = {
        "Latitude",
        "Longitude",
        "residual",
    }
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{PATH_RESIDUAL_ANALYSIS} is missing columns: {sorted(missing_columns)}"
        )

    spatial_df = df.dropna(
        subset=[
            "Latitude",
            "Longitude",
            "residual",
        ]
    ).copy()

    # Keep only plausible London-area points to prevent map scale distortion.
    spatial_df = spatial_df[
        spatial_df["Latitude"].between(51.25, 51.75)
        & spatial_df["Longitude"].between(-0.55, 0.35)
    ].copy()

    print(f"Rows available for spatial plots: {len(spatial_df):,}")

    return spatial_df


def plot_incident_density(df):
    fig, ax = plt.subplots(figsize=(10, 8))

    hexbin = ax.hexbin(
        df["Longitude"],
        df["Latitude"],
        gridsize=85,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )

    ax.set_title("Incident Density Across London")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    colorbar = fig.colorbar(hexbin, ax=ax)
    colorbar.set_label("Log incident count")

    output_path = PATH_OUTPUT_DIR / "incident_density_hexbin.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved incident density plot to: {output_path}")


def plot_mean_residual_hexbin(df):
    fig, ax = plt.subplots(figsize=(10, 8))

    residual_limit = np.nanpercentile(np.abs(df["residual"]), 95)

    hexbin = ax.hexbin(
        df["Longitude"],
        df["Latitude"],
        C=df["residual"],
        reduce_C_function=np.mean,
        gridsize=75,
        mincnt=10,
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
    )

    ax.set_title("Mean Residual by Location")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    colorbar = fig.colorbar(hexbin, ax=ax)
    colorbar.set_label("Mean residual (actual - predicted, seconds)")

    output_path = PATH_OUTPUT_DIR / "mean_residual_hexbin.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved mean residual plot to: {output_path}")


def plot_kde_density(df):
    # KDE can be slow on very large data, so sample for a smooth overview.
    kde_df = df.sample(
        min(80_000, len(df)),
        random_state=42,
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.kdeplot(
        data=kde_df,
        x="Longitude",
        y="Latitude",
        fill=True,
        levels=40,
        thresh=0.02,
        cmap="mako",
        ax=ax,
    )

    ax.set_title("Smoothed Incident Density Across London")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    output_path = PATH_OUTPUT_DIR / "incident_density_kde.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved KDE density plot to: {output_path}")


if __name__ == "__main__":
    main()
