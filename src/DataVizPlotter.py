"""Plotting helpers for exploratory analysis outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class EDAVisualizer:
    """Create and save common EDA charts to an output directory."""

    def __init__(self, path: str | Path | None):
        """Initialize the visualizer and configure the default plot style."""
        self.output_dir = Path(path) if path else Path("output/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="whitegrid")

    def plot_top_categories(
        self,
        df: pd.DataFrame,
        col: str,
        filename: str,
        top_n: int = 10,
    ) -> None:
        """Save a horizontal bar chart of the most common values in a column."""
        counts = df[col].value_counts().head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=counts.values, y=counts.index)

        plt.title(f"Top {top_n} {col}")
        plt.xlabel("Count")
        plt.ylabel(col)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_distribution(
        self,
        df: pd.DataFrame,
        col: str,
        filename: str,
        bins: int = 50,
    ) -> None:
        """Save a histogram and density curve for a numeric column."""
        data = df[col].dropna()

        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=bins, kde=True)

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_boxplot(self, df: pd.DataFrame, col: str, filename: str) -> None:
        """Save a combined violin and box plot for a numeric column."""
        data = df[col].dropna()

        plt.figure(figsize=(10, 6))

        sns.violinplot(x=data, inner=None, color="lightgray")
        sns.boxplot(x=data, width=0.2)

        plt.title(f"Distribution (Box + Violin) of {col}")
        plt.xlabel(col)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_grouped_mean(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        filename: str,
        top_n: int = 10,
    ) -> None:
        """Save a bar chart of the top grouped mean values."""
        grouped = (
            df.groupby(group_col)[value_col]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(data=grouped, x=value_col, y=group_col)

        plt.title(f"Average {value_col} by {group_col}")
        plt.xlabel(f"Mean {value_col}")
        plt.ylabel(group_col)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_correlation(self, df_cols: pd.DataFrame, filename: str) -> None:
        """Save a correlation heatmap for selected numeric columns."""
        corr = df_cols.corr()

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            cbar=True,
            linewidths=0.5,
        )

        plt.title("Correlation Analysis")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_top_boroughs(
        self,
        df: pd.DataFrame,
        filename: str,
        top_n: int = 10,
    ) -> None:
        """Save a bar chart for the boroughs with the most incident rows."""
        counts = df["IncGeo_BoroughName"].value_counts().head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=counts.values, y=counts.index)

        plt.title(f"Top {top_n} Boroughs")
        plt.xlabel("Count")
        plt.ylabel("Borough Name")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_top_incident_types(
        self,
        df: pd.DataFrame,
        filename: str,
        top_n: int = 10,
    ) -> None:
        """Save a bar chart for the most common incident groups."""
        counts = df["IncidentGroup"].value_counts().head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=counts.values, y=counts.index)

        plt.title(f"Top {top_n} Incident Types")
        plt.xlabel("Count")
        plt.ylabel("Incident Type")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
