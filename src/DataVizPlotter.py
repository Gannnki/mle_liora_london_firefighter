import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


class EDAVisualizer:
    def __init__(self, path):
        self.output_dir = Path(path) if path else Path("output/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # seaborn style for better aesthetics
        sns.set_theme(style="whitegrid")

    # ==============================
    # Top Categories
    # ==============================
    def plot_top_categories(self, df: pd.DataFrame, col: str, filename: str, top_n=10):
        counts = df[col].value_counts().head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=counts.values, y=counts.index)

        plt.title(f"Top {top_n} {col}")
        plt.xlabel("Count")
        plt.ylabel(col)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    # ==============================
    # Distribution
    # ==============================
    def plot_distribution(self, df: pd.DataFrame, col: str, filename: str, bins=50):
        data = df[col].dropna()

        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=bins, kde=True)

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_boxplot(self, df: pd.DataFrame, col: str, filename: str):
        data = df[col].dropna()

        plt.figure(figsize=(10, 6))

        sns.violinplot(x=data, inner=None, color="lightgray")
        sns.boxplot(x=data, width=0.2)

        plt.title(f"Distribution (Box + Violin) of {col}")
        plt.xlabel(col)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
    # ==============================
    # Grouped Mean
    # ==============================
    def plot_grouped_mean(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        filename: str,
        top_n=10
    ):
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

    # ==============================
    # Correlation Heatmap
    # ==============================
    def plot_correlation(self, df_cols, filename: str):
        corr = df_cols.corr()

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            cbar=True,
            linewidths=0.5 
        )

        plt.title("Correlation Analysis")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    # ==============================
    # Specific Plots
    # ==============================
    def plot_top_boroughs(self, df: pd.DataFrame, filename: str, top_n=10):
        counts = df["IncGeo_BoroughName"].value_counts().head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=counts.values, y=counts.index)

        plt.title(f"Top {top_n} Boroughs")
        plt.xlabel("Count")
        plt.ylabel("Borough Name")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_top_incident_types(self, df: pd.DataFrame, filename: str, top_n=10):
        counts = df["IncidentGroup"].value_counts().head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=counts.values, y=counts.index)

        plt.title(f"Top {top_n} Incident Types")
        plt.xlabel("Count")
        plt.ylabel("Incident Type")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()