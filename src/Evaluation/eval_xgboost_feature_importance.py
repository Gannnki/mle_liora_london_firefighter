"""Post-training XGBoost feature-importance artifact generation."""

from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent

PATH_MODEL = BASE_DIR / "artifacts/best_models/best_model.pkl"
PATH_X_TRAIN = BASE_DIR / "output/scalers/X_train_scaled.csv"
OUTPUT_DIR = BASE_DIR / "output/Analysis/XGBoost"

IMPORTANCE_TYPES = [
    "weight",
    "gain",
    "cover",
    "total_gain",
    "total_cover",
]


def make_safe_feature_names(columns):
    safe_columns = []
    seen = {}

    for column in columns:
        safe_name = re.sub(
            r"[^0-9A-Za-z_]+",
            "_",
            str(column),
        ).strip("_")

        if not safe_name:
            safe_name = "feature"

        count = seen.get(safe_name, 0)
        seen[safe_name] = count + 1

        if count:
            safe_name = f"{safe_name}_{count}"

        safe_columns.append(safe_name)

    return safe_columns


def load_final_model(path_model):
    model = joblib.load(path_model)
    model = getattr(model, "best_estimator_", model)

    if hasattr(model, "steps"):
        return model.steps[-1][1]

    return model


def feature_name_mapping(path_X_train):
    columns = pd.read_csv(
        path_X_train,
        nrows=0,
    ).columns

    safe_columns = make_safe_feature_names(columns)

    mapping = {
        f"f{i}": column
        for i, column in enumerate(safe_columns)
    }

    mapping.update(
        {
            column: column
            for column in safe_columns
        }
    )

    return mapping, safe_columns


def get_importance_frame(model, feature_mapping, all_features):
    if not hasattr(model, "get_booster"):
        raise TypeError(
            "The saved model does not expose get_booster(). "
            "This script expects an XGBoost model."
        )

    booster = model.get_booster()
    importance_df = pd.DataFrame(
        {
            "feature": all_features,
        }
    )

    for importance_type in IMPORTANCE_TYPES:
        scores = booster.get_score(
            importance_type=importance_type,
        )
        readable_scores = {
            feature_mapping.get(feature, feature): value
            for feature, value in scores.items()
        }

        importance_df[importance_type] = (
            importance_df["feature"]
            .map(readable_scores)
            .fillna(0)
        )

    for importance_type in IMPORTANCE_TYPES:
        total = importance_df[importance_type].sum()
        percent_col = f"{importance_type}_percent"

        if total == 0:
            importance_df[percent_col] = 0
        else:
            importance_df[percent_col] = (
                importance_df[importance_type] / total * 100
            )

    return importance_df.sort_values(
        "gain",
        ascending=False,
    )


def save_gain_plot(importance_df, output_dir, max_features=20):
    plot_data = (
        importance_df.head(max_features)
        .sort_values("gain_percent", ascending=True)
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_data["feature"],
        plot_data["gain_percent"],
        color="steelblue",
    )
    plt.xlabel("Share of XGBoost gain importance (%)")
    plt.title("XGBoost Built-in Feature Importance (Gain)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "xgboost_feature_importance_gain.png"
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return output_path


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_final_model(PATH_MODEL)
    feature_mapping, all_features = feature_name_mapping(PATH_X_TRAIN)
    importance_df = get_importance_frame(
        model,
        feature_mapping,
        all_features,
    )

    csv_path = OUTPUT_DIR / "xgboost_feature_importance.csv"
    importance_df.to_csv(
        csv_path,
        index=False,
    )

    plot_path = save_gain_plot(
        importance_df,
        OUTPUT_DIR,
    )

    print("Saved XGBoost feature importance:")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")
    print("\nTop 20 features by gain:")
    print(
        importance_df[
            [
                "feature",
                "gain",
                "gain_percent",
                "weight",
                "cover",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
