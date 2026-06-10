import joblib
import __main__
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin


BASE_DIR = Path(__file__).resolve().parent.parent.parent

PATH_TO_MODEL = BASE_DIR / "artifacts/best_models/best_model.pkl"

PATH_X_train = BASE_DIR / "output/scalers/X_train_scaled.csv"
PATH_X_test = BASE_DIR / "output/scalers/X_test_scaled.csv"

OUTPUT_PATH_SHAP = BASE_DIR / "output/Analysis/SHAP"


class EncodingTransformer(BaseEstimator, TransformerMixin):
    """Compatibility class for models pickled from a notebook __main__ scope."""

    def __init__(self, config=None):
        self.config = config
        self.fitted_encoders_ = {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        encoded_parts = []

        for col, cfg in self._feature_config().items():
            if col not in X.columns:
                continue

            encoding = cfg["encoding"]

            if encoding == "IGNORE":
                continue
            if encoding in ["NUMERIC_KEEP", "BINARY_KEEP"]:
                encoded_parts.append(X[[col]].copy())
            elif encoding == "CYCLIC":
                encoded_parts.append(self._cyclic_encode(X, col))
            else:
                encoded_parts.append(
                    self._transform_with_fitted_encoder(
                        X,
                        col,
                        encoding,
                    )
                )

        return pd.concat(encoded_parts, axis=1)

    def _feature_config(self):
        if self.config is None:
            return {}

        return self.config.get(
            "feature_encoding",
            self.config,
        )

    def _cyclic_encode(self, X, col):
        period_map = {
            "Month": 12,
            "Weekday": 7,
            "Hour": 24,
        }

        period = period_map[col]

        return pd.DataFrame(
            {
                f"{col}_sin": np.sin(2 * np.pi * X[col] / period),
                f"{col}_cos": np.cos(2 * np.pi * X[col] / period),
            },
            index=X.index,
        )

    def _transform_with_fitted_encoder(self, X, col, encoding):
        encoder_info = self.fitted_encoders_[col]
        encoder = encoder_info
        top_categories = None

        if isinstance(encoder_info, dict):
            encoder = encoder_info.get("encoder", encoder_info)
            top_categories = encoder_info.get("top_categories")
        elif isinstance(encoder_info, (list, tuple)):
            encoder = encoder_info[0]
            if len(encoder_info) > 1:
                top_categories = encoder_info[1]

        X_col = X[[col]].copy()

        if encoding == "TOP_N_PLUS_ONE_HOT" and top_categories is not None:
            X_col[col] = X_col[col].where(
                X_col[col].isin(top_categories),
                "Other",
            )

        encoded = encoder.transform(X_col)

        return encoded


# Todo: move it to trainer or utility modules
def run_shap_analysis(
    path_to_model,
    path_X_train,
    path_X_test,
    output_dir,
    max_background_samples=500,
    max_explain_samples=500,
    random_state=42,
):
    """Run SHAP analysis from saved model and dataset paths."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    __main__.EncodingTransformer = EncodingTransformer

    # load model
    model = joblib.load(
        path_to_model
    )
    model = getattr(
        model,
        "best_estimator_",
        model,
    )

    # load datasets
    X_train = pd.read_csv(
        path_X_train,
    )

    X_test = pd.read_csv(
        path_X_test,
    )

    # sample data for SHAP
    X_background = X_train.sample(
        min(max_background_samples, len(X_train)),
        random_state=random_state,
    )

    X_explain = X_test.sample(
        min(max_explain_samples, len(X_test)),
        random_state=random_state,
    )

    if hasattr(model, "steps"):
        preprocessing = model[:-1]
        final_model = model.steps[-1][1]

        X_background = preprocessing.transform(X_background)
        X_explain = preprocessing.transform(X_explain)

        feature_names = getattr(
            model.named_steps.get("scaler"),
            "feature_names_in_",
            None,
        )

        if feature_names is not None:
            X_background = pd.DataFrame(
                X_background,
                columns=feature_names,
            )
            X_explain = pd.DataFrame(
                X_explain,
                columns=feature_names,
            )
    else:
        final_model = model

    X_background = _ensure_numeric_float_frame(X_background)
    X_explain = _ensure_numeric_float_frame(X_explain)

    # create explainer
    explainer = shap.Explainer(
        final_model,
        X_background,
    )

    shap_values = explainer(
        X_explain,
    )

    shap_values = _convert_log1p_shap_to_original_scale(shap_values)

    # global feature importance
    shap.plots.bar(
        shap_values,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "shap_bar.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # beeswarm summary plot
    shap.plots.beeswarm(
        shap_values,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    _save_percent_importance_plot(
        shap_values,
        output_dir,
    )

    return shap_values


def _ensure_numeric_float_frame(X):
    X = pd.DataFrame(X).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    return X.astype(float)


def _convert_log1p_shap_to_original_scale(shap_values):
    """Approximate log1p-target SHAP values on the original target scale.

    The trained model predicts log1p(AttendanceTimeSeconds). SHAP values are
    additive in that log space, so the raw SHAP plot is hard to read as seconds.
    This rescales each row so:

        expm1(base_log) + sum(shap_seconds) = expm1(prediction_log)

    That keeps the explanation additive on the original response-time scale.
    """

    values_log = np.asarray(
        shap_values.values,
        dtype=float,
    )

    base_values_log = np.asarray(
        shap_values.base_values,
        dtype=float,
    )

    if base_values_log.ndim == 0:
        base_values_log = np.full(
            values_log.shape[0],
            float(base_values_log),
        )

    log_delta = values_log.sum(axis=1)
    base_values_original = np.expm1(base_values_log)
    prediction_original = np.expm1(base_values_log + log_delta)
    original_delta = prediction_original - base_values_original

    scale = np.divide(
        original_delta,
        log_delta,
        out=np.exp(base_values_log),
        where=np.abs(log_delta) > 1e-12,
    )

    values_original = values_log * scale[:, np.newaxis]

    return shap.Explanation(
        values=values_original,
        base_values=base_values_original,
        data=shap_values.data,
        display_data=getattr(shap_values, "display_data", None),
        feature_names=shap_values.feature_names,
        instance_names=getattr(shap_values, "instance_names", None),
        output_names=getattr(shap_values, "output_names", None),
    )


def _save_percent_importance_plot(
    shap_values,
    output_dir,
    max_features=20,
):
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    total_importance = mean_abs_shap.sum()

    if total_importance == 0:
        percentages = np.zeros_like(mean_abs_shap)
    else:
        percentages = mean_abs_shap / total_importance * 100

    feature_importance = pd.DataFrame(
        {
            "feature": shap_values.feature_names,
            "mean_abs_shap_seconds": mean_abs_shap,
            "importance_percent": percentages,
        }
    ).sort_values(
        "importance_percent",
        ascending=False,
    )

    csv_path = output_dir / "shap_feature_importance_percent.csv"
    feature_importance.to_csv(
        csv_path,
        index=False,
    )

    plot_data = feature_importance.head(max_features).sort_values(
        "importance_percent",
        ascending=True,
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_data["feature"],
        plot_data["importance_percent"],
        color="steelblue",
    )
    plt.xlabel("Share of mean absolute SHAP value (%)")
    plt.title("SHAP Feature Importance (% of Total)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    percent_plot_path = output_dir / "shap_feature_importance_percent.png"
    plt.savefig(
        percent_plot_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved percent SHAP importance to: {percent_plot_path}")
    print(f"Saved percent SHAP importance CSV to: {csv_path}")


if __name__ == "__main__":

    shap_values = run_shap_analysis(
        path_to_model=PATH_TO_MODEL,
        path_X_train=PATH_X_train,
        path_X_test=PATH_X_test,
        output_dir=OUTPUT_PATH_SHAP,
    )
