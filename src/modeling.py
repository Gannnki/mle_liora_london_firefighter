import yaml
import joblib
import numpy as np
import pandas as pd
import re

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_squared_log_error
)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor


MODEL_MAPPING = {
    "RandomForestRegressor": RandomForestRegressor,
    "LinearRegression": LinearRegression,
    "KNeighborsRegressor": KNeighborsRegressor,
}

OPTIONAL_MODEL_IMPORTS = {
    "XGBRegressor": ("xgboost", "XGBRegressor"),
    "LGBMRegressor": ("lightgbm", "LGBMRegressor"),
    "CatBoostRegressor": ("catboost", "CatBoostRegressor"),
}


def get_model_class(model_type):
    if model_type in MODEL_MAPPING:
        return MODEL_MAPPING[model_type]

    if model_type in OPTIONAL_MODEL_IMPORTS:
        module_name, class_name = OPTIONAL_MODEL_IMPORTS[model_type]

        try:
            module = __import__(
                module_name,
                fromlist=[class_name],
            )
        except ImportError as exc:
            raise ImportError(
                f"{model_type} requires the '{module_name}' package. "
                "Install the project requirements before running this model."
            ) from exc

        model_class = getattr(
            module,
            class_name,
        )
        MODEL_MAPPING[model_type] = model_class
        return model_class

    raise KeyError(
        f"Unknown model type '{model_type}'. Add it to MODEL_MAPPING first."
    )


class ModelTrainer:
    def __init__(
        self,
        path_X_train,
        path_X_test,

        path_y_train_log,
        path_y_test_log,

        path_y_train,
        path_y_test,

        path_model_config=None
    ):

        self.path_X_train = path_X_train
        self.path_X_test = path_X_test

        self.path_y_train_log = path_y_train_log
        self.path_y_test_log = path_y_test_log

        self.path_y_train = path_y_train
        self.path_y_test = path_y_test
        # Initialisation of the path to the model configuration file
        self.path_model_config =path_model_config
        self.fitted_models = {}

        self.load_data()

    def load_data(self):
        self.X_train = pd.read_csv(
            self.path_X_train,
            index_col=0,
        )

        self.X_test = pd.read_csv(
            self.path_X_test,
            index_col=0,
        )

        self._sanitize_feature_names()

        # original targets
        self.y_train = pd.read_csv(
            self.path_y_train,
        ).squeeze()

        self.y_test = pd.read_csv(
            self.path_y_test,
        ).squeeze()

        # log targets
        self.y_train_log = pd.read_csv(
            self.path_y_train_log,
        ).squeeze()

        self.y_test_log = pd.read_csv(
            self.path_y_test_log,
        ).squeeze()

    def _sanitize_feature_names(self):
        if list(self.X_train.columns) != list(self.X_test.columns):
            raise ValueError(
                "Train and test feature columns do not match before training."
            )

        sanitized_columns = self._make_safe_feature_names(
            self.X_train.columns
        )

        self.X_train.columns = sanitized_columns
        self.X_test.columns = sanitized_columns

    @staticmethod
    def _make_safe_feature_names(columns):
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

    def load_models(self, path_model_config):
        self.path_model_config = path_model_config

        with open(
            self.path_model_config,
            "r",
        ) as file:

            config = yaml.safe_load(file)

        self.models = []

        for model_config in config["models"]:

            model_class = get_model_class(
                model_config["type"]
            )

            model = model_class(
                **model_config["params"]
            )

            self.models.append(
                (
                    model_config["name"],
                    model,
                    model_config.get("grid_search", {}),
                )
            )

    def evaluate_model(
        self,
        model_name,
        model,
        grid_search_config=None,
    ):
        best_params = None

        if grid_search_config and grid_search_config.get("enabled", False):
            fixed_model = clone(model)
            fixed_model.fit(
                self.X_train,
                self.y_train_log,
            )

            fixed_metrics = self._evaluate_predictions(
                fixed_model,
                self.X_test,
                self.y_test,
            )

            print(f"\nFixed-parameter result for {model_name}:")
            print(
                f"Test MAE={fixed_metrics['MAE']}, "
                f"Test RMSE={fixed_metrics['RMSE']}, "
                f"Test R2 Score={fixed_metrics['R2 Score']}"
            )
            search_method = grid_search_config.get(
                "method",
                "grid",
            )

            if search_method == "randomized":
                print(f"Starting RandomizedSearchCV for {model_name}...")
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=grid_search_config[
                        "param_distributions"
                    ],
                    n_iter=grid_search_config.get("n_iter", 5),
                    cv=grid_search_config.get("cv", 3),
                    scoring=grid_search_config.get(
                        "scoring",
                        "neg_mean_absolute_error",
                    ),
                    n_jobs=grid_search_config.get("n_jobs", -1),
                    verbose=grid_search_config.get("verbose", 1),
                    random_state=grid_search_config.get("random_state", 42),
                )
            else:
                print(f"Starting GridSearchCV for {model_name}...")
                search = GridSearchCV(
                    estimator=model,
                    param_grid=grid_search_config["param_grid"],
                    cv=grid_search_config.get("cv", 3),
                    scoring=grid_search_config.get(
                        "scoring",
                        "neg_mean_absolute_error",
                    ),
                    n_jobs=grid_search_config.get("n_jobs", -1),
                    verbose=grid_search_config.get("verbose", 1),
                )

            search.fit(
                self.X_train,
                self.y_train_log,
            )

            model = search.best_estimator_
            best_params = search.best_params_
        else:
            # train on log-transformed target
            model.fit(
                self.X_train,
                self.y_train_log,
            )

        self.fitted_models[model_name] = model

        train_metrics = self._evaluate_predictions(
            model,
            self.X_train,
            self.y_train,
        )

        test_metrics = self._evaluate_predictions(
            model,
            self.X_test,
            self.y_test,
        )

        return {
            "Model": model_name,
            "Train MAE": train_metrics["MAE"],
            "Train RMSE": train_metrics["RMSE"],
            "Train R2 Score": train_metrics["R2 Score"],
            "Test MAE": test_metrics["MAE"],
            "Test RMSE": test_metrics["RMSE"],
            "Test R2 Score": test_metrics["R2 Score"],
            "Test RMSLE": test_metrics["RMSLE"],
            "Test P90": test_metrics['P90 absolute error'],
            "Best Params": best_params,
        }

    def _evaluate_predictions(
        self,
        model,
        X,
        y,
    ):
        y_pred = self._predict_original_scale(
            model,
            X,
        )

        mae = mean_absolute_error(
            y,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y,
                y_pred,
            )
        )

        r2 = r2_score(
            y,
            y_pred,
        )

        rmsle = np.sqrt(mean_squared_log_error(y, np.maximum(y_pred, 0)))

        p90_error = np.percentile(np.abs(y-y_pred), 90)

        return {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 4),
            "RMSLE": round(rmsle, 2),
            "P90 absolute error": round(p90_error, 4)
        }

    @staticmethod
    def _predict_original_scale(
        model,
        X,
    ):
        # Model is trained on log1p(target), so predictions are converted back.
        y_pred_log = model.predict(
            X
        )

        return np.expm1(
            y_pred_log
        )

    def save_predictions(
        self,
        model_name,
        output_path,
        dataset="test",
    ):
        if model_name not in self.fitted_models:
            raise ValueError(
                f"Model '{model_name}' has not been fitted."
            )

        if dataset == "train":
            X = self.X_train
            y = self.y_train
        elif dataset == "test":
            X = self.X_test
            y = self.y_test
        else:
            raise ValueError(
                "dataset must be either 'train' or 'test'."
            )

        y_pred = self._predict_original_scale(
            self.fitted_models[model_name],
            X,
        )

        df_predictions = pd.DataFrame(
            {
                "actual": y.to_numpy(),
                "prediction": y_pred,
            }
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df_predictions.to_csv(
            output_path,
            index=False,
        )

        return df_predictions

    def naive_baseline(self):

        y_train_pred_naive = np.full(
            len(self.y_train),
            self.y_train.mean(),
        )

        y_test_pred_naive = np.full(
            len(self.y_test),
            self.y_train.mean(),
        )

        return {
            "Model": "Naive Baseline (Mean)",
            "Train MAE": round(
                mean_absolute_error(
                    self.y_train,
                    y_train_pred_naive,
                ),
                2,
            ),
            "Train RMSE": round(
                np.sqrt(
                    mean_squared_error(
                        self.y_train,
                        y_train_pred_naive,
                    )
                ),
                2,
            ),
            "Train R2 Score": round(
                r2_score(
                    self.y_train,
                    y_train_pred_naive,
                ),
                4,
            ),
            "Test MAE": round(
                mean_absolute_error(
                    self.y_test,
                    y_test_pred_naive,
                ),
                2,
            ),
            "Test RMSE": round(
                np.sqrt(
                    mean_squared_error(
                        self.y_test,
                        y_test_pred_naive,
                    )
                ),
                2,
            ),
            "Test R2 Score": round(
                r2_score(
                    self.y_test,
                    y_test_pred_naive,
                ),
                4,
            ),
            "Best Params": None,
        }

    def run_models(self):
        # todo: make sure the models are loaded
        results = [
            self.evaluate_model(
                model_name,
                model,
                grid_search_config,
            )
            for model_name, model, grid_search_config in self.models
        ]

        results.append(
            self.naive_baseline()
        )

        df_results = (
            pd.DataFrame(results)
            .sort_values(
                "Test R2 Score",
                ascending=False,
            )
        )

        return df_results

    def save_best_model(
        self,
        results,
        output_path,
        metric="Test R2 Score",
    ):
        model_results = results[
            results["Model"].isin(self.fitted_models)
        ]

        if model_results.empty:
            raise ValueError("No fitted models available to save.")

        best_model_name = model_results.sort_values(
            metric,
            ascending=False,
        ).iloc[0]["Model"]

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.fitted_models[best_model_name],
            output_path,
        )

        return best_model_name
