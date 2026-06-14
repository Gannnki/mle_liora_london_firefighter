"""Model training utilities for the tabular regression pipeline."""

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

EVAL_SET_MODEL_MODULES = {
    "catboost",
    "lightgbm",
    "xgboost",
}

DEFAULT_EARLY_STOPPING_ROUNDS = 50


def get_model_class(model_type):
    """Return the estimator class for a configured model type."""
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
    """Train candidate regressors on log-transformed response time targets.

    The trainer reads preprocessed CSV splits, fits configured models on
    ``log1p(AttendanceTimeSeconds)``, evaluates predictions after converting
    them back with ``expm1``, and saves the best validation model artifact.
    """

    def __init__(
        self,
        path_X_train,
        path_X_validation,
        path_y_train_log,
        path_y_validation_log,
        path_y_train,
        path_y_validation,

        path_model_config=None,
    ):

        self.path_X_train = path_X_train
        self.path_X_validation = path_X_validation

        self.path_y_train_log = path_y_train_log
        self.path_y_validation_log = path_y_validation_log

        self.path_y_train = path_y_train
        self.path_y_validation = path_y_validation
        # Initialisation of the path to the model configuration file
        self.path_model_config =path_model_config
        self.fitted_models = {}

        self.load_data()
        
        # self._create_sample_weights()
        

    def load_data(self):
        self.X_train = pd.read_csv(
            self.path_X_train,
        )

        self.X_validation = pd.read_csv(
            self.path_X_validation,
        )

        self._sanitize_feature_names()

        # original targets
        self.y_train = pd.read_csv(
            self.path_y_train,
        ).squeeze()

        self.y_validation = pd.read_csv(
            self.path_y_validation,
        ).squeeze()

        # log targets
        self.y_train_log = pd.read_csv(
            self.path_y_train_log,
        ).squeeze()

        self.y_validation_log = pd.read_csv(
            self.path_y_validation_log,
        ).squeeze()

    # def _create_sample_weights(self):
    #     self.sample_weight = np.ones(
    #         len(self.y_train),
    #         dtype=float,
    #     )
    #     self.sample_weight[self.y_train < 100] = 1.8
    #     self.sample_weight[self.y_train > 400] = 2.5
    #
    #     low_target_count = int((self.y_train < 100).sum())
    #     high_target_count = int((self.y_train > 400).sum())
    #
    #     print(
    #         "Sample weights: "
    #         f"target < 100 gets weight 1.80 "
    #         f"[{low_target_count:,}/{len(self.y_train):,} rows], "
    #         f"target > 400 gets weight 2.5 "
    #         f"[{high_target_count:,}/{len(self.y_train):,} rows]"
    #     )

    def _sanitize_feature_names(self):
        if list(self.X_train.columns) != list(self.X_validation.columns):
            raise ValueError(
                "Train and validation feature columns do not match before training."
            )

        sanitized_columns = self._make_safe_feature_names(
            self.X_train.columns
        )

        self.X_train.columns = sanitized_columns
        self.X_validation.columns = sanitized_columns

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
        X_train_fit = self.X_train.to_numpy(dtype=np.float32, copy=False)
        X_validation_fit = self.X_validation.to_numpy(dtype=np.float32, copy=False)
        y_train_log_fit = self.y_train_log.to_numpy(dtype=np.float32, copy=False)
        y_validation_log_fit = self.y_validation_log.to_numpy(dtype=np.float32, copy=False)

        if grid_search_config and grid_search_config.get("enabled", False):
            fixed_model = clone(model)
            self._fit_model(
                fixed_model,
                X_train=X_train_fit,
                y_train_log=y_train_log_fit,
                eval_X_train=X_train_fit,
                eval_y_train_log=y_train_log_fit,
                eval_X_validation=X_validation_fit,
                eval_y_validation_log=y_validation_log_fit,
            )

            fixed_metrics = self._evaluate_predictions(
                fixed_model,
                X_validation_fit,
                self.y_validation,
            )

            print(f"\nFixed-parameter result for {model_name}:")
            print(
                f"Validation MAE={fixed_metrics['MAE']}, "
                f"Validation RMSE={fixed_metrics['RMSE']}, "
                f"Validation R2 Score={fixed_metrics['R2 Score']}"
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

            # sample_weight_fit_kwargs = self._sample_weight_fit_kwargs(model)
            X_search, y_search = self._search_training_data(
                X_train_fit,
                y_train_log_fit,
                grid_search_config,
            )

            search.fit(
                X_search,
                y_search,
                # **sample_weight_fit_kwargs,
            )

            model = search.best_estimator_
            best_params = search.best_params_
            self._fit_model(
                model,
                X_train=X_train_fit,
                y_train_log=y_train_log_fit,
                eval_X_train=X_train_fit,
                eval_y_train_log=y_train_log_fit,
                eval_X_validation=X_validation_fit,
                eval_y_validation_log=y_validation_log_fit,
            )
        else:
            self._fit_model(
                model,
                X_train=X_train_fit,
                y_train_log=y_train_log_fit,
                eval_X_train=X_train_fit,
                eval_y_train_log=y_train_log_fit,
                eval_X_validation=X_validation_fit,
                eval_y_validation_log=y_validation_log_fit,
            )

        self.fitted_models[model_name] = model

        train_metrics = self._evaluate_predictions(
            model,
            X_train_fit,
            self.y_train,
        )

        validation_metrics = self._evaluate_predictions(
            model,
            X_validation_fit,
            self.y_validation,
        )

        return {
            "Model": model_name,
            "Train MAE": train_metrics["MAE"],
            "Train RMSE": train_metrics["RMSE"],
            "Train R2 Score": train_metrics["R2 Score"],
            "Validation MAE": validation_metrics["MAE"],
            "Validation RMSE": validation_metrics["RMSE"],
            "Validation R2 Score": validation_metrics["R2 Score"],
            "Validation RMSLE": validation_metrics["RMSLE"],
            "Validation P90": validation_metrics['P90 absolute error'],
            "Best Params": best_params,
        }

    @staticmethod
    def _search_training_data(
        X_train,
        y_train,
        grid_search_config,
    ):
        sample_rows = grid_search_config.get("search_sample_rows")
        if sample_rows is None or sample_rows >= X_train.shape[0]:
            return X_train, y_train

        random_state = grid_search_config.get("random_state", 42)
        rng = np.random.default_rng(random_state)
        indices = np.sort(
            rng.choice(
                X_train.shape[0],
                size=sample_rows,
                replace=False,
            )
        )

        print(
            "Randomized search using sampled training rows: "
            f"{sample_rows:,}/{X_train.shape[0]:,}"
        )
        return X_train[indices], y_train[indices]

    def _fit_model(
        self,
        model,
        X_train=None,
        y_train_log=None,
        eval_X_train=None,
        eval_y_train_log=None,
        eval_X_validation=None,
        eval_y_validation_log=None,
    ):
        fit_kwargs = {}
        X_train = self.X_train if X_train is None else X_train
        y_train_log = self.y_train_log if y_train_log is None else y_train_log
        eval_X_train = self.X_train if eval_X_train is None else eval_X_train
        eval_y_train_log = (
            self.y_train_log if eval_y_train_log is None else eval_y_train_log
        )
        eval_X_validation = (
            self.X_validation if eval_X_validation is None else eval_X_validation
        )
        eval_y_validation_log = (
            self.y_validation_log
            if eval_y_validation_log is None
            else eval_y_validation_log
        )

        if self._supports_eval_set(model):
            self._configure_early_stopping(model)
            fit_kwargs["eval_set"] = [
                (eval_X_train, eval_y_train_log),
                (eval_X_validation, eval_y_validation_log),
            ]

        # fit_kwargs.update(
        #     self._sample_weight_fit_kwargs(model)
        # )

        model.fit(
            X_train,
            y_train_log,
            **fit_kwargs,
        )

    # def _sample_weight_fit_kwargs(
    #     self,
    #     model,
    # ):
    #     if not self._supports_fit_parameter(
    #         model,
    #         "sample_weight",
    #     ):
    #         return {}
    #
    #     return {
    #         "sample_weight": self.sample_weight,
    #     }
    #
    # @staticmethod
    # def _supports_fit_parameter(
    #     model,
    #     parameter_name,
    # ):
    #     try:
    #         fit_signature = inspect.signature(model.fit)
    #     except (TypeError, ValueError):
    #         return False
    #
    #     return parameter_name in fit_signature.parameters

    @staticmethod
    def _supports_eval_set(model):
        model_module = model.__class__.__module__.split(".")[0]
        return model_module in EVAL_SET_MODEL_MODULES

    @staticmethod
    def _configure_early_stopping(model):
        model_module = model.__class__.__module__.split(".")[0]
        params = model.get_params()

        if model_module in {"catboost", "xgboost"}:
            if params.get("early_stopping_rounds") is None:
                model.set_params(
                    early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
                )

        if model_module == "lightgbm":
            if params.get("early_stopping_round") is None:
                model.set_params(
                    early_stopping_round=DEFAULT_EARLY_STOPPING_ROUNDS,
                )

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
        dataset="validation",
    ):
        if model_name not in self.fitted_models:
            raise ValueError(
                f"Model '{model_name}' has not been fitted."
            )

        if dataset == "train":
            X = self.X_train
            y = self.y_train
        elif dataset in {"validation", "val"}:
            X = self.X_validation
            y = self.y_validation
        else:
            raise ValueError(
                "dataset must be either 'train' or 'validation'."
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

        y_validation_pred_naive = np.full(
            len(self.y_validation),
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
            "Validation MAE": round(
                mean_absolute_error(
                    self.y_validation,
                    y_validation_pred_naive,
                ),
                2,
            ),
            "Validation RMSE": round(
                np.sqrt(
                    mean_squared_error(
                        self.y_validation,
                        y_validation_pred_naive,
                    )
                ),
                2,
            ),
            "Validation R2 Score": round(
                r2_score(
                    self.y_validation,
                    y_validation_pred_naive,
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
                "Validation R2 Score",
                ascending=False,
            )
        )

        return df_results

    def save_best_model(
        self,
        results,
        output_path,
        metric="Validation R2 Score",
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
