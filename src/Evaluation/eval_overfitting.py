import numpy as np
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import joblib
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
path_X_train = BASE_DIR / "output/scalers/X_train_scaled.csv"
path_y_train = BASE_DIR / "output/data_splits/y_train_log.csv"
path_X_test = BASE_DIR / "output/scalers/X_test_scaled.csv"
path_y_test = BASE_DIR / "output/data_splits/y_test_log.csv"

path_to_model = BASE_DIR / "artifacts/best_models/best_model.pkl"


# load model
best_model = joblib.load(
        path_to_model
    )
best_model = getattr(
        best_model,
        "best_estimator_",
        best_model,
    )

    # load datasets
X_train = pd.read_csv(
    path_X_train,
)
y_train = pd.read_csv(
    path_y_train,
).squeeze("columns")
X_test = pd.read_csv(
    path_X_test,
)
y_test = pd.read_csv(
    path_y_test,
).squeeze("columns")

model = clone(best_model)
if hasattr(model, "set_params"):
    model_params = model.get_params()
    params_to_disable = {}

    if "early_stopping_rounds" in model_params:
        params_to_disable["early_stopping_rounds"] = None

    if "early_stopping_round" in model_params:
        params_to_disable["early_stopping_round"] = None

    if params_to_disable:
        model.set_params(**params_to_disable)

y_train_shuffled = np.random.permutation(y_train.to_numpy())

model.fit(
    X_train.to_numpy(dtype=np.float32, copy=False),
    y_train_shuffled.astype(np.float32, copy=False),
)
pred = model.predict(
    X_test.to_numpy(dtype=np.float32, copy=False)
)

print("Shuffled target R2:", r2_score(y_test, pred))
print("Shuffled target MAE:", mean_absolute_error(y_test, pred))
print("Shuffled target RMSE:", root_mean_squared_error(y_test, pred))
