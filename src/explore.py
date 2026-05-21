# I will delete later

# explore my training data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PATH_X_train = BASE_DIR / "output/scalers/X_train_scaled.csv"
PATH_X_test = BASE_DIR / "output/scalers/X_test_scaled.csv"

PATH_y_train_log = BASE_DIR / "output/data_splits/y_train_log.csv"
PATH_y_test_log = BASE_DIR / "output/data_splits/y_test_log.csv"

PATH_y_train = BASE_DIR / "output/data_splits/y_train.csv"
PATH_y_test = BASE_DIR / "output/data_splits/y_test.csv"

import pandas as pd
import numpy as np
y_train = pd.read_csv(PATH_y_train)

for q in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]:
    print(f"Quantile {q}: {y_train.quantile(q)}")

print("median seconds :", np.median(y_train))
print("mean seconds :", np.mean(y_train))
print("max seconds :", np.max(y_train))
print("p 90 seconds :", np.percentile(y_train, 90))
print("p 95 seconds :", np.percentile(y_train, 95))


# we do not have that 
