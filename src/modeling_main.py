import time
import warnings
from pathlib import Path

from modeling import ModelTrainer


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

PATH_X_train = BASE_DIR / "output/scalers/X_train_scaled.csv"
PATH_X_test = BASE_DIR / "output/scalers/X_test_scaled.csv"

PATH_y_train_log = BASE_DIR / "output/data_splits/y_train_log.csv"
PATH_y_test_log = BASE_DIR / "output/data_splits/y_test_log.csv"

PATH_y_train = BASE_DIR / "output/data_splits/y_train.csv"
PATH_y_test = BASE_DIR / "output/data_splits/y_test.csv"

PATH_baseline_model_config = BASE_DIR / "config/baseline_models.yaml"
PATH_advanced_model_config = BASE_DIR / "config/advanced_models.yaml"
PATH_xgboost_model_config = BASE_DIR / "config/xgboost_only.yaml"

PATH_best_model = BASE_DIR / "artifacts/best_models/best_model.pkl"
PATH_y_pred = BASE_DIR / "output/predictions/y_pred_test.csv"


def main():
    start_time = time.perf_counter()

    model_trainer = ModelTrainer(
        PATH_X_train,
        PATH_X_test,
        PATH_y_train_log,
        PATH_y_test_log,
        PATH_y_train,
        PATH_y_test,
    )

    model_trainer.load_models(PATH_xgboost_model_config)
    print("\nAdvanced model training begins:")
    results = model_trainer.run_models()

    print("\nAdvanced model results:")
    print(results.to_string(index=False))

    best_model_name = model_trainer.save_best_model(
        results,
        PATH_best_model,
    )
    print(f"\nSaved best model ({best_model_name}) to: {PATH_best_model}")

    model_trainer.save_predictions(
        best_model_name,
        PATH_y_pred,
        dataset="test",
    )
    print(f"Saved test predictions to: {PATH_y_pred}")

    elapsed_time = time.perf_counter() - start_time
    print(f"\nFinished in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
