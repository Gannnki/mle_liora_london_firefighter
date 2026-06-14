"""Build the production inference pipeline artifact from fitted components."""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.inference_pipeline import build_inference_pipeline


ENCODER_PATH = BASE_DIR / "artifacts" / "encoders" / "feature_encoder.pkl"
SCALER_PATH = BASE_DIR / "artifacts" / "scalers" / "feature_scaler.pkl"
MODEL_PATH = BASE_DIR / "artifacts" / "best_models" / "best_model.pkl"
OUTPUT_PATH = BASE_DIR / "artifacts" / "production" / "inference_pipeline.pkl"


def main() -> None:
    """Create the single production pickle used by FastAPI inference."""
    build_inference_pipeline(
        encoder_path=ENCODER_PATH,
        scaler_path=SCALER_PATH,
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
    )
    print(f"Inference pipeline saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
