"""Serve the exact inference bundle produced by the training pipeline."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.inference_pipeline import load_inference_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]


class ArtifactError(RuntimeError):
    """The configured production bundle is missing or incompatible."""


class PredictionService:
    def __init__(self, pipeline_path: Path | None = None) -> None:
        path = pipeline_path or Path(os.environ.get(
            "LFB_MODEL_PATH", str(REPO_ROOT / "artifacts/production/inference_pipeline.pkl")
        ))
        try:
            self.pipeline = load_inference_pipeline(path)
            self.pipeline.validate_artifacts()
        except Exception as exc:
            raise ArtifactError(
                f"Cannot load inference bundle at {path}. "
                "Run uv run python src/build_inference_pipeline.py first. "
                f"Reason: {exc}"
            ) from exc

    def predict(self, template: dict[str, Any], settings: dict[str, Any]) -> dict[str, float | int]:
        started = time.perf_counter()
        seconds = float(np.asarray(self.pipeline.predict_scenario(template, settings)).reshape(-1)[0])
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Model returned an invalid attendance time")
        return {
            "attendance_time_seconds": seconds,
            "minutes": int(seconds // 60),
            "remaining_seconds": int(seconds % 60),
            "inference_ms": (time.perf_counter() - started) * 1000,
        }
