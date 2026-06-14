"""MLflow tracking helpers for pipeline runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import mlflow
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent


def log_config_files(
    config_paths: Iterable[str | Path],
    artifact_dir: str = "configs",
) -> None:
    """Log configuration files as MLflow artifacts."""
    for config_path in config_paths:
        path = _resolve_path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Config path is not a file: {path}")

        mlflow.log_artifact(str(path), artifact_path=artifact_dir)


def log_data_fingerprint(
    data_paths: Iterable[str | Path],
    artifact_dir: str = "data",
) -> list[dict]:
    """Log dataset fingerprints and return the fingerprint records."""
    fingerprints = []

    for data_path in data_paths:
        path = _resolve_path(data_path)
        fingerprint = _path_fingerprint(path)
        fingerprints.append(fingerprint)

        name = _safe_metric_or_param_name(path.stem)
        mlflow.log_param(f"data_{name}_exists", fingerprint["exists"])

        if fingerprint["exists"] and fingerprint["type"] == "file":
            mlflow.log_param(f"data_{name}_size_bytes", fingerprint["size_bytes"])
            mlflow.log_param(f"data_{name}_sha256", fingerprint["sha256"])

    with _temporary_json_file("data_fingerprint.json", fingerprints) as json_path:
        mlflow.log_artifact(str(json_path), artifact_path=artifact_dir)

    return fingerprints


def log_metrics_csv(
    metrics_csv_path: str | Path,
    artifact_dir: str = "metrics",
    metric_prefix: str | None = None,
) -> pd.DataFrame:
    """Log a metrics CSV as an artifact and numeric values as MLflow metrics."""
    path = _resolve_path(metrics_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")

    metrics_df = pd.read_csv(path)
    mlflow.log_artifact(str(path), artifact_path=artifact_dir)

    for row_index, row in metrics_df.iterrows():
        row_label = _row_label(row, row_index)
        for column_name, value in row.items():
            if not pd.api.types.is_number(value):
                continue

            metric_name_parts = [
                part
                for part in [metric_prefix, row_label, str(column_name)]
                if part
            ]
            metric_name = _safe_metric_or_param_name("_".join(metric_name_parts))
            mlflow.log_metric(metric_name, float(value))

    return metrics_df


def log_artifacts(
    artifact_paths: Iterable[str | Path],
    artifact_dir: str = "artifacts",
) -> None:
    """Log files or directories as MLflow artifacts."""
    for artifact_path in artifact_paths:
        path = _resolve_path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact path not found: {path}")

        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=f"{artifact_dir}/{path.name}")
        else:
            mlflow.log_artifact(str(path), artifact_path=artifact_dir)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _path_fingerprint(path: Path) -> dict:
    fingerprint = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return fingerprint

    if path.is_file():
        fingerprint.update(
            {
                "type": "file",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    elif path.is_dir():
        fingerprint.update(
            {
                "type": "directory",
                "file_count": sum(1 for item in path.rglob("*") if item.is_file()),
            }
        )

    return fingerprint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_label(row: pd.Series, row_index: int) -> str:
    for candidate in ["Model", "model", "split", "Split", "dataset", "Dataset"]:
        if candidate in row and pd.notna(row[candidate]):
            return str(row[candidate])
    return f"row_{row_index}"


def _safe_metric_or_param_name(name: str) -> str:
    safe_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in name.strip()
    )
    return safe_name.strip("_") or "value"


class _temporary_json_file:
    def __init__(self, filename: str, payload: object):
        self.path = BASE_DIR / "output" / "tmp_tracking" / filename
        self.payload = payload

    def __enter__(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self.path

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
