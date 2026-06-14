"""Run the XGBoost ML pipeline from preprocessing through final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from tracking import (
    log_artifacts,
    log_config_files,
    log_data_fingerprint,
    log_metrics_csv,
)


BASE_DIR = Path(__file__).resolve().parent.parent

STAGES = [
    {
        "name": "preprocess",
        "script": BASE_DIR / "src" / "preprocessing_main.py",
        "outputs": [
            BASE_DIR / "output" / "data_splits",
            BASE_DIR / "output" / "encoders",
            BASE_DIR / "output" / "scalers",
            BASE_DIR / "artifacts" / "encoders" / "feature_encoder.pkl",
            BASE_DIR / "artifacts" / "scalers" / "feature_scaler.pkl",
        ],
    },
    {
        "name": "train",
        "script": BASE_DIR / "src" / "modeling_main.py",
        "outputs": [
            BASE_DIR / "artifacts" / "best_models" / "best_model.pkl",
            BASE_DIR / "output" / "predictions" / "y_pred_validation.csv",
        ],
    },
    {
        "name": "evaluate",
        "script": BASE_DIR / "src" / "model_eval.py",
        "outputs": [
            BASE_DIR / "output" / "predictions" / "model_eval_metrics.csv",
            BASE_DIR / "output" / "predictions" / "y_pred_test_eval.csv",
        ],
    },
]

TRACKED_INPUTS = [
    BASE_DIR / "config" / "pipeline_config.yaml",
    BASE_DIR / "config" / "xgboost_only.yaml",
    BASE_DIR / "requirements.txt",
    BASE_DIR / "data" / "dataset_with_filtered_distance_speed.csv",
]

CONFIG_FILES = [
    BASE_DIR / "config" / "pipeline_config.yaml",
    BASE_DIR / "config" / "xgboost_only.yaml",
]

DATA_FILES = [
    BASE_DIR / "data" / "dataset_with_filtered_distance_speed.csv",
]

METRICS_CSV_PATH = BASE_DIR / "output" / "predictions" / "model_eval_metrics.csv"

TRACKED_ARTIFACTS = [
    BASE_DIR / "artifacts" / "best_models" / "best_model.pkl",
    BASE_DIR / "output" / "predictions",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the XGBoost pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Run the XGBoost pipeline end to end.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Reuse existing preprocessed splits, encoders, and scalers.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse the existing trained best model.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip final train/validation/test evaluation.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional readable name stored in the run metadata.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///output/mlflow.db",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment-name",
        default="london-firefighter-xgboost",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Run the pipeline without MLflow tracking.",
    )
    return parser.parse_args()


def main() -> int:
    """Create run metadata, optionally start MLflow, and execute selected stages."""
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = BASE_DIR / "output" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "run_name": args.run_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "status": "running",
        "repo_root": str(BASE_DIR),
        "python_executable": sys.executable,
        "pipeline": "xgboost",
        "git": git_metadata(),
        "inputs": input_metadata(TRACKED_INPUTS),
        "stages": [],
    }

    metadata_path = run_dir / "run_metadata.json"
    write_metadata(metadata_path, metadata)

    selected_stages = [
        stage
        for stage in STAGES
        if not should_skip_stage(stage["name"], args)
    ]

    print(f"\nRun id: {run_id}")
    print(f"Metadata: {metadata_path}")
    print("Stages:", ", ".join(stage["name"] for stage in selected_stages))

    if args.disable_mlflow:
        return run_pipeline(selected_stages, metadata, metadata_path)

    setup_mlflow(args)
    mlflow_run_name = args.run_name or run_id

    with mlflow.start_run(run_name=mlflow_run_name):
        log_initial_mlflow_metadata(metadata)
        exit_code = run_pipeline(selected_stages, metadata, metadata_path)
        log_final_mlflow_outputs(metadata, metadata_path)
        return exit_code


def run_pipeline(
    selected_stages: list[dict],
    metadata: dict,
    metadata_path: Path,
) -> int:
    """Run each selected stage and persist run metadata after every stage."""
    pipeline_start_time = time.perf_counter()

    try:
        for stage in selected_stages:
            metadata["stages"].append(run_stage(stage))
            write_metadata(metadata_path, metadata)
    except subprocess.CalledProcessError as exc:
        metadata["total_duration_seconds"] = round(
            time.perf_counter() - pipeline_start_time,
            2,
        )
        metadata["status"] = "failed"
        metadata["failed_stage"] = exc.cmd
        metadata["returncode"] = exc.returncode
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_metadata(metadata_path, metadata)
        print(f"\nPipeline failed with return code {exc.returncode}.")
        print(f"Metadata saved to: {metadata_path}")
        return exc.returncode

    metadata["total_duration_seconds"] = round(
        time.perf_counter() - pipeline_start_time,
        2,
    )
    metadata["status"] = "completed"
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["outputs"] = output_metadata(selected_stages)
    write_metadata(metadata_path, metadata)

    print("\nPipeline completed successfully.")
    print(f"Total runtime: {format_duration(metadata['total_duration_seconds'])}")
    print(f"Metadata saved to: {metadata_path}")
    return 0


def setup_mlflow(args: argparse.Namespace) -> None:
    """Configure the MLflow tracking URI and experiment for this run."""
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)


def log_initial_mlflow_metadata(metadata: dict) -> None:
    """Log config files, data fingerprints, and run identity before training."""
    mlflow.set_tag("pipeline", "xgboost")
    mlflow.set_tag("run_id", metadata["run_id"])
    mlflow.set_tag("status", "running")

    mlflow.log_param("run_id", metadata["run_id"])
    mlflow.log_param("pipeline", metadata["pipeline"])
    mlflow.log_param("python_executable", metadata["python_executable"])

    git_info = metadata.get("git", {})
    for key, value in git_info.items():
        if value is not None:
            mlflow.log_param(f"git_{key}", value)

    log_config_files(CONFIG_FILES)
    log_data_fingerprint(DATA_FILES)


def log_final_mlflow_outputs(metadata: dict, metadata_path: Path) -> None:
    """Log final metrics, artifacts, stage durations, and run metadata to MLflow."""
    mlflow.set_tag("status", metadata["status"])

    for stage in metadata.get("stages", []):
        mlflow.log_metric(
            f"stage_{stage['name']}_duration_seconds",
            stage["duration_seconds"],
        )

    if "total_duration_seconds" in metadata:
        mlflow.log_metric(
            "pipeline_total_duration_seconds",
            metadata["total_duration_seconds"],
        )

    if METRICS_CSV_PATH.exists():
        log_metrics_csv(METRICS_CSV_PATH)

    existing_artifacts = [
        artifact_path
        for artifact_path in TRACKED_ARTIFACTS
        if artifact_path.exists()
    ]
    if existing_artifacts:
        log_artifacts(existing_artifacts)

    mlflow.log_artifact(str(metadata_path), artifact_path="metadata")


def should_skip_stage(stage_name: str, args: argparse.Namespace) -> bool:
    """Return whether a stage should be skipped for the parsed CLI options."""
    return (
        (stage_name == "preprocess" and args.skip_preprocess)
        or (stage_name == "train" and args.skip_train)
        or (stage_name == "evaluate" and args.skip_eval)
    )


def run_stage(stage: dict) -> dict:
    """Execute one pipeline stage as a subprocess and return timing metadata."""
    stage_name = stage["name"]
    script_path = stage["script"]
    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()

    print(f"\n=== {stage_name.upper()} ===")
    print(f"Running: {script_path}")

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        check=True,
    )

    duration_seconds = round(time.perf_counter() - start_time, 2)
    finished = datetime.now(timezone.utc)

    print(
        f"Finished {stage_name} in {format_duration(duration_seconds)} "
        f"({duration_seconds:.2f}s)."
    )

    return {
        "name": stage_name,
        "script": str(script_path),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": duration_seconds,
        "outputs": [path_metadata(path) for path in stage["outputs"]],
    }


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    minutes, remaining_seconds = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def git_metadata() -> dict:
    """Collect commit, branch, and dirty-worktree information when git is available."""
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(run_git(["status", "--porcelain"])),
    }


def run_git(args: list[str]) -> str | None:
    """Run a git command in the repository and return stripped stdout."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return result.stdout.strip()


def input_metadata(paths: list[Path]) -> list[dict]:
    """Build metadata records for tracked input files."""
    return [path_metadata(path, include_hash=True) for path in paths]


def output_metadata(stages: list[dict]) -> list[dict]:
    """Build metadata records for all declared outputs from completed stages."""
    outputs = []
    for stage in stages:
        outputs.extend(path_metadata(path) for path in stage["outputs"])
    return outputs


def path_metadata(path: Path, include_hash: bool = False) -> dict:
    """Return existence, type, size/count, and optional hash metadata for a path."""
    metadata = {
        "path": str(path),
        "exists": path.exists(),
    }

    if not path.exists():
        return metadata

    if path.is_file():
        metadata["type"] = "file"
        metadata["size_bytes"] = path.stat().st_size
        if include_hash:
            metadata["sha256"] = sha256_file(path)
    elif path.is_dir():
        metadata["type"] = "directory"
        metadata["file_count"] = sum(1 for item in path.rglob("*") if item.is_file())

    return metadata


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest for a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(path: Path, metadata: dict) -> None:
    """Write run metadata as pretty-printed JSON."""
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
