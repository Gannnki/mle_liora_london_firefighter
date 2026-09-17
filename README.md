# Liora 🔥 London Fire Brigade 

## 📌 Project Overview

This project predicts **London Fire Brigade first-vehicle attendance time**: the time between a station being mobilised and the first fire engine arriving at the incident scene.

The goal is to build a practical offline planning and diagnostic tool. It is not a live dispatch replacement, but it can help explain delays, identify difficult areas, and support operational planning.

The project combines:

* Historical incident and mobilisation records
* Road-distance and location enrichment
* Time, risk, and operational-pressure features
* XGBoost modelling and SHAP-based interpretation
* Streamlit scenario simulation and MLflow experiment tracking

---

## 👥 Team Members

* Yu
* Laura
* Khoi
* Kilian

---

## 📊 Data Sources

We use two official datasets:

### 1. Incident Records

Contains details about each incident:

* Time, location, incident type
* Property and geographic information

🔗 https://data.london.gov.uk/dataset/london-fire-brigade-incident-records

---

### 2. Mobilisation Records

Contains details of each dispatched fire engine:

* Dispatch, travel, and arrival times
* Station and resource information

🔗 https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records

### Final Modelling Scope

The original incident and mobilisation datasets each contain about **1.9 million records**, covering 2009 to early 2026. For the final modelling scope, we focus on the more recent period from **2021 to February 2026**.

This was chosen because the post-2021 records are more consistent and better reflect current London conditions, including traffic patterns, station operations, and urban structure.

After filtering and preparation, the project works with approximately:

* **1 million mobilisations**
* **680,000 incidents**

This provides enough data volume for modelling while reducing reliance on outdated historical behaviour.

---

## 🧩 Feature Engineering

A major improvement of the project is that it does not rely only on the raw London Fire Brigade files. The original data describes where incidents happened and which stations responded, but it does not fully describe how difficult the journey was.

The final feature set includes:

* **Road distance** between the incident and the responding station
* **Time context**, including rush hour, night shift, weekends, and holidays
* **Location context**, including central London indicators, borough information, distance to city centre, and route complexity
* **Operational context**, including recent station and borough deployment history
* **Risk flags** for special incident situations, such as road vehicle fires, outdoor incidents, or unusually large emergencies

The road-distance feature became the strongest single model driver, contributing more than **28%** of the final model explanation.

---

## 📈 Key Results

The final XGBoost model performs clearly better than a simple historical-average baseline and gives useful predictions for planning and analysis.

Main results:

* **Mean absolute error:** about **49.6 seconds**
* **Explained variation:** about **58.7%**
* **90% of predictions:** within roughly **2 minutes** of the actual attendance time
* **Most important driver:** road distance from station to incident

The model captures meaningful operational patterns: distance, central London conditions, time of day, route structure, and local operational pressure all affect attendance time.

Very large delays remain harder to predict because they often depend on live events, such as traffic jams, road closures, severe weather, or temporary station availability.

---

## ⚠️ Limitations

The project is built carefully to avoid data leakage. Fields that would only be known after arrival, such as recorded delay outcomes or post-incident response variables, are excluded from model training.

Current limitations:

* No live traffic, road closure, or weather feed
* Some location information is limited for privacy reasons
* Very long response-time outliers are difficult to predict from historical data alone
* Road-distance enrichment is computationally expensive
* The current system is an offline planning tool, not a real-time dispatch system

---

## 🧪 Scientific Contribution

This project shows that emergency response-time prediction improves when raw incident records are enriched with road-network distance, temporal context, operational history, and risk indicators.

It also provides interpretable evidence that attendance time is shaped not only by geographic distance, but also by route complexity, central London conditions, time of day, and station-level operational pressure.

---

## 📁 Project Resources

### Required Local Geo Files

Before running preprocessing or the one-command pipeline, make sure the `utils/` folder contains:

```text
utils/greater-london.gpkg
utils/London_Boroughs.gpkg
```

These files are used to generate borough and road-intersection features. The derived cache file `utils/london_intersections_27700.gpkg` can be generated by the preprocessing pipeline if it is missing.

### Documentation

* Project description:
  https://docs.google.com/document/d/1368CKhHYetKFK2qU7VwEnXspXk-ZTGqBtSWGnHBcJ7M

* Methodology:
  https://docs.google.com/document/d/1sbgOhiBA4hIYgkO-wrEDZrAejmoz9Ezr5EEwDqsdGMw

---

### Data Auditing Template

https://docs.google.com/spreadsheets/d/1JI7_DBcSXJl5UxB8VY-Ybyr3T1NdK0PCDkO1t-ZRjj8

---

## 🧠 Key Insight

* The problem is fundamentally a **regression task**
* Avoid data leakage by excluding arrival timestamps and derived response variables
* Use a strict time-based split instead of random splitting
* Fit categorical encoders and scalers on training data only
* Combine incident context, station context, road distance, and mobilisation information for prediction

---

## Local setup with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run commands
from the repository root. Python 3.12 is selected by `.python-version`.
`pyproject.toml` and the committed `uv.lock` are the dependency source of truth;
the former requirements files have been replaced.

```bash
uv sync --locked
uv run --locked pytest -q
```

The default environment includes training, API, UI and development tools. For a
smaller environment, select only the required groups:

```bash
uv sync --locked --no-default-groups --group api
uv sync --locked --no-default-groups --group ui
uv sync --locked --no-default-groups --group train
```

Use matching group options with `uv run`, or `uv run --no-sync` after the selected
sync, to avoid restoring the default groups. Optional notebooks/SHAP use
`uv sync --locked --group analysis`; LightGBM/CatBoost use `--group advanced`.
Experimental TensorFlow/PyTorch scripts are not included in the supported tabular
runtime and need separately managed compatible environments.

To update dependencies, edit them with `uv add` / `uv add --group <group>`, run
`uv lock`, and commit both files. Training and API use the same model dependency
group and lockfile. Linux, Windows and Docker use the same Python minor version.

## Train and package the model

The supported training entry point is:

```bash
uv run --locked python src/pipeline.py
```

It runs `preprocess -> train -> package_inference -> evaluate` and records runs
under `output/runs/`. It currently starts from the prepared input file
`data/dataset_with_filtered_distance_speed.csv`; downloading and enriching the
original LFB data is not automated by this command. The local geo resources
listed above, including station coordinates, must also be available in `utils/`.

```bash
uv run --locked python src/pipeline.py --skip-preprocess
uv run --locked python src/pipeline.py --skip-preprocess --skip-train
uv run --locked python src/pipeline.py --run-name "xgboost baseline"
```

If you already have these fitted components:

```text
artifacts/best_models/best_model.pkl
artifacts/encoders/feature_encoder.pkl
artifacts/scalers/feature_scaler.pkl
```

package them without retraining:

```bash
uv run --locked python src/build_inference_pipeline.py
uv run --locked python scripts/smoke_inference.py
```

This creates `artifacts/production/inference_pipeline.pkl`. The bundle contains
the fitted encoder, scaler, model and runtime metadata. Saved training/validation
frames are stripped from the deployment bundle; existing source pickles remain
untouched. Feature order/count is checked before packaging and loading. These
checks do not prove that separately supplied legacy components came from the
same training run: always package a matching set of components.

The API loads only this bundle, never falls back to separate Streamlit models,
and can use a different bundle via `LFB_MODEL_PATH`. Restart the API after
repackaging. To roll back locally, restore a previously saved bundle and restart.
Packaging is atomic, but model quality promotion/version registry is not yet
implemented; the pipeline is intended for one local training run at a time.

MLflow uses persistent local files in `output/`:

```bash
uv run --locked mlflow ui --backend-store-uri sqlite:///output/mlflow.db
```

Use `--disable-mlflow` on the pipeline if tracking is unnecessary.

## Run FastAPI and Streamlit locally

After building the inference bundle, open two terminals in the repository root:

```bash
# Terminal 1
uv run --locked uvicorn src.display_streamlit.api:app --host 127.0.0.1 --port 8000

# Terminal 2
uv run --locked python scripts/run_ui.py
```

- UI: http://localhost:8501
- API health: http://localhost:8000/health
- API documentation: http://localhost:8000/docs

`run_ui.py` sets the working directory so that all page images and CSVs resolve
correctly. Windows users can also run `streamlit_starter.bat` (starts both
processes) or `pipeline_starter.bat` (training).

Streamlit calls `LFB_API_URL`, defaulting to `http://127.0.0.1:8000`.
The simulator changes scenario controls but retains the selected historical
snapshot's location, public-holiday indicator and operational history. It is
not a live routing or dispatch service. Shared time/property/risk feature
functions are used by training and scenario inference. Rush-hour windows follow
the existing training definition, including weekends. Missing features actually
used during training are rejected rather than silently filled with zero.

Check all supplied demo scenarios against a running API:

```bash
uv run --locked python scripts/smoke_inference.py --api-url http://127.0.0.1:8000
```

## Run locally with Docker Compose

Install Docker with Compose v2. No AWS account or paid cloud services are needed.
Image builds require internet access for base images and locked dependencies.
The three Docker targets install only their own uv dependency group:

| Target | Purpose | Data |
|---|---|---|
| `trainer` | Preprocessing, XGBoost training, packaging, evaluation | Local input/output mounts |
| `api` | Load the inference bundle and serve predictions | Production bundle mounted read-only |
| `ui` | Streamlit pages and scenario controls | Small demo CSV and images included |

First build the bundle using the command above, then start the app:

```bash
docker compose up --build -d api ui
docker compose ps
docker compose logs -f api ui
```

Open http://localhost:8501. Ports are bound to localhost. Compose sets
`LFB_API_URL=http://api:8000`; the UI waits for the API's model-aware health check.
Data and model files are excluded from image layers. Model files stay on your
machine, outside the container lifecycle.

To train entirely in Docker, first place the required data and geo files in the
local `data/` and `utils/` directories and create writable output directories:

```bash
mkdir -p artifacts output mlruns
docker compose --profile training build trainer
docker compose --profile training run --rm trainer
# Or only package existing fitted components:
docker compose --profile training run --rm trainer python src/build_inference_pipeline.py
```

On Linux, if your user/group IDs differ from 1000, set `LOCAL_UID` and `LOCAL_GID`
to your `id -u` and `id -g` values before running the trainer. This keeps mounted
outputs writable by your user. The `utils/` mount is writable because geographic
caches may be generated during preprocessing. `output/` persists MLflow's SQLite
database, metrics and run metadata; `artifacts/` persists fitted models and
`mlruns/` persists MLflow artifact uploads. To view runs created inside Docker,
serve MLflow inside the same container paths:

```bash
docker compose --profile training run --rm -p 127.0.0.1:5000:5000 trainer mlflow ui --host 0.0.0.0 --backend-store-uri sqlite:///output/mlflow.db
```

MLflow stores absolute artifact locations. Keep local and Docker tracking
databases separate if you switch execution environments, or old artifact links
may point at paths that only exist in the other environment.

After retraining/repackaging:

```bash
docker compose restart api
uv run --locked python scripts/smoke_inference.py --api-url http://127.0.0.1:8000
```

Stop the local services with `docker compose down`. Your mounted data, model and
MLflow files remain. Do not run multiple trainers against the same output paths.

## Tests and CI

```bash
uv run --locked pytest -q
docker compose --profile training config --quiet
```

GitHub Actions installs locked uv dependencies, runs the unit/integration suite,
validates Compose and builds all three images. Integration tests fit a small
XGBoost model, package it, and verify that HTTP predictions match offline
predictions. They also check unavailable/incompatible models and invalid inputs.
Real local model files are intentionally not required by CI.
