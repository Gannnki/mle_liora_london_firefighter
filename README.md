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

## ▶️ One-Command Pipeline

Double-click from the repo root:

```text
pipeline_starter.bat
```

Run the full XGBoost preprocessing, training, and evaluation flow:

```bash
.venv\Scripts\python.exe src/pipeline.py
```

The pipeline runs:

```text
preprocess -> train -> package_inference -> evaluate
```

It creates the production inference artifact:

```text
artifacts/production/inference_pipeline.pkl
```

This file bundles the fitted encoder, scaler, and XGBoost model so an API can load one object for prediction.

Useful options:

```bash
.venv\Scripts\python.exe src/pipeline.py --skip-preprocess
.venv\Scripts\python.exe src/pipeline.py --skip-preprocess --skip-train
.venv\Scripts\python.exe src/pipeline.py --run-name "xgboost baseline"
```

Each run writes metadata to:

```text
output/runs/<run_id>/run_metadata.json
```

The pipeline also logs each run to MLflow by default using:

```text
sqlite:///output/mlflow.db
```

View runs with:

```bash
.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///output/mlflow.db
```

Use `--disable-mlflow` to run without experiment tracking.

---

## ✅ Tests and CI

Run tests locally:

```bash
.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions automatically runs the same test suite on every push and pull request using:

```text
.github/workflows/ci.yml
requirements-dev.txt
```

The CI checks core configuration, feature-engineering behaviour, pipeline helper logic, MLflow tracking helpers, and the production inference pipeline wrapper.

---

## 🖥️ Streamlit App

Double-click from the repo root:

```text
streamlit_starter.bat
```

The starter switches into `src/display_streamlit/` before launching so relative paths such as `data_streamlit/` resolve correctly.

The app opens at:

```text
http://localhost:8501
```

Command-line equivalent:

```bash
cd src\display_streamlit
..\..\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The Streamlit prototype is designed for:

* Scenario testing
* Model insight presentation
* MLflow UI Presentation
* Explaining how distance, location, time, and operational pressure affect predicted attendance time

---

## 🚀 Future Work

The future direction is to move from offline prediction toward real-time operational support.

Important next steps:

* Connect to live routing and traffic information
* Add live weather context
* Include current station and appliance availability
* Extend from response-time prediction to predictive dispatch planning
* Deploy the trained model behind a FastAPI service and let Streamlit call the API instead of loading model files directly

With these extensions, the system could evolve from a reporting and planning tool into a real-time decision-support layer for emergency response operations.
