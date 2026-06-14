import streamlit as st


st.set_page_config(
    page_title="MLflow Tracking",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .dashboard-card {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
    }
    .card-title {
        color: #0f172a;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .card-copy {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.55;
        margin: 0;
    }
    .evidence-chip {
        display: inline-block;
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .section-note {
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 16px 18px;
        border-radius: 8px;
        color: #334155;
        line-height: 1.55;
        margin-bottom: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 MLflow Experiment Tracking")
st.caption("Page 7 • Reproducible runs, metrics, artifacts, and model lineage")
st.write("")

st.markdown(
    """
    <div class="section-note">
        MLflow is used as the experiment ledger for the XGBoost pipeline. Each run links the training configuration,
        git state, data fingerprint, model metrics, prediction outputs, and saved artifacts, making model comparison
        auditable instead of relying on terminal logs or loose CSV files.
    </div>
    """,
    unsafe_allow_html=True,
)

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown(
        """
        <div class="dashboard-card">
            <span class="evidence-chip">Run lineage</span>
            <div class="card-title">Configuration Traceability</div>
            <p class="card-copy">
                The pipeline logs <code>pipeline_config.yaml</code>, <code>xgboost_only.yaml</code>,
                the active git commit, and a dataset fingerprint so each model can be traced back to its exact inputs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_col2:
    st.markdown(
        """
        <div class="dashboard-card">
            <span class="evidence-chip">Evaluation</span>
            <div class="card-title">Comparable Metrics</div>
            <p class="card-copy">
                Validation and test metrics are logged into one run record, including MAE, RMSE, R², RMSLE,
                P90 error, and stage runtime.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with summary_col3:
    st.markdown(
        """
        <div class="dashboard-card">
            <span class="evidence-chip">Artifacts</span>
            <div class="card-title">Model Output Registry</div>
            <p class="card-copy">
                The best model, prediction CSV files, and run metadata are grouped together, reducing the risk of
                mixing results from different experiments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.subheader("MLflow UI Presentation")

top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)

with top_left:
    st.markdown('<div class="card-title">1. Experiment Run Table</div>', unsafe_allow_html=True)
    st.write("The run table shows all tracked XGBoost executions in one experiment, including run names and status.")
    st.image(
        "data_streamlit/mlflow_run.PNG",
        use_container_width=True,
        caption="MLflow experiment run table",
    )

with top_right:
    st.markdown('<div class="card-title">2. Run-Specific Metadata</div>', unsafe_allow_html=True)
    st.write("A single run page connects parameters, git metadata, config files, and pipeline identity.")
    st.image(
        "data_streamlit/mlflow_run_specific.PNG",
        use_container_width=True,
        caption="Detailed view for one pipeline run",
    )

with bottom_left:
    st.markdown('<div class="card-title">3. Model Metrics</div>', unsafe_allow_html=True)
    st.write("Metrics are captured directly from the evaluation CSV so model quality can be compared across runs.")
    st.image(
        "data_streamlit/mlflow_modelmetrics.PNG",
        use_container_width=True,
        caption="Logged validation and test metrics",
    )

with bottom_right:
    st.markdown('<div class="card-title">4. Artifacts & Outputs</div>', unsafe_allow_html=True)
    st.write("Artifacts preserve the model output, predictions, and metadata produced by the pipeline run.")
    st.image(
        "data_streamlit/mlflow_artifacts.PNG",
        use_container_width=True,
        caption="Tracked model and pipeline artifacts",
    )

st.write("")
st.markdown(
    """
    <div class="dashboard-card">
        <div class="card-title">Why this matters</div>
        <p class="card-copy">
            The project can now answer practical MLE questions: which configuration produced the current model,
            which data version was used, how long each stage took, and whether a newer run actually improved the
            validation/test metrics. This is the bridge between a notebook-style result and a reproducible ML pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
