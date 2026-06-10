import streamlit as st
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="Model Insights", 
    page_icon="🤖", 
    layout="wide"
)

# Inject custom modern CSS 
st.markdown("""
    <style>
    .dashboard-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    .card-title {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .highlight-box {
        background-color: #f8fafc;
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid #64748b;
        margin-bottom: 15px;
    }
    .config-box {
        background-color: #fef2f2;
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid #e63946;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Header Section
st.title("🤖 Model Insights & Explainability")
st.caption("Page 5 • Technical Modeling Scope, Hyperparameter Configurations & SHAP Diagnostics")
st.write("")

# 3. ROW 1: Modeling Scope vs. Configuration (Two Columns)
col_scope, col_config = st.columns(2)

with col_scope:
    st.markdown("""
        <div class="dashboard-card" style="height: 100%;">
            <div class="card-title">⚙️ Production Training Scope</div>
            <p style="color: #334155; line-height: 1.6; margin-bottom: 16px;">
                To guarantee maximum predictive stability and eliminate structural data noise, the finalized 
                gradient boosting pipeline was strictly locked to a defined subset of the historical LFB logs:
            </p>
            <div class="highlight-box">
                <p style="margin: 0; color: #0f172a; font-weight: 500;">
                    📅 <b>Temporal Filter:</b> Only modern records spanning from <b>2021 to 2026</b> were utilized.
                </p>
            </div>
            <div class="highlight-box">
                <p style="margin: 0; color: #0f172a; font-weight: 500;">
                    📈 <b>Data Volume:</b> The model was scaled up and fitted on a massive population of <b>over 600,000 rows</b>.
                </p>
            </div>
            <div class="highlight-box">
                <p style="margin: 0; color: #0f172a; font-weight: 500;">
                    🚒 <b>Operational Constraint:</b> Limited strictly to <b>Pump 1 Only</b> (the primary first-responding fire engine).
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col_config:
    st.markdown("""
        <div class="dashboard-card" style="height: 100%;">
            <div class="card-title">🛠️ XGBoost Architecture: Configuration</div>
            <p style="color: #334155; line-height: 1.6; margin-bottom: 16px;">
                To prevent overfitting on the massive 600k row dataset while incorporating complex, non-linear interaction features, we implemented <b>Heavy Regularization</b>:
            </p>
            <div class="config-box">
                <p style="margin: 0; color: #0f172a; font-size: 0.95rem; line-height: 1.5;">
                    📉 <b>Conservative Learning:</b> Lowered <code>learning_rate</code> to <b>0.015</b> to enforce stable, gradual gradient steps across trees.
                </p>
            </div>
            <div class="config-box">
                <p style="margin: 0; color: #0f172a; font-size: 0.95rem; line-height: 1.5;">
                    🌳 <b>Tree Constraints:</b> Raised <code>min_child_weight</code> to <b>6</b> and introduced <code>gamma: 0.1</code> to explicitly penalize trivial node splits.
                </p>
            </div>
            <div class="config-box">
                <p style="margin: 0; color: #0f172a; font-size: 0.95rem; line-height: 1.5;">
                    🛡️ <b>Structural Regularization:</b> Significantly increased <code>reg_lambda: 9</code> (L2 regularization) and <code>reg_alpha: 0.1</code> (L1 regularization) alongside a tighter column and row <code>subsample: 0.85</code>.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# 4. ROW 2: SHAP Explainability & Global Feature Importance
st.write("")
st.subheader("🧬 SHAP (SHapley Additive exPlanations) & Feature Drivers")

# --- CHART 1: Full Width Hero Chart ---
#st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🎯 1. Global Feature Importance Baseline</div>', unsafe_allow_html=True)
st.write("This global split importance illustrates the absolute mathematical weight each engineered feature contributes across all XGBoost trees:")

try:
    
    st.image("data_streamlit/shap_feature_importance_percent.png", use_container_width=True)
    
    st.markdown("""
    <p style="color: #475569; font-size: 0.9rem; line-height: 1.5; margin-top: 12px;">
    💡 <b>Key Finding:</b> Spatial characteristics heavily dominate the model's architecture. 
    The primary road distance (<i>distance_fire_to_station</i>) captures over <b>28% of the total predictive power</b>, which jumps to more than <b>35%</b> when combined with its non-linear transformation (<i>distance_sqrt</i>). 
    Furthermore, our engineered structural routing proxy (<i>detour_ratio</i>) ranks as the third most vital driver, validating our data auditing phase.
    </p>
    """, unsafe_allow_html=True)
    
except Exception as e:
    st.info("💡 Place your 'shap_feature_importance_percent.png' inside the 'data_streamlit/' directory to display this chart.")

st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# --- CHARTS 2 & 3: Side-by-Side Layout ---
col_shap1, col_shap2 = st.columns(2)

with col_shap1:
    #st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📈 2. SHAP Beeswarm Summary Plot</div>', unsafe_allow_html=True)
    st.write("SHAP Impact: Red indicates high feature values, blue indicates low values. Visualizes how variables accelerate or delay predictions:")
    
    try:
     
        st.image("data_streamlit/shap_beeswarm.png", use_container_width=True)
        
        st.markdown("""
        <p style="color: #475569; font-size: 0.9rem; line-height: 1.5; margin-top: 12px;">
        💡 <b>Key Finding:</b> The beeswarm plot confirms that <i>distance_fire_to_station</i> and its transformation <i>distance_sqrt</i> are the absolute dominant drivers, where high distances (red) push the arrival time up significantly. 
        Crucially, <b>Is_Nightshift</b> shows a clear positive shift (red dots on the right), validating the human-factor latency overhead during late hours. 
        The 'Sum of 506 other features' demonstrates how our highly regularized model successfully aggregates hundreds of sparse categorical splits without destabilizing the core prediction.
        </p>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.info("💡 Place your 'shap_beeswarm.png' inside the 'data_streamlit/' directory to display this chart.")
        
    st.markdown('</div>', unsafe_allow_html=True)

with col_shap2:
    #st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔍 3. Model Residuals & Error Diagnostics</div>', unsafe_allow_html=True)
    st.write("Advanced diagnostic plotting comparing prediction errors (Residuals) against actual target values:")
    
    try:
       
        st.image("data_streamlit/residual_plot.png", use_container_width=True)
        
        st.markdown("""
        <p style="color: #475569; font-size: 0.9rem; line-height: 1.5; margin-top: 12px;">
        💡 <b>Key Finding:</b> The residual plot uncovers the explicit boundaries of our static tabular dataset. 
        For very fast dispatches (Actual < 200s), the negative residuals prove a systematic <b>regression-to-the-mean</b> effect, where the model overestimates rapid arrivals. 
        Conversely, for long-duration responses (Actual > 600s), residuals spike heavily into positive territory (underestimation), visually confirming our <b>Outlier Gap</b> caused by the absence of real-time traffic telemetry.
        </p>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.info("💡 Place your 'residual_plot.png' inside the 'data_streamlit/' directory to display this chart.")
        
    st.markdown('</div>', unsafe_allow_html=True)