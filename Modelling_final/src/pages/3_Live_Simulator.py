import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Used for loading scikit-learn / xgboost pickles
import time

# 1. Page Configuration
st.set_page_config(
    page_title="Live Incident Simulator", 
    page_icon="🎛️", 
    layout="wide"
)

# Custom SaaS-style CSS for modern white cards
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
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Heavy Resource Caching: Load your real pickle files only ONCE
@st.cache_resource
def load_ml_pipeline():
    try:
        # Path structurally updated to match your 'models_streamlit' folder
        model = joblib.load("models_streamlit/best_model.pkl")
        scaler = joblib.load("models_streamlit/feature_scaler.pkl")
        encoder = joblib.load("models_streamlit/feature_encoder.pkl")
        return model, scaler, encoder
    except Exception as e:
        st.error(f"⚠️ Error loading pickle files from 'models_streamlit/' folder: {e}")
        return None, None, None

# Initialize production pipeline components
model, scaler, encoder = load_ml_pipeline()

# 3. Header Layout
st.title("🎛️ Live Incident Simulator")
st.caption("Page 4 • Real-Time Live Inference using Production XGBoost Pipeline")
st.write("")

if model is None:
    st.warning("Please ensure 'best_model.pkl', 'feature_scaler.pkl', and 'feature_encoder.pkl' are placed inside your 'models_streamlit/' directory.")

st.write("")

# 4. Interactive UI Inputs (Based on your top SHAP drivers)
col_input1, col_input2 = st.columns(2)

with col_input1:
    #st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📍 High-Impact Spatial & Routing Features</div>', unsafe_allow_html=True)
    
    distance_fire_to_station = st.slider("Driven Road Distance to Station (km):", min_value=0.1, max_value=25.0, value=3.2, step=0.1)
    detour_ratio = st.slider("Detour Ratio (Route Circuitousness):", min_value=1.0, max_value=3.0, value=1.2, step=0.05)
    distance_to_city_center_km = st.slider("Distance to London Center (km):", min_value=0.0, max_value=40.0, value=5.5, step=0.5)
    borough_intersection_density = st.slider("Borough Intersection Density (per sq km):", min_value=5.0, max_value=150.0, value=45.0, step=5.0)
    st.markdown('</div>', unsafe_allow_html=True)

with col_input2:
    #st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⏰ Temporal & Incident Context</div>', unsafe_allow_html=True)
    
    hour = st.slider("Hour of Day (0 - 23):", min_value=0, max_value=23, value=15)
    is_nightshift = st.selectbox("Is Nightshift Active? (23:00 - 06:00):", options=["No", "Yes"])
    incident_group = st.selectbox("Incident Group Type:", options=["False Alarm", "Special Service", "Fire"])
    incident_intersection_count_500m = st.slider("Intersections within 500m Radius:", min_value=0, max_value=30, value=8)
    concurrent_same_borough = st.number_input("Concurrent Active Incidents in Same Borough:", min_value=0, max_value=10, value=0)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# 5. Live Production Inference Block
if st.button("Run Real-Time ML Prediction 🚀", use_container_width=True) and model is not None:
    
    with st.spinner("Executing live inference through XGBoost pipeline..."):
        
        # --- STEP 1: Background Data Engineering ---
        distance_sqrt = np.sqrt(distance_fire_to_station)
        distance_squared = distance_fire_to_station ** 2
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        is_nightshift_val = 1 if is_nightshift == "Yes" else 0
        
       
