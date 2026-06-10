import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import joblib
import time
import os
import sys
import re
from pathlib import Path

# 1. Get stable absolute paths no matter where Streamlit is launched from
CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
REPO_ROOT = APP_DIR.parents[1]

# 2. Add the Streamlit app folder to Python's system path so pickle can find the classes
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from FeatureEngineering import FeatureEncoder, FeatureScaler

# ==========================================
# 1. PAGE CONFIGURATION & MODERN SAAS CSS
# ==========================================
st.set_page_config(
    page_title="Live Incident Simulator", 
    page_icon="🎛️", 
    layout="wide"
)

# Custom SaaS-style CSS for white cards and dashboard alignment
st.markdown("""
    <style>
    .dashboard-card {
        background-color: #f8fafc;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #ef4444;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .card-title {
        color: #0f172a;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .kpi-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .kpi-val {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-lbl {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HEAVY RESOURCE CACHING (REAL PRODUCTION MODELS)
# ==========================================
@st.cache_resource
def load_ml_pipeline():
    """Load production artifacts from the app bundle or the project artifacts folder."""
    artifact_paths = {
        "model": [
            APP_DIR / "models_streamlit" / "best_model.pkl",
            REPO_ROOT / "artifacts" / "best_models" / "best_model.pkl",
        ],
        "scaler": [
            APP_DIR / "models_streamlit" / "feature_scaler.pkl",
            REPO_ROOT / "artifacts" / "scalers" / "feature_scaler.pkl",
        ],
        "encoder": [
            APP_DIR / "models_streamlit" / "feature_encoder.pkl",
            REPO_ROOT / "artifacts" / "encoders" / "feature_encoder.pkl",
        ],
    }

    loaded = {}
    missing = []

    for artifact_name, candidates in artifact_paths.items():
        artifact_path = next((path for path in candidates if path.exists()), None)
        if artifact_path is None:
            missing.append(
                f"{artifact_name}: " + " or ".join(str(path) for path in candidates)
            )
            loaded[artifact_name] = None
            continue

        try:
            loaded[artifact_name] = joblib.load(artifact_path)
        except Exception as e:
            st.error(f"⚠️ Error loading {artifact_name} from {artifact_path}: {e}")
            loaded[artifact_name] = None

    if missing:
        st.error(
            "⚠️ Missing ML artifact(s). Add the trained pickle files to "
            "`src/display_streamlit/models_streamlit/` or the root `artifacts/` folder.\n\n"
            + "\n".join(f"- {item}" for item in missing)
        )

    model_obj = loaded["model"]
    scaler_obj = loaded["scaler"]
    if model_obj is not None and scaler_obj is not None:
        model_feature_count = getattr(model_obj, "n_features_in_", None)
        if model_feature_count is None and hasattr(model_obj, "get_booster"):
            try:
                model_feature_count = model_obj.get_booster().num_features()
            except Exception:
                model_feature_count = None

        scaler_feature_count = len(getattr(scaler_obj, "fitted_columns", []) or [])
        if (
            model_feature_count is not None
            and scaler_feature_count
            and int(model_feature_count) != scaler_feature_count
        ):
            st.error(
                "⚠️ Incompatible ML artifacts: `best_model.pkl` expects "
                f"{int(model_feature_count)} features, but `feature_scaler.pkl` "
                f"was fitted with {scaler_feature_count}. Use the encoder/scaler "
                "saved from the same training run as the model."
            )
            loaded["model"] = None

    return loaded["model"], loaded["scaler"], loaded["encoder"]

# Initialize real production pipeline components
model, scaler, encoder = load_ml_pipeline()

@st.cache_data
def load_demo_scenarios():
    """Loads the 50 random test dataset rows generated in the notebook."""
    try:
        df = pd.read_csv(APP_DIR / "models_streamlit" / "demo_scenarios.csv", index_col=0)
        
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors='coerce')
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors='coerce')
        return df.dropna(subset=["Latitude", "Longitude"])
    except Exception as e:
        st.error(f"⚠️ Error loading 'demo_scenarios.csv': {e}")
        return None

df_scenarios = load_demo_scenarios()

# ==========================================
# 3. MAIN INTERFACE & 50/50 SPLIT LAYOUT WITH NATIVE MAP
# ==========================================
st.title("🎛️ Live Incident Simulator")
st.caption("Page 4 • Real-Time Live Inference via Interactive Geo-Selection")
st.write("")

if df_scenarios is None:
    st.warning("Please ensure 'demo_scenarios.csv' with 50 rows is generated inside 'models_streamlit/'.")
else:
    # Dynamically build dropdown labels combining incident index, station and borough
    df_scenarios["Selector_Label"] = [
        f"Incident #{i+1} - Station: {row['DeployedFromStation_Name']} ({row['IncGeo_BoroughName']})" 
        for i, row in df_scenarios.iterrows()
    ]
    
    st.write("### 📍 1. Select Incident Location via Map Control")
    
   
    col_map, col_geo_info = st.columns(2)
    
    with col_map:
        st.markdown('<div class="card-title">🗺️ London Incident Location Tracker</div>', unsafe_allow_html=True)
        
        # 1. Dropdown selection to choose one of the 50 rows
        selected_label = st.selectbox("Choose an Incident Template to visualize:", options=df_scenarios["Selector_Label"].tolist())
        X_live_template = df_scenarios[df_scenarios["Selector_Label"] == selected_label].copy()
        
        # 2. Extract coordinates of the single selected active point
        inc_lat = float(X_live_template["Latitude"].values[0])
        inc_lon = float(X_live_template["Longitude"].values[0])
        
        # 3. Create a dataframe containing ONLY this single selected point
        # st.map requires the columns to be strictly named 'lat' and 'lon'
        single_point_df = pd.DataFrame({
            "lat": [inc_lat],
            "lon": [inc_lon]
        })
        
        try:
            # Render the selected point with an explicit map center.
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=inc_lat,
                        longitude=inc_lon,
                        zoom=11,
                        pitch=0,
                    ),
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=single_point_df,
                            get_position="[lon, lat]",
                            get_radius=130,
                            get_fill_color=[239, 68, 68, 190],
                            pickable=True,
                        )
                    ],
                    tooltip={"text": "Selected incident"},
                ),
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error rendering map: {e}")
        
    with col_geo_info:
        st.markdown('<div style="height: 55px;"></div>', unsafe_allow_html=True) # Structural alignment spacer
        st.markdown('<div class="card-title">🗺️ Extracted Geographical Metadata</div>', unsafe_allow_html=True)
        
        # Gather immutable geographical data from the selected row
        station = X_live_template["DeployedFromStation_Name"].values[0]
        distance_m = float(X_live_template["distance_fire_to_station"].values[0])
        is_central = int(X_live_template["Is_central_London"].values[0])
        dist_center_km = float(X_live_template["distance_to_city_center_km"].values[0])
        borough = X_live_template["IncGeo_BoroughName"].values[0]
        
        # Display extracted features cleanly as non-editable SaaS KPI blocks
        st.markdown(f"""
            <div class="dashboard-card">
                <p style="margin: 0 0 10px 0; color: #64748b; font-size: 0.85rem; font-weight: bold; text-transform: uppercase;">Active Profile</p>
                <h3 style="margin: 0 0 15px 0; color: #0f172a;">{borough} Region</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div class="kpi-box"><div class="kpi-val" style="color: #ef4444;">{station}</div><div class="kpi-lbl">Responding Station</div></div>
                    <div class="kpi-box"><div class="kpi-val">{distance_m:.1f} m</div><div class="kpi-lbl">Route Distance</div></div>
                    <div class="kpi-box"><div class="kpi-val">{'YES ✅' if is_central == 1 else 'NO ❌'}</div><div class="kpi-lbl">Central London</div></div>
                    <div class="kpi-box"><div class="kpi-val">{dist_center_km:.2f} km</div><div class="kpi-lbl">To London Center</div></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.write("---")
    
    # ==========================================
    # STEP 4: ADDITIONAL CONFIGURATIONS (USER DROPDOWNS)
    # ==========================================
    st.write("### ⏰ 2. Configure Temporal Settings & Incident Properties")
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.markdown('<div class="card-title">🕒 Time & Calendars</div>', unsafe_allow_html=True)
        month = st.slider("Month of the Year:", min_value=1, max_value=12, value=6)
        
        weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        selected_weekday_str = st.selectbox("Day of the Week:", options=list(weekday_map.keys()), index=2)
        weekday_val = weekday_map[selected_weekday_str]
        
        hour = st.slider("Hour of Day (0 - 23):", min_value=0, max_value=23, value=14)
        
        # Apply precise feature engineering constraints live
        is_nightshift_bool = (hour >= 23) or (hour < 6)
        is_rush_hour_bool = ((hour >= 7) and (hour <= 9)) or ((hour >= 16) and (hour <= 19))
        is_weekend_bool = weekday_val >= 5
        
        st.write("🔄 **Live Derived Time Flags:**")
        st.info(f"""
        * **Is Nightshift:** `{'1 (Active)' if is_nightshift_bool else '0 (Inactive)'}`
        * **Is Rush Hour:** `{'1 (Active)' if is_rush_hour_bool else '0 (Inactive)'}`
        * **Is Weekend:** `{'1 (Active)' if is_weekend_bool else '0 (Inactive)'}`
        """)
        
    with col_input2:
        st.markdown('<div class="card-title">🚨 Incident Specifics</div>', unsafe_allow_html=True)
        
        # 1. main choice
        incident_group = st.selectbox("Incident Group Type:", options=["Fire", "Special Service"])
        
        # All options
        all_special_service_options = [
            'NoSpecialService', 'Lift Release', 'RTC', 'Effecting entry/exit',
            'No action (not false alarm)', 'Advice Only', 'Flooding', 'Assist other agencies',
            'Removal of objects from people', 'Suicide/attempts', 'Hazardous Materials incident',
            'Making Safe (not RTC)', 'Animal assistance incidents', 'Evacuation (no fire)',
            'Medical Incident', 'Spills and Leaks (not RTC)', 'Other rescue/release of persons',
            'Other Transport incident', 'Stand By', 'Rescue or evacuation from water', 'Water provision'
        ]
        
        # seperate logic
        if incident_group == "Fire":
            display_options = ["NoSpecialService"]
            is_disabled = True
        else:
            # for specialservice all possible but 'NoSpecialService' 
            display_options = [opt for opt in all_special_service_options if opt != "NoSpecialService"]
            is_disabled = False

        # 2. Dropdown 
        special_service_type = st.selectbox(
            "Special Service Type:", 
            options=display_options,
            index=0,  
            disabled=is_disabled,
            help="This option is only configurable when 'Special Service' is selected above.",
            key=f"special_service_dropdown_{incident_group}"  # dynamic key to change dropdown
        )

    with col_input3:
        st.markdown('<div class="card-title">🏢 Property Characteristics</div>', unsafe_allow_html=True)
        property_category_options = [
            'Non Residential', 'Outdoor', 'Dwelling', 'Road Vehicle', 
            'Outdoor Structure', 'Other Residential', 'Boat', 'Rail Vehicle', 'Aircraft'
        ]
        property_category = st.selectbox("Property Category:", options=property_category_options, index=2)
        
        property_type_options = [
            "Purpose Built Flats/Maisonettes - 4 to 9 storeys", "House - single occupancy",
            "Purpose Built Flats/Maisonettes - Up to 3 storeys", "Self contained Sheltered Housing",
            "Converted Flat/Maisonettes - 3 or more storeys", "Purpose Built Flats/Maisonettes - 10 or more storeys",
            "Converted Flat/Maisonette - Up to 2 storeys", "Car", "Purpose built office", "Single shop"
        ]
        property_type = st.selectbox("Property Type (Top 10):", options=property_type_options)

    st.write("")
    st.write("---")
    

# ==========================================
# 5. ML INFERENCE: FEATURE ENGINEERING & PROCESSING
# ==========================================

def _numcalls_bucket_to_ordinal(series):
    """
    Direct replication of your Notebook ordinal mapping logic.
    Converts text buckets into exact float values.
    """
    bucket_map = {
        "0": 0.0,
        "1": 1.0,
        "2": 2.0,
        "3": 3.0,
        "4-5": 4.5,
        "6-10": 8.0,
        "10+": 12.0,
    }

    numeric_values = pd.to_numeric(
        series,
        errors="coerce",
    )

    mapped_values = (
        series
        .astype(str)
        .str.strip()
        .map(bucket_map)
    )

    return mapped_values.fillna(numeric_values).fillna(0.0)


def _ensure_encoder_input_columns(X, encoder):
    """Supply defaults for configured features not present in demo_scenarios.csv."""
    if encoder is None or not hasattr(encoder, "feature_config"):
        return X

    X_prepared = X.copy()
    for col, cfg in encoder.feature_config.items():
        if col in X_prepared.columns:
            continue

        encoding = cfg.get("encoding")
        if encoding in {"NUMERIC_KEEP", "BINARY_KEEP"}:
            X_prepared[col] = 0
        elif encoding == "CYCLIC":
            X_prepared[col] = 0
        else:
            X_prepared[col] = "Unknown"

    return X_prepared


def _expected_model_features(model, scaler):
    """Return the model/scaler feature order when the artifact exposes it."""
    if scaler is not None and getattr(scaler, "fitted_columns", None):
        return list(scaler.fitted_columns)

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    booster_getter = getattr(model, "get_booster", None)
    if booster_getter is not None:
        try:
            booster_names = booster_getter().feature_names
            if booster_names:
                return list(booster_names)
        except Exception:
            pass

    return None


def _model_expected_feature_count(model):
    """Best-effort feature count discovery across sklearn and XGBoost artifacts."""
    for attr in ("n_features_in_", "n_features_"):
        value = getattr(model, attr, None)
        if value is not None:
            return int(value)

    booster_getter = getattr(model, "get_booster", None)
    if booster_getter is not None:
        try:
            return int(booster_getter().num_features())
        except Exception:
            pass

    return None

# Trigger real inference inside the existing Streamlit framework
if model is None:
    st.warning("Prediction is disabled until `best_model.pkl` is available.")
elif df_scenarios is not None and st.button("Run Real-Time ML Prediction 🚀", use_container_width=True):
    with st.spinner("Executing structural feature transformations & risk calculation..."):
        
        # 1. Start with a fresh copy of the selected 1-row historical template
        X_live = X_live_template.copy()
        
        # 2. Inject primary user selections from the UI widgets
        X_live["Month"] = int(month)
        X_live["Weekday"] = int(weekday_val)
        X_live["Hour"] = int(hour)
        X_live["IncidentGroup"] = incident_group
        X_live["SpecialServiceType"] = special_service_type
        X_live["PropertyCategory"] = property_category
        X_live["PropertyType"] = property_type
   
        
        # 3. Synchronize live-derived temporal binary constraints
        X_live["Is_Nightshift"] = 1 if (hour >= 23 or hour < 6) else 0
        X_live["Is_Rush_Hour"] = 1 if ((hour >= 7 and hour <= 9) or (hour >= 16 and hour <= 19)) and (weekday_val < 5) else 0
        X_live["Is_Weekend"] = 1 if (weekday_val >= 5) else 0
        X_live["Is_SpecialService"] = 1 if incident_group == "Special Service" else 0

        # 4. Property Access Complexity Feature
        X_live["property_access_complexity"] = (
            X_live["PropertyType"]
            .str.contains("Flat|Maisonette|Care|Hospital|School|Sheltered|Estate", case=False, na=False)
            .astype(int)
        )

        # 5. Live Risk Feature Engineering (Direct replication of your Notebook method)
        X_live["risk_property_outdoor"] = (X_live["PropertyCategory"].eq("Outdoor")).astype(int)
        X_live["risk_property_road_vehicle"] = (X_live["PropertyCategory"].eq("Road Vehicle")).astype(int)
        X_live["risk_property_outdoor_structure"] = (X_live["PropertyCategory"].eq("Outdoor Structure")).astype(int)
        
        # Ordinal tracking for calls bucket using your precise logic
        numcalls_ord = _numcalls_bucket_to_ordinal(X_live["NumOfCalls_bucket"])
        X_live["NumOfCalls_ord"] = numcalls_ord
        X_live["NumOfCalls_log"] = np.log1p(numcalls_ord)
        
        X_live["risk_many_calls"] = (X_live["NumOfCalls_ord"] >= 3).astype(int)
        X_live["risk_very_many_calls"] = (X_live["NumOfCalls_ord"] >= 12).astype(int)
        
        X_live["risk_special_service"] = (
            (X_live["Is_SpecialService"] == 1) | (X_live["IncidentGroup"].eq("Special Service"))
        ).astype(int)
        
        X_live["risk_fire"] = (X_live["IncidentGroup"].eq("Fire")).astype(int)
        X_live["risk_noncentral"] = (X_live["Is_central_London"] == 0).astype(int)
        X_live["risk_repeated_call"] = (X_live["Is_RepeatedCall"] == 1).astype(int)
        
        X_live["risk_weekday_4"] = (X_live["Weekday"] == 4).astype(int)
        X_live["risk_weekday_2"] = (X_live["Weekday"] == 2).astype(int)
        X_live["risk_month_3_5_6"] = (X_live["Month"].isin([3, 5, 6])).astype(int)
        X_live["risk_not_nightshift"] = (X_live["Is_Nightshift"] == 0).astype(int)
        X_live["risk_not_weekend"] = (X_live["Is_Weekend"] == 0).astype(int)
        
        # Sum columns horizontally to compute high residual risk score
        risk_cols = [
            "risk_property_outdoor", "risk_property_road_vehicle", "risk_property_outdoor_structure",
            "risk_many_calls", "risk_very_many_calls", "risk_special_service", "risk_fire",
            "risk_noncentral", "risk_repeated_call", "risk_weekday_4", "risk_weekday_2",
            "risk_month_3_5_6", "risk_not_nightshift", "risk_not_weekend"
        ]
        X_live["high_residual_risk_score"] = X_live[risk_cols].sum(axis=1)

        # 6. Risk Interaction Features (Direct replication of your Notebook method)
        X_live["many_calls_x_outdoor"] = X_live["risk_many_calls"] * X_live["risk_property_outdoor"]
        X_live["many_calls_x_road_vehicle"] = X_live["risk_many_calls"] * X_live["risk_property_road_vehicle"]
        X_live["many_calls_x_special"] = X_live["risk_many_calls"] * X_live["risk_special_service"]
        X_live["many_calls_x_noncentral"] = X_live["risk_many_calls"] * X_live["risk_noncentral"]
        X_live["road_vehicle_x_noncentral"] = X_live["risk_property_road_vehicle"] * X_live["risk_noncentral"]
        X_live["outdoor_x_noncentral"] = X_live["risk_property_outdoor"] * X_live["risk_noncentral"]
        X_live["repeated_x_many_calls"] = X_live["risk_repeated_call"] * X_live["risk_many_calls"]
        X_live["fire_x_many_calls"] = X_live["risk_fire"] * X_live["risk_many_calls"]

        # =========================================================================
        # 7. PIPELINE PREPARATION (MATCHING MODELING.PY)
        # =========================================================================
        
        # Step 1: Drop only the UI dropdown helper column
        X_final_input = X_live.drop(columns=["Selector_Label"], errors="ignore")
        X_final_input = _ensure_encoder_input_columns(X_final_input, encoder)
        
        # Step 2: Perform index reset exactly like standard processing steps
        X_final_input = X_final_input.reset_index(drop=True)
            
        # Step 3: Enforce strict datatype matching based on the scenarios dataframe schema
        schema_df = df_scenarios.drop(columns=["Selector_Label"], errors="ignore")
        base_schema_dtypes = schema_df.dtypes.to_dict()
        
        valid_dtypes = {k: v for k, v in base_schema_dtypes.items() if k in X_final_input.columns}
        X_final_input = X_final_input.astype(valid_dtypes)

        # =========================================================================
        # 8. PIPELINE EXECUTION 
        # =========================================================================
        try:
            # Step 1: Run the single live row through your pipeline transformations
            X_encoded = encoder.transform(X_final_input) if encoder is not None else X_final_input.copy()
            X_scaled = scaler.transform(X_encoded) if scaler is not None else X_encoded.copy()

            expected_features = _expected_model_features(model, scaler)
            if expected_features is not None and hasattr(X_scaled, "reindex"):
                X_scaled = X_scaled.reindex(columns=expected_features, fill_value=0)
    
            # Step 2: EXACT REPLICATION OF MODELING.PY (evaluate_model method)
            # We convert the DataFrame into a pure NumPy matrix with float32 datatype.
            
            if hasattr(X_scaled, "to_numpy"):
                X_matrix_final = X_scaled.to_numpy(dtype=np.float32, copy=False)
            else:
                X_matrix_final = np.array(X_scaled, dtype=np.float32)

            # Step 3: Execute Production XGBoost Inference
            expected_count = _model_expected_feature_count(model)
            if expected_count is not None and X_matrix_final.shape[1] != expected_count:
                raise ValueError(
                    f"Feature shape mismatch before prediction: model expects "
                    f"{expected_count}, live pipeline produced {X_matrix_final.shape[1]}."
                )

            start_time = time.time()
            prediction_seconds = model.predict(X_matrix_final)
            inference_time = time.time() - start_time
    
            # Step 4: Extract the numeric scalar value safely from the output structure
            if hasattr(prediction_seconds, "item"):
                pred_val = float(prediction_seconds.item())
            elif hasattr(prediction_seconds, "__len__") and len(prediction_seconds) > 0:
                pred_val = float(prediction_seconds[0])
            else:
                pred_val = float(prediction_seconds)

            # Step 5: Execute log-reversal transform to restore true operational seconds
            pred_val = np.expm1(pred_val)
        
            minutes = int(pred_val // 60)
            seconds = int(pred_val % 60)
    
            st.success(f"Inference successfully calculated in {inference_time*1000:.2f} ms")
            st.markdown(f"""
                <div style="background-color: #f8fafc; padding: 24px; border-radius: 12px; border-left: 6px solid #ef4444; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                    <span style="color: #64748b; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Attendance Time</span>
                    <h1 style="color: #0f172a; margin: 8px 0 4px 0; font-size: 3.2rem; font-weight: 800;">{minutes} Min. {seconds} Sek.</h1>
                </div>
            """, unsafe_allow_html=True)
    
        except Exception as pipeline_error:
            st.error(f"❌ Critical Error during live pipeline transformation or prediction: {pipeline_error}")
