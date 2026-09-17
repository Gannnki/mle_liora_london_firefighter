import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import json
from pathlib import Path
import requests

# 1. Get stable absolute paths no matter where Streamlit is launched from
CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
API_BASE_URL = os.getenv("LFB_API_URL", "http://127.0.0.1:8000").rstrip("/")

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
st.caption("Historical scenario simulation • Geography, holidays and workload stay fixed to the selected template")
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

# Trigger inference through the FastAPI backend. The backend owns all model
# artifacts and feature transformations; Streamlit only sends UI inputs.
if df_scenarios is not None and st.button("Run Real-Time ML Prediction 🚀", use_container_width=True):
    with st.spinner("Executing structural feature transformations & risk calculation..."):
        payload = {
            "template": json.loads(X_live_template.drop(columns=["Selector_Label"], errors="ignore").to_json(orient="records"))[0],
            "month": int(month),
            "weekday": int(weekday_val),
            "hour": int(hour),
            "incident_group": incident_group,
            "special_service_type": special_service_type,
            "property_category": property_category,
            "property_type": property_type,
        }

        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            st.success(f"Inference successfully calculated in {result['inference_ms']:.2f} ms")
            st.markdown(f"""
                <div style="background-color: #f8fafc; padding: 24px; border-radius: 12px; border-left: 6px solid #ef4444; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                    <span style="color: #64748b; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Attendance Time</span>
                    <h1 style="color: #0f172a; margin: 8px 0 4px 0; font-size: 3.2rem; font-weight: 800;">{result['minutes']} Min. {result['remaining_seconds']} Sek.</h1>
                </div>
            """, unsafe_allow_html=True)
        except requests.ConnectionError:
            st.error(f"❌ Cannot reach the prediction API at `{API_BASE_URL}`. Start FastAPI first.")
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except ValueError:
                detail = exc.response.text
            st.error(f"❌ Prediction API error: {detail}")
        except (requests.RequestException, KeyError, ValueError) as exc:
            st.error(f"❌ Prediction request failed: {exc}")
