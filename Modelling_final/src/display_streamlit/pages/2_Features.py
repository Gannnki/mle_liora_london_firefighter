import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Feature Dictionary", 
    page_icon="🛠️", 
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

# 2. Header
st.title("🛠️ Interactive Feature Dictionary")
st.caption("Page 3 • Comprehensive Audit of Engineered Inputs & Mathematical Transformations")
st.write("")

st.markdown("""
Our preprocessing pipeline extracts and transforms complex spatial, temporal, and risk metadata. 
Select any feature category and variable below to explore its data definition and operational rationale:
""")

st.write("")

# 3. Complete Excel Feature Set categorized into groups (Part 1)
feature_groups = {
    "🕒 Temporal Features (Time & Cycles)": {
        "CalYear": "The calendar year of the incident.",
        "Month": "The calendar month (1 to 12) of the incident.",
        "Weekday": "The day of the week (0=Monday to 6=Sunday) to capture weekly traffic cycles.",
        "Hour": "The specific hour of the day (0 to 23) when the emergency was mobilised, extracted out of DatetimeMobilised.",
        "Is_Nightshift": "Binary flag (1/0) for incidents occurring between 23:00 and 06:00, accounting for the 'Night Shift Effect' where turnout times are systematically higher due to human factors.",
        "Is_Rush_Hour": "Binary flag (1/0) for peak traffic periods (e.g., 07:00–09:00 and 16:00–19:00), capturing the influence of urban congestion on travel times.",
        "Is_Weekend": "Binary flag (1/0) indicating if the incident occurred on Saturday or Sunday.",
        "Is_Public_Holiday": "Binary flag (1/0) marking official UK public holidays with altered traffic patterns."
    },
    "📍 Spatial & Geographic Features": {
        "Is_central_London": "Binary flag (1/0) indicating if the location falls within the 14 core inner London boroughs.",
        "DeployedFromStation_Name": "The unique text name of the primary fire station responding to the incident.",
        "IncGeo_BoroughName": "The text name of the London borough where the incident occurred.",
        "Latitude": "The global north-south GPS coordinate of the incident location.",
        "Longitude": "The global east-west GPS coordinate of the incident location.",
        "distance_to_city_center_km": "Calculates the exact great-circle distance (in kilometers) from the incident location to London's geographical center (Charing Cross) using the Haversine formula.",
        "concurrent_same_borough": "This feature calculates the number of other active incidents occurring within the same borough at the exact same time.",
        "borough_intersection_count": "The total number of road intersections within the respective London borough.",
        "borough_intersection_density": "The number of intersections per square kilometer within the borough.",
        "incident_intersection_count_500m": "The number of road intersections within a 500-meter radius of the incident location.",
        "incident_intersection_density_500m": "Intersection density per square kilometer within a 500-meter radius.",
        "incident_intersection_count_1000m": "The number of road intersections within a 1,000-meter (1 km) radius of the incident location.",
        "incident_intersection_density_1000m": "Intersection density per square kilometer within a 1-kilometer radius.",
    },
    "📐 Distance Transformations & Routing Proxies": {
        "distance_fire_to_station": "The actual road network distance traveled by the fire engine from the station to the incident scene.",
        "distance_sqrt": "The square root of the driven distance, calculated using the mathematical formula: np.sqrt(distance_fire_to_station).",
        "distance_squared": "The squared distance, calculated using the mathematical formula: distance_fire_to_station ** 2.",
        "distance_bin": "A categorical feature binning the distance into 6 fixed classes (0-500m, 500m-1km, 1-2km, 2-3km, 3-5km, 5km+).",
        "area_distance_bin": "This categorical feature clusters dispatches into segmentations based on whether the incident occurs in central or outer London alongside specific distance thresholds (center: 0–500m, 500m–1km, 1–2km; outer: 3–4km, 4–6km, 6km+), allowing the model to capture non-linear traffic and routing speeds unique to each urban zone.",
        "detour_ratio": "Calculates the ratio of driven road distance to straight-line distance, serving as a structural routing proxy that flags urban circuitousness and topological barriers like the River Thames."
    },
    "🏢 Property & Incident Structural Metadata": {
        "IncidentGroup": "The broad category of the incident (e.g., Fire, False Alarm, Special Service).",
        "Is_SpecialService": "Binary flag (1/0) indicating a technical rescue or non-fire emergency dispatch.",
        "SpecialServiceType": "Specific sub-category details for technical rescues (e.g., flooding, heavy extrication).",
        "PropertyCategory": "The broad classification of the incident site (e.g., Dwelling, Outdoor, Road Vehicle).",
        "PropertyType": "The highly specific structural type of the scene (e.g., high-rise flat, warehouse, public park).",
        "property_access_complexity": "Converts property categories into an ordinal score to flag structural access barriers like high-rises."
    },
    "📊 Call Volume Transformations": {
        "Is_RepeatedCall": "Binary flag (1/0) indicating if multiple duplicate calls were received for this same running incident.",
        "NumOfCalls_bucket": "Categorical feature classifying the frequency of incoming calls into specific operational buckets.",
        "NumOfCalls_ord": "The transformation of textual categories into true, continuous numbers (decimals).",
        "NumOfCalls_log": "The logarithmic version of NumOfCalls_ord, calculated using the mathematical formula np.log1p(NumOfCalls_ord)—meaning ln(number of calls + 1)."
    },
    "⚡ Risk Features (Binary Flags)": {
        "risk_property_outdoor": "Binary flag (1/0) indicating an open-air incident (often complicates precise geolocation and access).",
        "risk_property_road_vehicle": "Binary flag (1/0) for a vehicle fire in live traffic (high risk of traffic jams and routing delays).",
        "risk_property_outdoor_structure": "Binary flag (1/0) for open-air structures like bridges or masts (often requiring specialized equipment).",
        "risk_many_calls": "Binary flag (1/0) triggering at >=3 emergency calls (proxy for a larger, highly visible incident).",
        "risk_very_many_calls": "Binary flag (1/0) triggering at >=12 calls (proxy for a massive major event causing local chaos).",
        "risk_special_service": "Binary flag (1/0) for non-fire technical rescues (e.g., hazmat, which require distant specialized vehicles).",
        "risk_fire": "Binary flag (1/0) for actual fire incidents (high urgency, often requiring local street closures).",
        "risk_noncentral": "Binary flag (1/0) for outer London boroughs (historically associated with longer transit distances).",
        "risk_repeated_call": "Binary flag (1/0) indicating a follow-up or secondary notification for an already active incident.",
        "risk_weekday_4": "Binary flag (1/0) for Friday dispatches (captures elevated traffic from early weekend commuters).",
        "risk_weekday_2": "Binary flag (1/0) for Wednesday dispatches (statistically tied to higher historic transit friction).",
        "risk_month_3_5_6": "Binary flag (1/0) for March, May, and June (spring/early summer spike in outdoor incidents).",
        "risk_not_nightshift": "Binary flag (1/0) for daytime dispatches (when rush-hour and delivery traffic blocks roads).",
        "risk_not_weekend": "Binary flag (1/0) for weekday dispatches (characterized by heavy commuter and commercial traffic).",
        "high_residual_risk_score": "The sum of all 14 binary risk flags (ranging from 0 to 14), providing an aggregated score of historically severe operational and temporal contexts."
    },
    "🔗 Interaction Terms": {
        "many_calls_x_outdoor": "High call volume and outdoor incident (e.g., a large, highly visible forest or wildfire).",
        "many_calls_x_road_vehicle": "High call volume and a burning vehicle on a road (indicates severe traffic accidents with high visibility and heavy gridlock potential).",
        "many_calls_x_special": "High call volume and a special service incident (e.g., severe structural collapses or hazardous materials accidents).",
        "many_calls_x_noncentral": "High call volume and an incident in the outer boroughs.",
        "road_vehicle_x_noncentral": "Vehicle fire and an outer borough incident (longer response paths on expressways outside the city core).",
        "outdoor_x_noncentral": "Outdoor incident and an outer borough location.",
        "repeated_x_many_calls": "Repeated emergency call and high total call volume (an indicator of escalating large-scale events).",
        "fire_x_many_calls": "Combines fire incidents with high call volume."
    },
    "📈 Rolling Time-Series Features": {
        
        "station_rolling_mean_last_3": "Average attendance time of the station's last three responses.",
        "borough_rolling_mean_last_3": "Average attendance time of the last three responses within the borough.",
        "borough_rolling_max_last_3": "Peak attendance time among the last three responses within the borough.",
        "station_inertia_rolling_mean_last_3": "Tracks the average deviation of the station's last three responses from its historical mean.",
        "borough_inertia_rolling_mean_last_3": "Tracks the average deviation of the borough's last three responses from its historical mean.",
        "borough_inertia_rolling_max_last_3": "The largest delay (peak deviation) among the borough's last three responses compared to its historical mean.",
        "expected_time_by_station_inertia": "Estimated response time for the station, based on how much its recent dispatches were delayed.",
        "expected_time_by_borough_inertia": "Estimated response time for the borough, based on how much recent local dispatches were delayed."
    }
}

# 4. Filter Layout
col_group, col_feat = st.columns(2)

with col_group:
    selected_group = st.selectbox("1. Select Feature Category:", options=list(feature_groups.keys()))

with col_feat:
    selected_feature = st.selectbox("2. Select Variable to Inspect:", options=list(feature_groups[selected_group].keys()))

st.write("")

# 5. Render the Dynamic Card
feature_description = feature_groups[selected_group][selected_feature]

st.markdown(f"""
    <div class="dashboard-card">
        <div class="card-title">📋 Variable: <span style="color: #e63946;">{selected_feature}</span></div>
        <p style="color: #0f172a; font-size: 1.05rem; line-height: 1.6; margin: 0;">
            </b>{feature_description}
        </p>
    </div>
""", unsafe_allow_html=True)