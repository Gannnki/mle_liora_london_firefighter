import streamlit as st
import pandas as pd
import numpy as np

# 1. Standard Page Configuration
st.set_page_config(
    page_title="LFB Analytics", 
    page_icon="🚒", 
    layout="wide"
)

st.sidebar.markdown(
    """
    <style>
        /* Sucht nach dem ersten Eintrag im automatischen Menü und benennt ihn um */
        [data-testid="stSidebarNavItems"] li:first-child span {
            visibility: hidden;
            position: relative;
        }
        [data-testid="stSidebarNavItems"] li:first-child span::after {
            content: "Overview";
            visibility: visible;
            position: absolute;
            left: 0;
            top: 0;
            white-space: nowrap;
        }
    </style>
""",
    unsafe_allow_html=True,
)



# Inject custom modern CSS 
st.markdown("""
    <style>
    /* Styling for the KPI and content cards */
    .dashboard-card {
        background-color: #f8fafc;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    /* Style titles inside custom cards */
    .card-title {
        color: #0f172a;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Header Section
st.title("🚒 London Fire Brigade Performance")

# Leave generous whitespace instead of hard lines
st.write("")
st.write("")

# 3. Project Mission Card
st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">🎯 Project Mission</div>
        <p style="color: #475569; margin: 0; line-height: 1.6;">
            This platform hosts an optimized Machine Learning system designed to predict emergency response times 
            for the London Fire Brigade. By uncovering patterns within structural traffic routing data, dispatch times, 
            and geographic bottlenecks, we enable data-driven resource dispatch planning.
        </p>
    </div>
""", unsafe_allow_html=True)

st.write("")

# 4. KPI Metrics Section (Leveraging native elements over the new clean background)
st.subheader("Key Performance Indicators")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total Logged Incidents", value="1.281.992")

with col2:
    st.metric(label="Avg. Response Time (first Pump)", value="5m 32s")

with col3:
    st.metric(label="Covered Years", value="2009 - 2026")

st.write("")
st.write("")

# 5. Map Section (Loading 5,000 real LFB incident locations)
st.subheader("Incident Geographic Hotspots")

# Wrap the map inside dashboard-card container
#st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🗺️ London Fleet Dispatch Density</div>', unsafe_allow_html=True)
st.write("Displaying a sample of 5,000 dispatches across Greater London:")

try:
    # Load the real spatial sample telemetry file
    map_df = pd.read_csv("data_streamlit/lfb_map_sample.csv")
    st.map(map_df, zoom=10)
except Exception as e:
    # Display error and render the original dummy map as a safe fallback
    st.error(f"Error loading real map telemetry: {e}")
    london_lat, london_lon = 51.5074, -0.1278
    fallback_df = pd.DataFrame({
        "lat": np.random.normal(london_lat, 0.04, size=1000),
        "lon": np.random.normal(london_lon, 0.06, size=1000)
    })
    st.map(fallback_df, zoom=10)

st.markdown('</div>', unsafe_allow_html=True)

# 6. Two-Column Insights Layout using custom CSS cards
#st.subheader("Exploratory Insights")
col4, col5 = st.columns(2)

with col4:
    st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">🗂️ Data Architecture</div>
            <p style="color: #475569; margin: 0; line-height: 1.6;">
            The project utilizes records from the UK government spanning from <b>2009 to 2026</b>, which are officially updated on a quarterly basis. 
            Structurally, two data files exist:  <i>Incident Data</i> and <i>Mobilisation Data</i>. 
            It's a  <b>one-to-many relationship</b>, as for a single emergency incident multiple fire appliances and stations can be mobilized.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">🚨 Why This Project Matters</div>
            <p style="color: #475569; margin: 0; line-height: 1.6;">
            In emergency response, every single second saved can mean the difference between saving or losing a life. 
            By accurately predicting response times, the London Fire Brigade can optimize its dispatch strategies, 
            identify structural bottlenecks in city traffic, and improve risk-based resource planning. 
            This data-driven approach directly enhances public safety across London's complex urban landscape.
            </p>
        </div>
""", unsafe_allow_html=True)