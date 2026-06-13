import streamlit as st
import pandas as pd

# 1. Page Configuration (Note: page_config can be set on subpages too)
st.set_page_config(
    page_title="Data Insights", 
    page_icon="🤖", 
    layout="wide"
)

# Inject custom modern CSS 
st.markdown("""
    <style>
    /* Styling for the white dashboard cards with subtle shadow */
    .dashboard-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
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
st.title("🤖 Data Insights")
st.caption("Page 2 • Data Analytics & Target Distributions")

st.write("")

# 3. ROW 1: Hero Section (Image Left, Text Right)
col1, col2 = st.columns([1.2, 1])

with col1:
    #st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔥 Incident Group Distribution</div>', unsafe_allow_html=True)
    try:
        # Display the chart image
        st.image("data_streamlit/incident_group.png", use_container_width=True)
        
        # Professional caption explaining the chart's findings
        st.markdown("""
        <p style="color: #475569; font-size: 0.9rem; line-height: 1.5; margin-top: 12px;">
        💡 <b>Key Finding:</b> The audit reveals that <b>False Alarms</b> make up the vast majority of all logged dispatches. 
        This is followed by <b>Special Services</b> (such as rescues or hazardous materials), while actual <b>Fires</b> 
        represent the smallest share of the overall operational volume.
        </p>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load incident_group.png: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">🔍 Data Audit & Visualization Phase</div>
            <p style="color: #475569; margin: 0; line-height: 1.6;">
            Before building our predictive system, we performed an extensive <b>Data Audit</b> to validate data integrity 
            and uncover hidden relationships. The charts and matrices below present key insights from our 
            <b>Data Visualization</b> phase.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">📍 The Distance Challenge</div>
            <p style="color: #475569; margin: 0; line-height: 1.6;">
            A major limitation in the raw dataset was the complete absence of a pre-calculated distance feature 
            between the fire station and the incident location. The raw logs only provided geographic 
            coordinates. To fix this, we had to compute the actual routing 
            distances.
            </p>
        </div>
    """, unsafe_allow_html=True)


st.write("")

# 4. ROW 2: Demands & Hourly Distributions (Two Images Side-by-Side)
col3, col4 = st.columns(2)

with col3:
    #st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⏰ Incident Types by Hour of Day</div>', unsafe_allow_html=True)
    try:
        st.image("data_streamlit/incident_types_hourly.png", use_container_width=True)
        
   
        st.markdown("""
        <p style="color: #475569; font-size: 0.9rem; line-height: 1.5; margin-top: 12px;">
        💡 <b>Key Finding:</b> All three incident categories follow a highly synchronized diurnial pattern. 
        Operational activity reaches its absolute <b>low point between 4:00 AM and 5:00 AM</b>. 
        Volume then surges rapidly throughout the morning, exhibiting a strong double-peak behavior with a midday plateau and a final <b>evening peak between 5:00 PM and 7:00 PM</b> before dropping overnight.
        </p>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load incident_types_hourly.png: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


with col4:
    #st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌙 Turnout Time Over Hour (Night Shift Effect)</div>', unsafe_allow_html=True)
    try:
        st.image("data_streamlit/turnout_time_hourly.png", use_container_width=True)
        
        
        st.markdown("""
        <p style="color: #475569; font-size: 0.9rem; line-height: 1.5; margin-top: 12px;">
        💡 <b>Key Finding:</b> The graph uncovers a severe <b>night shift effect</b> on turnout times. 
        While daytime turnout stays consistently low and efficient at around <b>70 to 75 seconds</b>, latency surges significantly during the deep night, peaking between <b>3:00 AM and 5:00 AM at over 110 seconds</b>. 
        This is a vital human-factor variable that our XGBoost model must structurally account for.
        </p>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load turnout_time_hourly.png: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

# 5. ROW 3: London Map 

st.markdown('<div class="card-title">🗺️ Incident Distribution by Location</div>', unsafe_allow_html=True)
try:
    st.image("data_streamlit/london_map.png", use_container_width=True)
        
    st.markdown("""
    <p style="color: #475569; font-size: 0.9rem; line-height: 1.5; margin-top: 12px;">
    💡 <b>Key Finding:</b> The 2D histogram shows a spatial pattern of the incidents: There are more incidents in the central part of London and less incidents 
in the outskirt of London
    </p>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Could not load london_map.png: {e}")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# 5. ROW 3: Histogramms

st.markdown('<div class="card-title">📊 Response Time Histogram</div>', unsafe_allow_html=True)
try:
    st.image("data_streamlit/target.png", use_container_width=True)
        
    st.markdown("""
    <p style="color: #475569; font-size: 0.9rem; line-height: 1.5; margin-top: 12px;">
    💡 <b>Key Finding:</b> The 2 components (travel time and turn out time) of the attendance time has different distributions. The travel time has larger variations while the turn out time has much smaller variations. Overall, the travel time significantly affect the target variable (the attendance time).
    </p>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Could not load target.png: {e}")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

