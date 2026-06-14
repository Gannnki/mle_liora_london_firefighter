import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Conclusion & Limitations", 
    page_icon="🏆", 
    layout="wide"
)

# Custom SaaS-style CSS 
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
        margin-bottom: 14px;
    }
    .metric-highlight {
        color: #e63946;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Header Section
st.title("🏆 Conclusion & Future Roadmap")
st.caption("Page 6 • Limitations & Final Project Conclusion")
st.write("")


# --- SECTION 1: CORE ACHIEVEMENTS ---
st.subheader("Core Achievements")


st.markdown("""
    <style>
    .timeline-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        border-top: 5px solid #1f77b4;
        margin-bottom: 20px;
    }
    .timeline-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #0f172a;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

# --- step 1 ---
with col1:
    st.markdown("""
        <div class="timeline-card" style="border-top-color: #0284c7;">
            <div class="timeline-header">🛠️ 1. Data Engineering</div>
            <p style="color: #334155; line-height: 1.6; font-size: 0.95rem; margin-bottom: 0;">
                This project successfully transformed inconsistent, raw LFB records into a clean, feature-rich matrix.
                A prime example of this is the integrating of real-world road-distance metrics to ensure high routing intelligence.
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- step 2 ---
with col2:
    st.markdown("""
        <div class="timeline-card" style="border-top-color: #22c55e;">
            <div class="timeline-header">📈 2. Model Performance</div>
            <p style="color: #334155; line-height: 1.6; font-size: 0.95rem;">
                Our final XGBoost model emerged as the definitive top performer, achieving a <b>Test MAE of 49.95 seconds</b> and explaining <b>55.73% of the total variance</b> (R² = 0.5773) in response times.
            </p>
            <p style="color: #475569; line-height: 1.6; font-size: 0.9rem; margin-top: 8px; font-style: italic; margin-bottom: 0;">
                On average, arrivals are forecasted with an error under 50s. The P90 metric proves that 90% of predictions deviate by less than two minutes.
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- step 3 ---
with col3:
    st.markdown("""
        <div class="timeline-card" style="border-top-color: #6366f1;">
            <div class="timeline-header">🧠 3. Interpretability</div>
            <p style="color: #334155; line-height: 1.6; font-size: 0.95rem; margin-bottom: 0;">
                Through SHAP interpretability analysis, we verified that the model’s internal logic mirrors realistic physics and urban geography. 
                This is driven primarily by spatial travel distance (controlling over 28% of total predictive weight) and heavily refined by non-linear factors like inner-city traffic and cyclical temporal patterns.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.write("")
# --- SECTION 1: LIMITATIONS ---
st.subheader("Limitations")

# 2 rows
col_img, col_takeaway = st.columns([1.5, 1])  

with col_img:
    # load picture
    st.image("data_streamlit/limitations.png", use_container_width=True)

with col_takeaway:
    
    st.markdown("""
        <style>
            [data-testid="column"]:nth-child(2) {
                display: flex;
                flex-direction: column;
                justify-content: center;
                height: 100%;
            }
        </style>
    """, unsafe_allow_html=True)
    
   
    with st.container(border=True):
        st.markdown("#### 🎯 Key Takeaway for Users")
        st.markdown("""
        Offline model excel at predicting structural, geographical trends, but require live data integrations to capture localized, chaotic spikes.
        """)

# --- SECTION 2: CONCLUSION ---
st.subheader("Strategic Value & Impact")

st.markdown("""
        <div class="dashboard-card" style="border-left: 4px solid #e63946; height: 100%; padding: 15px;">
            <div class="card-title" style="color: #e63946; font-weight: bold; margin-bottom: 12px;">💡 How to use it now</div>
            <p style="color: #334155; line-height: 1.6; margin-bottom: 16px;">
                Even with only the historical data, the system successfully points out reasons for historical traffic delays. 
                It offers a practical framework that can be used already now.
            </p>
            <div style="color: #475569; line-height: 1.6; font-size: 0.95rem;">
                <strong style="color: #334155;">Immediate Usage:</strong>
                <ul style="margin-top: 8px; padding-left: 20px;">
                    <li style="margin-bottom: 6px;"><strong>Diagnostic tool:</strong> Serves as a diagnostic tool to find and explain past traffic bottlenecks and delays.</li>
                    <li style="margin-bottom: 6px;"><strong>Risk Mapping:</strong> Enables teams to highlight high-risk zones and specific times where fire engines lose the most critical time.</li>
                    <li style="margin-bottom: 6px;"><strong>Scenario Testing:</strong> Allows the fire brigade to simulate and test offline "what-if" scenarios safely.</li>
                    <li style="margin-bottom: 0;"><strong>Data-Driven Leadership:</strong> Empowers managers to make objective, data-backed decisions for resource management.</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- SECTION 3: FUTURE OUTLOOK & OPERATIONAL DEPLOYMENT ---
st.write("")
st.subheader("Operational Deployment & Future Outlook")

col_fut1, col_fut2 = st.columns(2)

with col_fut1:
    st.markdown("""
        <div class="dashboard-card" style="height: 100%;">
            <div class="card-title">🚀 Real-World Integration</div>
            <p style="color: #334155; line-height: 1.6; font-size: 0.95rem; margin-bottom: 0;">
                In a live deployment scenario, this XGBoost engine would serve as a <b>Microservice API</b> integrated 
                directly into the LFB's Computer-Aided Dispatch (CAD) software. The moment an emergency call is received, 
                the system automatically reads the caller's location and gives traffic-smart recommendations.
                Instead of dispatching the closest station based on flat geography, the control room receives a 
                <b>data-driven recommendation</b> choosing the vehicle that circumvents traffic constraints most efficiently.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col_fut2:
    st.markdown("""
        <div class="dashboard-card" style="border-left: 4px solid #64748b; height: 100%;">
            <div class="card-title">🔮 Future Research & Pipeline Scaling</div>
            <p style="color: #475569; line-height: 1.6; font-size: 0.95rem; margin-bottom: 0;">
                Roadmap to transition our system:<br><br>
                1. <b>Live Routing:</b> Connect the pipeline with a dynamic mapping API (e.g., Google Maps or OpenStreetMap) to continuously update the route using live, real-time data.<br>
                2. <b>Weather & Environmental Context:</b> Inject live weather data (heavy rain, snow, visibility constraints).<br>
                3. <b>Predictive Incident Dispatching:</b> Shift from predicting response times to forcasting incident probabilities.
            </p>
        </div>
    """, unsafe_allow_html=True)
