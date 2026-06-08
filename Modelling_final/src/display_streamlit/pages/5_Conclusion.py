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
st.title("🏆 Project Synthesis & Operational Evaluation")
st.caption("Page 6 • Critical Limitations & Final Project Conclusion")
st.write("")

# --- SECTION 1: LIMITATIONS ---
st.subheader("7. Limitations")

# Unified clean container for the overview
with st.container(border=True):
    st.markdown("### Operational Boundaries & Challenges")
    st.markdown("""
    While our finalized XGBoost model demonstrates good predictive accuracy and strong statistical stability, 
    an honest assessment of its deployment readiness reveals several critical limitations. These constraints 
    are primarily driven by data boundaries and the inherent randomness of urban emergency environments.
    """)

st.write("")

# Displaying each limitation in an native sidebar-info box layout
col_lim1, col_lim2, col_lim3 = st.columns(3)

with col_lim1:
    with st.container(border=True):
        st.markdown("#### 🚗 Real-Time Traffic Gap")
        st.markdown("""
        The model relies entirely on historical, static tabular data (time, coordinates, distances). 
        It lacks integration with live traffic feeds (Google Maps API) or real-world dispatch anomalies (protests, construction). 
        This absence explains the *Outlier Gap* in our residual analysis, underestimating extreme arrival delays.
        """)

with col_lim2:
    with st.container(border=True):
        st.markdown("#### 🧠 Micro-Level Human Behavior")
        st.markdown("""
        Emergency response times are highly dependent on variables that cannot be quantified in a tabular dataset. 
        Turnout routines within the station, individual crew driving profiles, or civilian driver behavior 
        introduce random noise that no offline model can fully capture.
        """)

with col_lim3:
    with st.container(border=True):
        st.markdown("#### 📈 Overestimation of Rapid Arrivals")
        st.markdown("""
        As shown in the advanced diagnostics, the model struggles with very fast dispatches (under 200 seconds), 
        shifting them back toward the operational mean. This regression-to-the-mean effect lowers sensitivity 
        to localized, optimal conditions right next to a station.
        """)

# --- SECTION 2: CONCLUSION ---
st.subheader("8. Conclusion")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("""
        <div class="dashboard-card" style="height: 100%;">
            <div class="card-title">🚒 Executive Summary</div>
            <p style="color: #334155; line-height: 1.65; margin-bottom: 12px;">
                This project successfully developed and evaluated a highly robust machine learning pipeline to predict emergency attendance times for the London Fire Brigade (LFB). By progressing systematically from basic baselines to tuned, state-of-the-art gradient boosting frameworks, we established an architecture that balances statistical rigor with practical operational utility.
            </p>
            <p style="color: #334155; line-height: 1.65; margin-bottom: 12px;">
                Our final XGBoost model emerged as the definitive top performer, achieving a <span class="metric-highlight">Test MAE of 49.95 seconds</span> and explaining <span class="metric-highlight">55.73% of the total variance</span> (R² = 0.5773) in response times. This means that, on average, the model forecasts fire engine arrivals with an error margin of under 50 seconds. Furthermore, the P90 metric of 114.88 seconds proves that 90% of all test predictions deviate by less than two minutes from reality, demonstrating a dependable level of reliability for logistical planning.
            </p>
            <p style="color: #334155; line-height: 1.65; margin-bottom: 0;">
                Through SHAP interpretability analysis, we verified that the model’s internal logic mirrors realistic physics and urban geography, driven primarily by spatial travel distance (controlling over 28% of total predictive weight) and heavily refined by non-linear factors like inner-city traffic penalties, detour routing choices, and cyclical temporal patterns.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="dashboard-card" style="border-left: 4px solid #e63946; height: 100%;">
            <div class="card-title" style="color: #e63946;">💡 Strategic Value & Impact</div>
            <p style="color: #334155; line-height: 1.6; margin-bottom: 16px;">
                In conclusion, despite the challenges of unpredictable traffic anomalies, this model provides a highly valuable, data-driven foundation for predictive resource allocation.
            </p>
            <p style="color: #475569; line-height: 1.6; font-size: 0.95rem; margin-bottom: 0;">
                It offers a practical framework that public safety stakeholders can utilize to optimize emergency infrastructure, mitigate logistical risks, and ultimately accelerate lifesaving emergency dispatches across London.
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- SECTION 3: FUTURE OUTLOOK & OPERATIONAL DEPLOYMENT ---
st.write("")
st.subheader("9. Operational Deployment & Future Outlook")

col_fut1, col_fut2 = st.columns(2)

with col_fut1:
    st.markdown("""
        <div class="dashboard-card" style="height: 100%;">
            <div class="card-title">🚀 Real-World Operational Integration</div>
            <p style="color: #334155; line-height: 1.6; font-size: 0.95rem; margin-bottom: 0;">
                In a live deployment scenario, this XGBoost engine would serve as a <b>Microservice API</b> integrated 
                directly into the LFB's Computer-Aided Dispatch (CAD) software. The moment an emergency call is received, 
                the system would automatically fetch the active engine's route and feed it into the model. 
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
                To transition this system into a true production-grade platform, three next steps are paramount:<br><br>
                1. <b>Live Routing Telemetry:</b> Connect the data engineering pipeline with a dynamic mapping API (e.g., Google Maps or OpenStreetMap) to continuously update with real-time gridlock data.<br>
                2. <b>Weather & Environmental Context:</b> Incorporate live meteorological flags (heavy rain, snow, visibility constraints) to map seasonal transit friction.<br>
                3. <b>Predictive Incident Dispatching:</b> Expand the architecture from predicting response times to forecasting local incident probabilities based on deep historical cycles.
            </p>
        </div>
    """, unsafe_allow_html=True)
