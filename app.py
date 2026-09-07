import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="EV Grid Risk Forecaster",
    page_icon="⚡",
    layout="centered"
)

# Custom High-End Modern CSS, Animations & Footer Styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Header Container */
    .hero-container {
        background: linear-gradient(135deg, rgba(16, 24, 40, 0.95), rgba(11, 15, 25, 0.98));
        border: 1px solid rgba(0, 230, 255, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 230, 255, 0.08);
        text-align: center;
    }
    
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 16px;
    }

    /* Animated EV Charging Track */
    .ev-track {
        position: relative;
        width: 100%;
        height: 65px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        align-items: center;
    }
    
    .energy-wire {
        position: absolute;
        top: 50%;
        left: 0;
        width: 100%;
        height: 2px;
        background: repeating-linear-gradient(90deg, #00f2fe, #00f2fe 10px, transparent 10px, transparent 20px);
        animation: energyPulse 1.2s linear infinite;
    }

    @keyframes energyPulse {
        0% { background-position: 0 0; }
        100% { background-position: 40px 0; }
    }

    .ev-car-icon {
        position: absolute;
        font-size: 28px;
        animation: carDrive 6s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    }

    @keyframes carDrive {
        0% { left: -40px; transform: scaleX(1); }
        48% { left: calc(100% - 40px); transform: scaleX(1); }
        50% { transform: scaleX(-1); }
        98% { left: -40px; transform: scaleX(-1); }
        100% { transform: scaleX(1); }
    }

    /* Metric & Prediction Card Styles */
    .result-card {
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .card-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 78, 59, 0.3));
        border: 1px solid #10b981;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    }
    
    .card-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(120, 53, 15, 0.3));
        border: 1px solid #f59e0b;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.2);
    }
    
    .card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(127, 29, 29, 0.35));
        border: 1px solid #ef4444;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
        animation: pulseHigh 1.8s infinite alternate;
    }
    
    @keyframes pulseHigh {
        0% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.2); }
        100% { box-shadow: 0 0 32px rgba(239, 68, 68, 0.5); }
    }

    /* Custom Signature Footer */
    .custom-footer {
        margin-top: 50px;
        margin-bottom: 20px;
        padding: 16px 20px;
        text-align: center;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        font-size: 13.5px;
        font-weight: 500;
        color: #94a3b8;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .custom-footer b {
        color: #f1f5f9;
        font-weight: 600;
    }

    .heart-pulse {
        display: inline-block;
        animation: heartBeat 1.4s infinite;
    }

    @keyframes heartBeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.25); }
        28% { transform: scale(1); }
        42% { transform: scale(1.25); }
        70% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Load Model Pipeline
@st.cache_resource
def load_bundle():
    return joblib.load('ev_grid_model.pkl')

bundle = load_bundle()
pipeline = bundle['pipeline']
classes = bundle['classes']

# Hero Header with CSS-Animated EV Track
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ Smart Grid EV Peak Forecaster</div>
    <div class="hero-subtitle">Real-Time Municipal Load Telemetry & Predictive AI Risk Monitor</div>
    <div class="ev-track">
        <div class="energy-wire"></div>
        <div class="ev-car-icon">⚡🚗💨</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Form Section
with st.form("input_form"):
    st.markdown("### 📍 Charging Corridor Topology")
    col1, col2 = st.columns(2)
    with col1:
        city_zone = st.selectbox("Municipal Zone", ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"])
    with col2:
        station_type = st.selectbox("Charger Tier", ["Normal", "Fast", "Supercharger"])

    st.markdown("### 📊 Live Telemetry & Feeders")
    c1, c2 = st.columns(2)
    with c1:
        vehicles_charged = st.slider("Active Fleet (Vehicles)", min_value=1, max_value=30, value=12)
        duration = st.slider("Session Duration (mins)", 15, 120, 45)
        energy_dispensed = st.number_input("Dispensed Energy (kWh)", 10.0, 500.0, 180.0, step=10.0)
    with c2:
        grid_load = st.slider("Feeder Base Load (MW)", 50, 500, 260)
        renewable = st.slider("Renewable Buffer (%)", 0, 100, 35)

    st.markdown("### 🕒 Temporal Parameters")
    hour = st.slider("Hour (0-23)", 0, 23, 18)
    day = st.selectbox(
        "Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x]
    )

    submit = st.form_submit_button("⚡ Run Grid Risk Assessment")

# Dynamic Output Execution
if submit:
    # 1. Cyclical trigonometric projections
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    day_sin = np.sin(2 * np.pi * day / 7.0)
    day_cos = np.cos(2 * np.pi * day / 7.0)

    # 2. DataFrame generation
    input_data = pd.DataFrame([{
        'city_zone': city_zone,
        'station_type': station_type,
        'vehicles_charged': vehicles_charged,
        'avg_charging_duration_minutes': duration,
        'energy_dispensed_kwh': energy_dispensed,
        'grid_load_mw': grid_load,
        'renewable_energy_used_percent': renewable,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos
    }])

    # 3. Model Inference
    prediction_idx = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    pred_label = classes[prediction_idx]

    # Map output styling cards
    card_class = "card-low" if pred_label == "Low" else ("card-medium" if pred_label == "Medium" else "card-high")
    status_icon = "🟢" if pred_label == "Low" else ("🟡" if pred_label == "Medium" else "🔴")
    recommendation = {
        "Low": "Feeder line capacity is optimal. No active throttling required.",
        "Medium": "Feeder stress is elevated. Prepare sub-station auxiliary storage dispatch.",
        "High": "CRITICAL RISK: Potential thermal overload detected! Initiate dynamic load shedding."
    }[pred_label]

    # Render Modern Animated Card
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div style="font-size: 14px; letter-spacing: 1.5px; text-transform: uppercase; color: #cbd5e1; margin-bottom: 4px;">Predicted Status</div>
        <div style="font-size: 32px; font-weight: 800; margin-bottom: 10px;">{status_icon} {pred_label.upper()} RISK</div>
        <div style="font-size: 14px; opacity: 0.9; max-width: 480px; margin: 0 auto;">{recommendation}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Distribution
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    prob_dict = dict(zip(classes, probabilities))
    m1.metric("Low Risk Prob", f"{prob_dict.get('Low', 0)*100:.1f}%")
    m2.metric("Medium Risk Prob", f"{prob_dict.get('Medium', 0)*100:.1f}%")
    m3.metric("High Risk Prob", f"{prob_dict.get('High', 0)*100:.1f}%")

# Signature Footer
st.markdown("""
<div class="custom-footer">
    Made with <span class="heart-pulse">❤️</span> by <b>Kiran Kumar😁</b> &nbsp;|&nbsp; save electricity ⚡🔋
</div>
""", unsafe_allow_html=True)
