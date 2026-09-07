import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="EV Grid Risk Forecaster", layout="centered", page_icon="⚡")

@st.cache_resource
def load_bundle():
    return joblib.load('ev_grid_model.pkl')

bundle = load_bundle()
pipeline = bundle['pipeline']
classes = bundle['classes']

st.title("⚡ Smart Grid EV Peak Risk Forecaster")
st.markdown("Assess regional power grid overload vulnerability based on EV telemetry and load conditions.")

with st.form("input_form"):
    st.subheader("Station Configuration")
    col1, col2 = st.columns(2)
    with col1:
        city_zone = st.selectbox("City Zone", ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"])
    with col2:
        station_type = st.selectbox("Station Type", ["Normal", "Fast", "Supercharger"])

    st.subheader("Telemetry & Grid State")
    c1, c2 = st.columns(2)
    with c1:
        vehicles_charged = st.slider("Vehicles Charging", min_value=1, max_value=30, value=12)
        duration = st.slider("Avg Charging Duration (mins)", 15, 120, 45)
        energy_dispensed = st.number_input("Energy Dispensed (kWh)", 10.0, 500.0, 180.0, step=10.0)
    with c2:
        grid_load = st.slider("Current Regional Base Load (MW)", 50, 500, 260)
        renewable = st.slider("Renewable Energy Used (%)", 0, 100, 35)

    st.subheader("Temporal Schedule")
    hour = st.slider("Hour of Day (24h format)", 0, 23, 18)
    day = st.selectbox(
        "Day of Week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x]
    )

    submit = st.form_submit_button("Assess Overload Risk")

if submit:
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    day_sin = np.sin(2 * np.pi * day / 7.0)
    day_cos = np.cos(2 * np.pi * day / 7.0)

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

    prediction_idx = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    pred_label = classes[prediction_idx]

    st.markdown("---")
    st.subheader(f"Risk Assessment: **{pred_label.upper()}**")

    prob_df = pd.DataFrame({
        'Risk Tier': classes,
        'Confidence Score': [f"{p * 100:.1f}%" for p in probabilities]
    })
    st.table(prob_df)

    if pred_label == "High":
        st.error("🚨 CRITICAL WARNING: Immediate risk of feeder line overload. Dispatch dynamic demand-response throttling.")
    elif pred_label == "Medium":
        st.warning("⚠️ ATTENTION: Operating near safety threshold. Prepare auxiliary battery storage dispatch.")
    else:
        st.success("✅ STABLE: Grid is operating with sufficient headroom.")
