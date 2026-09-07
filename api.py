from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib

# Initialize application
app = FastAPI(
    title="Smart Grid EV Overload Inference API",
    version="1.0.0",
    description="Enterprise predictive load-shedding and risk scoring service for EV charging infrastructure."
)

# Load serialized model pipeline
try:
    bundle = joblib.load("ev_grid_model.pkl")
    pipeline = bundle["pipeline"]
    classes = bundle["classes"]
except Exception as e:
    raise RuntimeError(f"Failed to load ev_grid_model.pkl: {str(e)}")

# Request schema with validation
class TelemetryPayload(BaseModel):
    city_zone: str = Field(..., example="Zone A", description="Municipal grid deployment sector")
    station_type: str = Field(..., example="Supercharger", description="Hardware tier: Normal, Fast, or Supercharger")
    vehicles_charged: int = Field(..., ge=1, le=100, example=14, description="Simultaneously active charging sessions")
    avg_charging_duration_minutes: float = Field(..., ge=1.0, example=45.0, description="Average active dwell time in minutes")
    energy_dispensed_kwh: float = Field(..., ge=0.0, example=210.5, description="Cumulative dispensed kWh on current cycle")
    grid_load_mw: float = Field(..., ge=0.0, example=280.0, description="Base feeder loading in megawatts")
    renewable_energy_used_percent: float = Field(..., ge=0.0, le=100.0, example=25.0, description="Renewable energy offset buffer percentage")
    hour: int = Field(..., ge=0, le=23, example=19, description="Current hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, example=0, description="Day of week (0=Monday, 6=Sunday)")

# Response schema
class RiskAssessmentResponse(BaseModel):
    predicted_risk_tier: str
    confidence_score: float
    probabilities: dict
    ocpp_recommended_action: str
    feeder_status: str

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint for Kubernetes / load-balancer probes."""
    return {"status": "healthy", "service": "ev-grid-risk-engine"}

@app.post("/predict", response_model=RiskAssessmentResponse, tags=["Inference"])
def assess_overload_risk(payload: TelemetryPayload):
    try:
        # 1. Cyclical trigonometric projections
        hour_sin = np.sin(2 * np.pi * payload.hour / 24.0)
        hour_cos = np.cos(2 * np.pi * payload.hour / 24.0)
        day_sin = np.sin(2 * np.pi * payload.day_of_week / 7.0)
        day_cos = np.cos(2 * np.pi * payload.day_of_week / 7.0)

        # 2. DataFrame generation for Scikit-Learn pipeline
        input_data = pd.DataFrame([{
            'city_zone': payload.city_zone,
            'station_type': payload.station_type,
            'vehicles_charged': payload.vehicles_charged,
            'avg_charging_duration_minutes': payload.avg_charging_duration_minutes,
            'energy_dispensed_kwh': payload.energy_dispensed_kwh,
            'grid_load_mw': payload.grid_load_mw,
            'renewable_energy_used_percent': payload.renewable_energy_used_percent,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_sin': day_sin,
            'day_cos': day_cos
        }])

        # 3. Model Inference
        prediction_idx = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        pred_label = classes[prediction_idx]

        # 4. Action Mapping (OCPP 1.6J/2.0.1 smart charging profiles)
        action_map = {
            "Low": "MAINTAIN_MAX_THROUGHPUT: Feeder headroom optimal. SetChargingProfile max limit 100%.",
            "Medium": "THROTTLE_SECONDARY_FEEDERS: Feeder strain elevated. SetChargingProfile to 60% rated current.",
            "High": "CRITICAL_LOAD_SHED: Overload imminent. SetChargingProfile to 25% or switch to onsite BESS storage."
        }
        
        prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}

        return RiskAssessmentResponse(
            predicted_risk_tier=pred_label,
            confidence_score=round(float(np.max(probabilities)), 4),
            probabilities=prob_dict,
            ocpp_recommended_action=action_map.get(pred_label, "MONITOR"),
            feeder_status="OPTIMAL" if pred_label == "Low" else ("WARNING" if pred_label == "Medium" else "DANGER")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")
