import datetime
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="AeroShield AQI Predictor API",
    description="API for Air Quality Index Forecasting based on pollutant and meteorological readings.",
    version="1.0.0"
)

# Paths to models
MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"

# Load the model and scaler
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Trained model or scaler artifacts not found. Please train the model first.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class PredictRequest(BaseModel):
    pm25: float = Field(35.0, alias="PM2.5", description="PM2.5 concentration in µg/m³")
    pm10: float = Field(70.0, alias="PM10", description="PM10 concentration in µg/m³")
    no2: float = Field(40.0, alias="NO2", description="NO₂ concentration in µg/m³")
    co: float = Field(0.8, alias="CO", description="CO concentration in mg/m³")
    so2: float = Field(15.0, alias="SO2", description="SO₂ concentration in µg/m³")
    temperature: float = Field(25.0, alias="Temperature", description="Temperature in °C")
    humidity: float = Field(60.0, alias="Humidity", description="Relative humidity percentage")
    timestamp: str = Field(None, description="ISO-8601 timestamp string (e.g. 2026-07-20T12:00:00). Defaults to current time.")

    class Config:
        populate_by_name = True

def get_aqi_details(aqi):
    if aqi <= 50:
        return {"category": "Good", "color": "#10b981", "desc": "Air quality is satisfactory, and air pollution poses little or no risk."}
    elif aqi <= 100:
        return {"category": "Moderate", "color": "#eab308", "desc": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."}
    elif aqi <= 150:
        return {"category": "Unhealthy for Sensitive Groups", "color": "#f97316", "desc": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."}
    elif aqi <= 200:
        return {"category": "Unhealthy", "color": "#ef4444", "desc": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."}
    elif aqi <= 300:
        return {"category": "Very Unhealthy", "color": "#a855f7", "desc": "Health alert: The risk of health effects is increased for everyone."}
    else:
        return {"category": "Hazardous", "color": "#7f1d1d", "desc": "Health warning of emergency conditions: The entire population is more likely to be affected."}

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "AeroShield AQI Predictor API is operational.",
        "endpoints": {
            "predict": "/predict (POST)",
            "docs": "/docs (GET)"
        }
    }

@app.post("/predict")
def predict_aqi(payload: PredictRequest):
    try:
        # Determine temporal variables
        if payload.timestamp:
            dt = datetime.datetime.fromisoformat(payload.timestamp.replace("Z", "+00:00"))
        else:
            dt = datetime.datetime.now()
            
        hour = dt.hour
        month = dt.month
        day_of_week = dt.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Calculate cyclical encodings
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12.0)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12.0)
        
        # Construct features DataFrame matching the exact training columns order:
        # ["PM2.5", "PM10", "NO2", "CO", "SO2", "Temperature", "Humidity", 
        #  "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week", "is_weekend"]
        input_data = pd.DataFrame([{
            "PM2.5": payload.pm25,
            "PM10": payload.pm10,
            "NO2": payload.no2,
            "CO": payload.co,
            "SO2": payload.so2,
            "Temperature": payload.temperature,
            "Humidity": payload.humidity,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend
        }])
        
        # Scaling
        input_scaled = scaler.transform(input_data)
        
        # Prediction
        pred = model.predict(input_scaled)[0]
        aqi_val = float(np.clip(pred, 0, 500))
        
        # Category details
        details = get_aqi_details(aqi_val)
        
        return {
            "predicted_aqi": round(aqi_val, 1),
            "category": details["category"],
            "color_code": details["color"],
            "description": details["desc"],
            "input_parameters": {
                "pollutants": {
                    "PM2.5": payload.pm25,
                    "PM10": payload.pm10,
                    "NO2": payload.no2,
                    "CO": payload.co,
                    "SO2": payload.so2
                },
                "meteorology": {
                    "Temperature": payload.temperature,
                    "Humidity": payload.humidity
                },
                "temporal": {
                    "hour": hour,
                    "month": month,
                    "day_of_week": day_of_week,
                    "is_weekend": bool(is_weekend)
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
