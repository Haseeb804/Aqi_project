"""
FastAPI prediction endpoint.

Endpoints:
  GET  /health         – health check
  POST /predict        – AQI forecast for {city, date}
  GET  /current/{city} – current AQI + forecast

Run:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    AQI_CATEGORIES,
    DEFAULT_CITY,
    FEATURE_COLS,
    HOPSWORKS_HOST,
    HOPSWORKS_KEY,
    HOPSWORKS_PROJECT,
    MODEL_LOOKBACK,
    MODELS_DIR,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from utils.store_client import FeatureStore

app = FastAPI(
    title="AQI Forecast API",
    description="3-day AQI forecasting API for Pakistani cities",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: load model and feature store ─────────────────────────────────────
_model = None
_model_info: dict = {}
_store: FeatureStore | None = None


@app.on_event("startup")
def startup() -> None:
    global _model, _model_info, _store

    _store = FeatureStore(
        hopsworks_key=HOPSWORKS_KEY,
        hopsworks_project=HOPSWORKS_PROJECT,
        hopsworks_host=HOPSWORKS_HOST,
        fg_name=FEATURE_GROUP_NAME,
        fg_version=FEATURE_GROUP_VERSION,
    )

    info_path = MODELS_DIR / "best_model_info.pkl"
    if info_path.exists():
        _model_info = joblib.load(info_path)
        model_type  = _model_info.get("model_type", "sklearn")
        model_path  = _model_info.get("model_path", "")

        if Path(model_path).exists():
            if model_type == "LSTM":
                import tensorflow as tf
                _model = tf.keras.models.load_model(model_path)
            else:
                _model = joblib.load(model_path)
            print(f"[API] Loaded model: {_model_info.get('model_name')} from {model_path}")
        else:
            print(f"[API] WARNING: model path {model_path} not found")
    else:
        print("[API] WARNING: no model found – /predict will return 503")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    city: str = Field(default=DEFAULT_CITY, description="City name")
    date: Optional[str] = Field(
        default=None,
        description="Reference date YYYY-MM-DD (defaults to today)",
        example="2024-06-15",
    )


class DayForecast(BaseModel):
    day: int
    date: str
    predicted_aqi: float
    category: str
    color: str
    lower_bound: float
    upper_bound: float


class PredictResponse(BaseModel):
    city: str
    reference_date: str
    current_aqi: Optional[float]
    current_category: Optional[str]
    forecast: list[DayForecast]
    model_name: str
    model_rmse: float
    generated_at: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _aqi_category(aqi: float) -> tuple[str, str]:
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label, color
    return "Hazardous", "#7e0023"


def _run_inference(features: pd.DataFrame) -> tuple[np.ndarray, float]:
    feat_cols  = _model_info.get("feature_cols", FEATURE_COLS)
    lookback   = _model_info.get("lookback", MODEL_LOOKBACK)
    model_type = _model_info.get("model_type", "sklearn")
    pred_std   = _model_info.get("pred_std_mean", 10.0)

    available = [c for c in feat_cols if c in features.columns]
    if not available:
        raise HTTPException(status_code=422, detail="Feature columns missing in stored data")

    X = features[available].values.astype(np.float32)

    if model_type == "LSTM":
        if len(X) < lookback:
            X = np.pad(X, ((lookback - len(X), 0), (0, 0)), mode="edge")
        seq   = X[-lookback:][np.newaxis, :, :]
        preds = _model.predict(seq, verbose=0)[0]
    else:
        row   = X[-1:, :]
        preds = _model.predict(row)[0]

    return np.asarray(preds, dtype=float), float(pred_std)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_name": _model_info.get("model_name"),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Forecast"])
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training_pipeline.py first.",
        )
    if _store is None:
        raise HTTPException(status_code=503, detail="Feature store not initialised")

    # Parse reference date
    if req.date:
        try:
            ref_date = datetime.strptime(req.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        ref_date = datetime.utcnow()

    # Load features
    features = _store.read_latest(city=req.city, n_hours=48)
    if features.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No feature data for city '{req.city}'. Run the feature or backfill pipeline.",
        )

    current_aqi = float(features["aqi"].iloc[-1]) if "aqi" in features.columns else None
    curr_label, _ = _aqi_category(current_aqi) if current_aqi else ("Unknown", "#888")

    # Inference
    preds, pred_std = _run_inference(features)

    forecast_days: list[DayForecast] = []
    for i, pred_val in enumerate(preds, start=1):
        day_date = (ref_date + timedelta(days=i)).strftime("%Y-%m-%d")
        label, color = _aqi_category(float(pred_val))
        forecast_days.append(DayForecast(
            day=i,
            date=day_date,
            predicted_aqi=round(float(pred_val), 1),
            category=label,
            color=color,
            lower_bound=round(max(0.0, float(pred_val) - 1.5 * pred_std), 1),
            upper_bound=round(float(pred_val) + 1.5 * pred_std, 1),
        ))

    return PredictResponse(
        city=req.city,
        reference_date=ref_date.strftime("%Y-%m-%d"),
        current_aqi=round(current_aqi, 1) if current_aqi else None,
        current_category=curr_label,
        forecast=forecast_days,
        model_name=_model_info.get("model_name", "unknown"),
        model_rmse=round(_model_info.get("metrics", {}).get("rmse", 0.0), 4),
        generated_at=datetime.utcnow().isoformat(),
    )


@app.get("/current/{city}", tags=["Forecast"])
def current_aqi(city: str) -> dict:
    """Quick endpoint: current AQI + 3-day forecast for a city."""
    return predict(PredictRequest(city=city)).dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
