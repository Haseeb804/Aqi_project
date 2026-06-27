"""
Streamlit Dashboard – Real-Time 3-Day AQI Forecasting
Faisalabad, Pakistan

Run:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Forecast – Faisalabad",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

REFRESH_INTERVAL_SECONDS = 1800   # 30 minutes


# ── Helpers ───────────────────────────────────────────────────────────────────

def aqi_category(aqi: float) -> tuple[str, str]:
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label, color
    return "Hazardous", "#7e0023"


@st.cache_resource(show_spinner="Connecting to feature store...")
def get_store() -> FeatureStore:
    return FeatureStore(
        hopsworks_key=HOPSWORKS_KEY,
        hopsworks_project=HOPSWORKS_PROJECT,
        hopsworks_host=HOPSWORKS_HOST,
        fg_name=FEATURE_GROUP_NAME,
        fg_version=FEATURE_GROUP_VERSION,
    )


@st.cache_resource(show_spinner="Loading model...")
def load_model_info() -> dict:
    info_path = MODELS_DIR / "best_model_info.pkl"
    if not info_path.exists():
        return {}
    return joblib.load(info_path)


@st.cache_resource(show_spinner="Loading model weights...")
def load_model(info: dict):
    if not info:
        return None
    model_type = info.get("model_type", "sklearn")
    model_path = info.get("model_path", "")
    if not Path(model_path).exists():
        return None
    if model_type == "LSTM":
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    return joblib.load(model_path)


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS, show_spinner="Fetching latest features...")
def fetch_latest_features(city: str, n_hours: int = 48) -> pd.DataFrame:
    store = get_store()
    return store.read_latest(city=city, n_hours=n_hours)


def run_inference(model, info: dict, features: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Returns (predictions shape (3,), prediction_std).
    """
    feat_cols = info.get("feature_cols", FEATURE_COLS)
    available = [c for c in feat_cols if c in features.columns]
    lookback   = info.get("lookback", MODEL_LOOKBACK)
    model_type = info.get("model_type", "sklearn")
    pred_std   = info.get("pred_std_mean", 10.0)

    if features.empty or len(available) == 0:
        return np.array([float("nan")] * 3), pred_std

    X = features[available].values.astype(np.float32)

    if model_type == "LSTM":
        if len(X) < lookback:
            X = np.pad(X, ((lookback - len(X), 0), (0, 0)), mode="edge")
        seq = X[-lookback:][np.newaxis, :, :]   # (1, lookback, n_features)
        preds = model.predict(seq, verbose=0)[0]
    else:
        row = X[-1:, :]   # (1, n_features)
        preds = model.predict(row)[0]

    return np.asarray(preds, dtype=float), float(pred_std)


def shap_chart_fig() -> go.Figure | None:
    shap_path = MODELS_DIR / "shap_importance.png"
    if not shap_path.exists():
        return None
    import base64
    with open(shap_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{img_b64}",
            xref="paper", yref="paper",
            x=0, y=1, sizex=1, sizey=1,
            sizing="contain", opacity=1, layer="below",
        )
    )
    fig.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=400,
    )
    return fig


def gauge_fig(aqi_val: float) -> go.Figure:
    label, color = aqi_category(aqi_val)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi_val,
        title={"text": f"Current AQI<br><span style='font-size:14px'>{label}</span>"},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0,   50],  "color": "#00e400"},
                {"range": [51,  100], "color": "#ffff00"},
                {"range": [101, 150], "color": "#ff7e00"},
                {"range": [151, 200], "color": "#ff0000"},
                {"range": [201, 300], "color": "#8f3f97"},
                {"range": [301, 500], "color": "#7e0023"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": 150,
            },
        },
    ))
    fig.update_layout(height=300, margin={"l": 20, "r": 20, "t": 40, "b": 0})
    return fig


def forecast_fig(
    history: pd.DataFrame,
    predictions: np.ndarray,
    pred_std: float,
) -> go.Figure:
    fig = go.Figure()

    # Historical AQI (last 48 h)
    if not history.empty and "aqi" in history.columns:
        ts = pd.to_datetime(history["timestamp_utc"])
        fig.add_trace(go.Scatter(
            x=ts, y=history["aqi"],
            mode="lines",
            name="Historical AQI",
            line={"color": "#2196F3", "width": 2},
        ))
        last_ts = ts.max()
    else:
        last_ts = pd.Timestamp.now()

    # Forecast points (Day 1, 2, 3)
    forecast_ts = [last_ts + pd.Timedelta(hours=24 * d) for d in range(1, 4)]
    upper = predictions + 1.5 * pred_std
    lower = np.maximum(predictions - 1.5 * pred_std, 0)

    # Confidence interval band
    fig.add_trace(go.Scatter(
        x=forecast_ts + forecast_ts[::-1],
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(255,152,0,0.2)",
        line={"color": "rgba(255,152,0,0)"},
        hoverinfo="skip",
        showlegend=False,
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_ts,
        y=predictions,
        mode="lines+markers",
        name="3-Day Forecast",
        line={"color": "#FF9800", "width": 3, "dash": "dot"},
        marker={"size": 10},
        text=[f"Day {d}: {v:.0f}" for d, v in enumerate(predictions, 1)],
        hovertemplate="%{text}<extra></extra>",
    ))

    # Unhealthy threshold line
    fig.add_hline(
        y=150, line_dash="dash",
        line_color="red", opacity=0.6,
        annotation_text="Unhealthy (150)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="AQI Forecast – Faisalabad",
        xaxis_title="Date / Time",
        yaxis_title="AQI",
        legend={"orientation": "h", "y": -0.2},
        height=400,
        hovermode="x unified",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#FAFAFA",
        xaxis={"gridcolor": "#333"},
        yaxis={"gridcolor": "#333"},
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🌫️ AQI Forecast Dashboard – Faisalabad, Pakistan")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        city = st.selectbox(
            "Select City",
            options=["Faisalabad", "Lahore", "Karachi", "Islamabad"],
            index=0,
        )
        show_raw = st.checkbox("Show raw feature data", value=False)
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Auto-refresh every 30 min • Last run: {pd.Timestamp.now().strftime('%H:%M:%S')}")

    # ── Load model ────────────────────────────────────────────────────────────
    info  = load_model_info()
    model = load_model(info)

    if not info or model is None:
        st.warning(
            "No trained model found. Run `python -m src.training_pipeline` first, "
            "or run the GitHub Actions training job."
        )
        model_ready = False
    else:
        model_ready = True
        model_name = info.get("model_name", "Unknown")
        metrics    = info.get("metrics", {})
        with st.sidebar:
            st.success(f"Model: **{model_name}**")
            st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
            st.metric("MAE",  f"{metrics.get('mae', 0):.2f}")
            st.metric("R²",   f"{metrics.get('r2', 0):.3f}")

    # ── Fetch features ────────────────────────────────────────────────────────
    features = fetch_latest_features(city, n_hours=48)

    if features.empty:
        st.error(
            "No feature data available for this city. "
            "Run the feature pipeline or backfill first."
        )
        st.stop()

    current_aqi = float(features["aqi"].iloc[-1]) if "aqi" in features.columns else float("nan")

    # ── Alert banner ──────────────────────────────────────────────────────────
    if not np.isnan(current_aqi) and current_aqi > 150:
        label, _ = aqi_category(current_aqi)
        st.error(
            f"🚨 **AIR QUALITY ALERT** – Current AQI is **{current_aqi:.0f}** ({label}). "
            "Avoid outdoor activities. Wear N95 masks if going outside.",
            icon="🚨",
        )

    # ── Run inference ─────────────────────────────────────────────────────────
    if model_ready:
        predictions, pred_std = run_inference(model, info, features)
    else:
        predictions = np.full(3, float("nan"))
        pred_std    = 10.0

    # ── Layout ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        st.metric(
            "Current AQI",
            f"{current_aqi:.0f}" if not np.isnan(current_aqi) else "N/A",
            delta=None,
        )
        label, color = aqi_category(current_aqi) if not np.isnan(current_aqi) else ("Unknown", "#888")
        st.markdown(f"<span style='background:{color};padding:4px 10px;border-radius:6px;color:#000;font-weight:bold'>{label}</span>", unsafe_allow_html=True)

    for day_idx, (col, day) in enumerate(zip([col2, col3, col4], ["Tomorrow", "Day 2", "Day 3"]), 1):
        pred_val = predictions[day_idx - 1] if model_ready and not np.isnan(predictions[day_idx - 1]) else float("nan")
        with col:
            st.metric(
                day,
                f"{pred_val:.0f}" if not np.isnan(pred_val) else "N/A",
            )
            if not np.isnan(pred_val):
                dlabel, dcolor = aqi_category(pred_val)
                st.markdown(f"<span style='background:{dcolor};padding:4px 10px;border-radius:6px;color:#000;font-size:12px'>{dlabel}</span>", unsafe_allow_html=True)

    st.divider()

    # ── Main chart row ────────────────────────────────────────────────────────
    chart_col, gauge_col = st.columns([2, 1])
    with chart_col:
        st.plotly_chart(
            forecast_fig(features, predictions, pred_std),
            width="stretch",
        )
    with gauge_col:
        if not np.isnan(current_aqi):
            st.plotly_chart(gauge_fig(current_aqi), width="stretch")
        else:
            st.info("Current AQI gauge unavailable")

    # ── SHAP chart ────────────────────────────────────────────────────────────
    st.subheader("Feature Importance (SHAP)")
    shap_fig = shap_chart_fig()
    if shap_fig:
        st.plotly_chart(shap_fig, width="stretch")
    else:
        st.info("SHAP chart not yet generated. Train the model to see feature importances.")

    # ── Raw data toggle ───────────────────────────────────────────────────────
    if show_raw:
        st.subheader("Raw Feature Data (last 48 h)")
        st.dataframe(features.tail(48), use_container_width=True)

    # ── AQI scale legend ─────────────────────────────────────────────────────
    with st.expander("📖 AQI Scale Reference"):
        for lo, hi, label, color in AQI_CATEGORIES:
            st.markdown(
                f"<span style='background:{color};padding:3px 8px;border-radius:4px;color:#000'>"
                f"**{lo}–{hi}**</span>  {label}",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
