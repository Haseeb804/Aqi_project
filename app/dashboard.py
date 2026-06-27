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
import plotly.express as px
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
    POLLUTANTS,
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


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS, show_spinner="Loading historical data...")
def fetch_all_features(city: str) -> pd.DataFrame:
    store = get_store()
    return store.read(city=city)


def run_inference(model, info: dict, features: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Returns (predictions shape (3,), prediction_std)."""
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
        seq = X[-lookback:][np.newaxis, :, :]
        preds = model.predict(seq, verbose=0)[0]
    else:
        row = X[-1:, :]
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
        height=420,
        paper_bgcolor="#0e1117",
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
    fig.update_layout(
        height=300, margin={"l": 20, "r": 20, "t": 40, "b": 0},
        paper_bgcolor="#0e1117", font_color="#FAFAFA",
    )
    return fig


def forecast_fig(history: pd.DataFrame, predictions: np.ndarray, pred_std: float) -> go.Figure:
    fig = go.Figure()

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

    forecast_ts = [last_ts + pd.Timedelta(hours=24 * d) for d in range(1, 4)]
    upper = predictions + 1.5 * pred_std
    lower = np.maximum(predictions - 1.5 * pred_std, 0)

    fig.add_trace(go.Scatter(
        x=forecast_ts + forecast_ts[::-1],
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(255,152,0,0.2)",
        line={"color": "rgba(255,152,0,0)"},
        hoverinfo="skip",
        showlegend=False,
    ))

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
        height=420,
        hovermode="x unified",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#FAFAFA",
        xaxis={"gridcolor": "#333"},
        yaxis={"gridcolor": "#333"},
    )
    return fig


# ── EDA chart helpers ─────────────────────────────────────────────────────────

def _eda_add_category(df: pd.DataFrame) -> pd.DataFrame:
    def _cat(aqi):
        for lo, hi, label, _ in AQI_CATEGORIES:
            if lo <= aqi <= hi:
                return label
        return "Hazardous"
    df = df.copy()
    if "aqi" in df.columns:
        df["aqi_category"] = df["aqi"].apply(_cat)
    return df


def eda_distribution_fig(df: pd.DataFrame) -> go.Figure:
    df = _eda_add_category(df)
    AQI_LABELS = [l for _, _, l, _ in AQI_CATEGORIES]
    AQI_COLORS_MAP = {l: c for _, _, l, c in AQI_CATEGORIES}
    counts = df["aqi_category"].value_counts()
    ordered = [l for l in AQI_LABELS if l in counts.index]

    fig = go.Figure(go.Bar(
        x=ordered,
        y=[counts[c] for c in ordered],
        marker_color=[AQI_COLORS_MAP[c] for c in ordered],
        text=[f"{counts[c]:,}" for c in ordered],
        textposition="outside",
    ))
    fig.update_layout(
        title="AQI Reading Count by Category",
        xaxis_title="Category", yaxis_title="Count",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#FAFAFA", height=380,
        xaxis={"gridcolor": "#333"}, yaxis={"gridcolor": "#333"},
    )
    return fig


def eda_monthly_trend_fig(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df["year_month"] = df["timestamp_utc"].dt.to_period("M")
    monthly = (
        df.groupby("year_month")["aqi"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    monthly["dt"] = monthly["year_month"].dt.to_timestamp()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["dt"],
        y=monthly["mean"] + monthly["std"],
        mode="lines", line={"color": "rgba(33,150,243,0)"},
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["dt"],
        y=monthly["mean"] - monthly["std"],
        fill="tonexty",
        fillcolor="rgba(33,150,243,0.2)",
        mode="lines", line={"color": "rgba(33,150,243,0)"},
        name="±1 Std Dev", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["dt"],
        y=monthly["mean"],
        mode="lines+markers",
        name="Monthly Mean AQI",
        line={"color": "#2196F3", "width": 2},
        marker={"size": 5},
    ))
    fig.add_hline(y=150, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="Unhealthy (150)")
    fig.update_layout(
        title="Annual AQI Trend (Monthly Mean ± Std)",
        xaxis_title="Date", yaxis_title="AQI",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#FAFAFA", height=380,
        xaxis={"gridcolor": "#333"}, yaxis={"gridcolor": "#333"},
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def eda_heatmap_fig(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    if "hour" not in df.columns:
        df["hour"] = df["timestamp_utc"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp_utc"].dt.dayofweek

    pivot = (
        df.groupby(["day_of_week", "hour"])["aqi"]
        .mean()
        .reset_index()
        .pivot(index="day_of_week", columns="hour", values="aqi")
    )
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.index = [day_names[i] for i in pivot.index if i < 7]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        colorbar={"title": "Mean AQI"},
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Mean AQI: %{z:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title="Mean AQI by Hour × Day of Week",
        xaxis_title="Hour of Day (UTC)", yaxis_title="",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#FAFAFA", height=380,
    )
    return fig


def eda_correlation_fig(df: pd.DataFrame) -> go.Figure:
    cols = ["aqi"] + [p for p in POLLUTANTS if p in df.columns]
    corr = df[cols].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        colorbar={"title": "r"},
        hovertemplate="X: %{x}<br>Y: %{y}<br>r = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Pollutant Correlation Heatmap",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#FAFAFA", height=420,
    )
    return fig


def eda_pollutant_ts_fig(df: pd.DataFrame, pollutant: str) -> go.Figure:
    sample = df.sort_values("timestamp_utc").tail(720)
    ts = pd.to_datetime(sample["timestamp_utc"])
    pol_labels = {"pm25": "PM2.5 (μg/m³)", "pm10": "PM10 (μg/m³)",
                  "no2": "NO₂ (μg/m³)", "o3": "O₃ (μg/m³)",
                  "co": "CO (μg/m³)", "so2": "SO₂ (μg/m³)"}
    colors = {"pm25": "#2196F3", "pm10": "#FF9800", "no2": "#4CAF50",
              "o3": "#9C27B0", "co": "#F44336", "so2": "#00BCD4"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=sample[pollutant],
        mode="lines",
        fill="tozeroy",
        name=pol_labels.get(pollutant, pollutant),
        line={"color": colors.get(pollutant, "#2196F3"), "width": 1.2},
        fillcolor=colors.get(pollutant, "#2196F3").replace("#", "rgba(") + ",0.2)",
    ))
    fig.update_layout(
        title=f"{pol_labels.get(pollutant, pollutant)} – Last 30 Days",
        xaxis_title="Date", yaxis_title=pol_labels.get(pollutant, pollutant),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#FAFAFA", height=320,
        xaxis={"gridcolor": "#333"}, yaxis={"gridcolor": "#333"},
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

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_forecast, tab_eda, tab_about = st.tabs(["📈 Forecast", "🔬 EDA", "ℹ️ About"])

    # ─────────────────────────────── FORECAST TAB ────────────────────────────
    with tab_forecast:
        features = fetch_latest_features(city, n_hours=48)

        if features.empty:
            st.error(
                "No feature data available for this city. "
                "Run the feature pipeline or backfill first."
            )
            st.stop()

        current_aqi = float(features["aqi"].iloc[-1]) if "aqi" in features.columns else float("nan")

        # Alert banner
        if not np.isnan(current_aqi) and current_aqi > 150:
            label, _ = aqi_category(current_aqi)
            st.error(
                f"🚨 **AIR QUALITY ALERT** – Current AQI is **{current_aqi:.0f}** ({label}). "
                "Avoid outdoor activities. Wear N95 masks if going outside.",
                icon="🚨",
            )

        # Run inference
        if model_ready:
            predictions, pred_std = run_inference(model, info, features)
        else:
            predictions = np.full(3, float("nan"))
            pred_std    = 10.0

        # Metric cards
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            st.metric(
                "Current AQI",
                f"{current_aqi:.0f}" if not np.isnan(current_aqi) else "N/A",
            )
            label, color = aqi_category(current_aqi) if not np.isnan(current_aqi) else ("Unknown", "#888")
            st.markdown(
                f"<span style='background:{color};padding:4px 10px;border-radius:6px;"
                f"color:#000;font-weight:bold'>{label}</span>",
                unsafe_allow_html=True,
            )

        for day_idx, (col, day) in enumerate(zip([col2, col3, col4], ["Tomorrow", "Day 2", "Day 3"]), 1):
            pred_val = (
                predictions[day_idx - 1]
                if model_ready and not np.isnan(predictions[day_idx - 1])
                else float("nan")
            )
            with col:
                st.metric(day, f"{pred_val:.0f}" if not np.isnan(pred_val) else "N/A")
                if not np.isnan(pred_val):
                    dlabel, dcolor = aqi_category(pred_val)
                    st.markdown(
                        f"<span style='background:{dcolor};padding:4px 10px;border-radius:6px;"
                        f"color:#000;font-size:12px'>{dlabel}</span>",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # Main charts
        chart_col, gauge_col = st.columns([2, 1])
        with chart_col:
            st.plotly_chart(forecast_fig(features, predictions, pred_std), use_container_width=True)
        with gauge_col:
            if not np.isnan(current_aqi):
                st.plotly_chart(gauge_fig(current_aqi), use_container_width=True)
            else:
                st.info("Current AQI gauge unavailable")

        # SHAP
        st.subheader("🔍 Feature Importance (SHAP)")
        shap_fig = shap_chart_fig()
        if shap_fig:
            st.plotly_chart(shap_fig, use_container_width=True)
        else:
            st.info("SHAP chart not yet generated. Train the model to see feature importances.")

        # Raw data toggle
        if show_raw:
            st.subheader("Raw Feature Data (last 48 h)")
            st.dataframe(features.tail(48), use_container_width=True)

        # AQI scale legend
        with st.expander("📖 AQI Scale Reference"):
            for lo, hi, label, color in AQI_CATEGORIES:
                st.markdown(
                    f"<span style='background:{color};padding:3px 8px;border-radius:4px;color:#000'>"
                    f"**{lo}–{hi}**</span>  {label}",
                    unsafe_allow_html=True,
                )

    # ─────────────────────────────── EDA TAB ─────────────────────────────────
    with tab_eda:
        st.subheader("🔬 Exploratory Data Analysis")
        st.caption("Interactive charts based on all available historical feature data.")

        all_features = fetch_all_features(city)

        if all_features.empty:
            st.warning(
                "No historical data loaded. Run the backfill pipeline:\n"
                "`python -m src.backfill_pipeline --start 2022-01-01 --end 2024-01-01`"
            )
        else:
            all_features["timestamp_utc"] = pd.to_datetime(all_features["timestamp_utc"])

            # Summary stats
            st.info(
                f"📊  **{len(all_features):,}** hourly readings  •  "
                f"From **{all_features['timestamp_utc'].min().date()}** "
                f"to **{all_features['timestamp_utc'].max().date()}**"
            )

            # Row 1
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(eda_distribution_fig(all_features), use_container_width=True)
            with c2:
                st.plotly_chart(eda_monthly_trend_fig(all_features), use_container_width=True)

            # Row 2
            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(eda_heatmap_fig(all_features), use_container_width=True)
            with c4:
                st.plotly_chart(eda_correlation_fig(all_features), use_container_width=True)

            # Pollutant selector
            st.subheader("Pollutant Time-Series (Last 30 Days)")
            avail = [p for p in POLLUTANTS if p in all_features.columns]
            pol_labels = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO₂",
                          "o3": "O₃", "co": "CO", "so2": "SO₂"}
            if avail:
                selected_pol = st.selectbox(
                    "Select pollutant",
                    options=avail,
                    format_func=lambda p: pol_labels.get(p, p),
                )
                st.plotly_chart(eda_pollutant_ts_fig(all_features, selected_pol), use_container_width=True)

            # Descriptive stats table
            with st.expander("📋 Descriptive Statistics"):
                cols = ["aqi"] + [p for p in POLLUTANTS if p in all_features.columns]
                st.dataframe(all_features[cols].describe().round(2), use_container_width=True)

    # ─────────────────────────────── ABOUT TAB ───────────────────────────────
    with tab_about:
        st.subheader("ℹ️ About This Project")
        st.markdown("""
**AQI Prediction System – Faisalabad, Pakistan**

An end-to-end serverless machine learning pipeline for 3-day Air Quality Index (AQI) forecasting.

### 🛠️ Technology Stack
| Component | Technology |
|-----------|-----------|
| Data Sources | AQICN API, OpenWeatherMap API |
| Feature Store | Hopsworks (cloud) / Local Parquet (fallback) |
| ML Models | Random Forest, Ridge Regression, LSTM (TensorFlow) |
| Model Tracking | MLflow |
| Explainability | SHAP / Feature Importance |
| Automation | GitHub Actions (hourly + daily) + Apache Airflow |
| Dashboard | Streamlit |
| API | FastAPI |

### 📐 Pipeline Overview
1. **Feature Pipeline** – Hourly data fetch → feature engineering → feature store upsert
2. **Backfill Pipeline** – Historical OpenWeather data for model training
3. **Training Pipeline** – RF + Ridge + LSTM training, evaluation, SHAP, model registry
4. **Dashboard** – Real-time forecast + EDA charts + hazardous AQI alerts

### 📊 Features Engineered
- **Raw**: PM2.5, PM10, NO₂, O₃, CO, SO₂
- **Time**: hour, day_of_week, month, season
- **Rolling averages**: 3h / 6h / 12h for all pollutants + AQI
- **Derived**: AQI change rate (momentum)

### 🔗 Links
- [GitHub Repository](https://github.com/Haseeb804/Aqi_project)
- [AQICN API](https://aqicn.org/api/)
- [OpenWeather Air Pollution API](https://openweathermap.org/api/air-pollution)
- [Hopsworks Feature Store](https://www.hopsworks.ai/)
        """)


if __name__ == "__main__":
    main()
