"""Central configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# ── API Keys ──────────────────────────────────────────────────────────────────
AQICN_KEY: str = os.environ["AQICN_KEY"]
OPENWEATHER_KEY: str = os.environ["OPENWEATHER_KEY"]
HOPSWORKS_KEY: str = os.getenv("HOPSWORKS_KEY", "")
HOPSWORKS_PROJECT: str = os.getenv("HOPSWORKS_PROJECT", "aqi_faisalabad")
HOPSWORKS_HOST: str = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")

# ── City ──────────────────────────────────────────────────────────────────────
DEFAULT_CITY: str = os.getenv("DEFAULT_CITY", "Faisalabad")
CITY_LAT: float = float(os.getenv("CITY_LAT", "31.4504"))
CITY_LON: float = float(os.getenv("CITY_LON", "73.1350"))

# AQICN geo-based station query (most reliable for less-monitored cities)
AQICN_STATION: str = f"geo:{CITY_LAT};{CITY_LON}"

# ── Feature Store ─────────────────────────────────────────────────────────────
FEATURE_GROUP_NAME: str = "aqi_features"
FEATURE_GROUP_VERSION: int = 1
FEATURE_VIEW_NAME: str = "aqi_feature_view"
FEATURE_VIEW_VERSION: int = 1

# ── ML ────────────────────────────────────────────────────────────────────────
MODEL_LOOKBACK: int = int(os.getenv("MODEL_LOOKBACK_HOURS", "24"))
FORECAST_DAYS: int = int(os.getenv("FORECAST_DAYS", "3"))
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", str(_ROOT / "mlruns"))
MLFLOW_EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", "aqi_forecasting")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = _ROOT
DATA_DIR: Path = _ROOT / "data"
MODELS_DIR: Path = _ROOT / "models"
LOGS_DIR: Path = _ROOT / "logs"

for _d in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Pollutants ────────────────────────────────────────────────────────────────
POLLUTANTS: list[str] = ["pm25", "pm10", "no2", "o3", "co", "so2"]

FEATURE_COLS: list[str] = [
    # Raw pollutants
    "pm25", "pm10", "no2", "o3", "co", "so2",
    # Time
    "hour", "day_of_week", "month", "season",
    # Derived
    "aqi_change_rate",
    # Rolling 3h
    "pm25_3h_avg", "pm10_3h_avg", "no2_3h_avg",
    "o3_3h_avg",   "co_3h_avg",   "so2_3h_avg",
    # Rolling 6h
    "pm25_6h_avg", "pm10_6h_avg", "no2_6h_avg",
    "o3_6h_avg",   "co_6h_avg",   "so2_6h_avg",
    # Rolling 12h
    "pm25_12h_avg", "pm10_12h_avg", "no2_12h_avg",
    "o3_12h_avg",   "co_12h_avg",   "so2_12h_avg",
    # AQI rolling
    "aqi_3h_avg", "aqi_6h_avg", "aqi_12h_avg",
]

TARGET_COLS: list[str] = ["target_day1", "target_day2", "target_day3"]

# AQI category thresholds (US EPA)
AQI_CATEGORIES: list[tuple[int, int, str, str]] = [
    (0,   50,  "Good",                 "#00e400"),
    (51,  100, "Moderate",             "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy",            "#ff0000"),
    (201, 300, "Very Unhealthy",       "#8f3f97"),
    (301, 500, "Hazardous",            "#7e0023"),
]
