"""
Live hourly feature pipeline.

Fetches current AQI data for Faisalabad, engineers features using the last
12 h of history for accurate rolling averages, and upserts to the feature store.

Can be run standalone:
    python -m src.feature_pipeline

Or called by the Airflow DAG / GitHub Actions workflow.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import (
    AQICN_KEY,
    AQICN_STATION,
    CITY_LAT,
    CITY_LON,
    DEFAULT_CITY,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    HOPSWORKS_HOST,
    HOPSWORKS_KEY,
    HOPSWORKS_PROJECT,
    LOGS_DIR,
    OPENWEATHER_KEY,
)
from utils.api_client import AQICNClient, OpenWeatherClient, fetch_current_with_fallback
from utils.feature_engineering import engineer_features, merge_with_history
from utils.store_client import FeatureStore

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "feature_pipeline.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


def run_pipeline() -> pd.DataFrame:
    """
    Execute one pipeline iteration:
    1. Fetch current AQI reading.
    2. Load recent history for rolling feature computation.
    3. Compute features.
    4. Upsert to feature store.
    Returns the single-row engineered DataFrame.
    """
    log.info("=== Feature pipeline started ===")

    # ── 1. Initialise clients ──────────────────────────────────────────────
    aqicn = AQICNClient(api_key=AQICN_KEY)
    ow    = OpenWeatherClient(api_key=OPENWEATHER_KEY)
    store = FeatureStore(
        hopsworks_key=HOPSWORKS_KEY,
        hopsworks_project=HOPSWORKS_PROJECT,
        hopsworks_host=HOPSWORKS_HOST,
        fg_name=FEATURE_GROUP_NAME,
        fg_version=FEATURE_GROUP_VERSION,
    )

    # ── 2. Fetch current reading ───────────────────────────────────────────
    log.info("Fetching current AQI for %s (station=%s)", DEFAULT_CITY, AQICN_STATION)
    raw = fetch_current_with_fallback(
        aqicn_client=aqicn,
        ow_client=ow,
        station=AQICN_STATION,
        lat=CITY_LAT,
        lon=CITY_LON,
    )
    raw["city"] = DEFAULT_CITY
    log.info(
        "Got reading: AQI=%.1f  PM2.5=%.1f  PM10=%.1f  NO2=%.1f  O3=%.1f",
        raw.get("aqi", float("nan")),
        raw.get("pm25", float("nan")),
        raw.get("pm10", float("nan")),
        raw.get("no2",  float("nan")),
        raw.get("o3",   float("nan")),
    )

    # ── 3. Load recent history for rolling averages ────────────────────────
    history = store.read_latest(city=DEFAULT_CITY, n_hours=13)  # 12h + margin
    log.info("Loaded %d historical rows for rolling averages", len(history))

    # ── 4. Merge & engineer features ──────────────────────────────────────
    if not history.empty:
        combined = merge_with_history(raw, history)
    else:
        # No history yet: engineer standalone (rolling avgs = current values)
        combined = engineer_features(pd.DataFrame([raw]))

    # Keep only the latest (new) row
    combined = combined.sort_values("timestamp_utc")
    new_row = combined.tail(1).copy()

    log.info(
        "Engineered %d features for timestamp %s",
        len(new_row.columns),
        new_row["timestamp_utc"].iloc[0],
    )

    # ── 5. Upsert ──────────────────────────────────────────────────────────
    store.upsert(new_row, tag="live")
    log.info("Upserted 1 row to feature store")
    log.info("=== Feature pipeline complete ===")

    return new_row


if __name__ == "__main__":
    try:
        result = run_pipeline()
        print("\nPipeline output:")
        print(result.to_string())
    except Exception:
        log.exception("Feature pipeline failed")
        sys.exit(1)
