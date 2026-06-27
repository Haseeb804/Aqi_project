"""
Apache Airflow DAG – AQI Feature Pipeline

Runs every hour. For each execution it:
  1. Fetches the current AQI reading for Faisalabad.
  2. Engineers time and rolling features.
  3. Upserts the row to the feature store.

Schedule: hourly (0 * * * *)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is importable inside Airflow workers
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator

log = logging.getLogger(__name__)

# ── Default arguments ─────────────────────────────────────────────────────────
default_args = {
    "owner": "aqi_team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "email_on_failure": True,
    "email_on_retry": False,
    "email": [os.getenv("ALERT_EMAIL", "haseebahamadnoul@gmail.com")],
}


# ── Task callables ─────────────────────────────────────────────────────────────

def task_fetch_and_engineer(**context) -> dict:
    """Fetch current AQI and engineer features."""
    from utils.api_client import AQICNClient, OpenWeatherClient, fetch_current_with_fallback
    from utils.feature_engineering import engineer_features, merge_with_history
    from utils.store_client import FeatureStore
    from src.config import (
        AQICN_KEY, AQICN_STATION, CITY_LAT, CITY_LON, DEFAULT_CITY,
        OPENWEATHER_KEY, HOPSWORKS_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST,
        FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
    )
    import pandas as pd

    aqicn = AQICNClient(api_key=AQICN_KEY)
    ow    = OpenWeatherClient(api_key=OPENWEATHER_KEY)

    log.info("Fetching current AQI for %s", DEFAULT_CITY)
    raw = fetch_current_with_fallback(aqicn, ow, AQICN_STATION, CITY_LAT, CITY_LON)
    raw["city"] = DEFAULT_CITY

    store = FeatureStore(
        hopsworks_key=HOPSWORKS_KEY,
        hopsworks_project=HOPSWORKS_PROJECT,
        hopsworks_host=HOPSWORKS_HOST,
        fg_name=FEATURE_GROUP_NAME,
        fg_version=FEATURE_GROUP_VERSION,
    )
    history = store.read_latest(city=DEFAULT_CITY, n_hours=13)

    if not history.empty:
        combined = merge_with_history(raw, history)
    else:
        combined = engineer_features(pd.DataFrame([raw]))

    new_row = combined.sort_values("timestamp_utc").tail(1).copy()

    # Push to XCom so the store task can use it
    context["ti"].xcom_push(key="new_row_json", value=new_row.to_json(orient="records"))
    log.info("AQI=%.1f  timestamp=%s", raw.get("aqi", 0), new_row["timestamp_utc"].iloc[0])
    return {"status": "ok", "aqi": raw.get("aqi")}


def task_store_features(**context) -> None:
    """Upsert the engineered row into the feature store."""
    import json
    import pandas as pd
    from utils.store_client import FeatureStore
    from src.config import (
        HOPSWORKS_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST,
        FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
    )

    raw_json = context["ti"].xcom_pull(task_ids="fetch_and_engineer", key="new_row_json")
    if not raw_json:
        raise ValueError("No data received from fetch_and_engineer task")

    new_row = pd.read_json(raw_json, orient="records")

    store = FeatureStore(
        hopsworks_key=HOPSWORKS_KEY,
        hopsworks_project=HOPSWORKS_PROJECT,
        hopsworks_host=HOPSWORKS_HOST,
        fg_name=FEATURE_GROUP_NAME,
        fg_version=FEATURE_GROUP_VERSION,
    )
    store.upsert(new_row, tag="live")
    log.info("Stored 1 row to feature store")


def task_check_aqi_alert(**context) -> None:
    """Log a warning if current AQI exceeds 150 (Unhealthy threshold)."""
    result = context["ti"].xcom_pull(task_ids="fetch_and_engineer")
    aqi = result.get("aqi", 0) if result else 0
    if aqi and aqi > 150:
        log.warning("AQI ALERT: current AQI=%.0f exceeds Unhealthy threshold (150)", aqi)


# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="aqi_feature_pipeline",
    description="Hourly AQI feature ingestion for Faisalabad, Pakistan",
    default_args=default_args,
    schedule_interval="0 * * * *",   # every hour at :00
    catchup=False,
    max_active_runs=1,
    tags=["aqi", "feature-pipeline", "faisalabad"],
    doc_md=__doc__,
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_and_engineer",
        python_callable=task_fetch_and_engineer,
        execution_timeout=timedelta(minutes=10),
    )

    store_task = PythonOperator(
        task_id="store_features",
        python_callable=task_store_features,
        execution_timeout=timedelta(minutes=5),
    )

    alert_task = PythonOperator(
        task_id="check_aqi_alert",
        python_callable=task_check_aqi_alert,
        trigger_rule="all_done",   # run even if upstream failed
    )

    fetch_task >> store_task >> alert_task
