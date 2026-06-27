"""
Apache Airflow DAG – AQI Training Pipeline

Runs once daily at midnight UTC. For each execution it:
  1. Loads feature data from the feature store (Hopsworks or local Parquet).
  2. Trains Random Forest, Ridge Regression, and LSTM models.
  3. Evaluates models and selects the best by RMSE.
  4. Generates SHAP feature-importance chart.
  5. Registers the best model to Hopsworks / MLflow.

Schedule: daily (0 0 * * *)
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
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.email import EmailOperator

log = logging.getLogger(__name__)

# ── Default arguments ─────────────────────────────────────────────────────────
default_args = {
    "owner": "aqi_team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(hours=1),
    "email_on_failure": True,
    "email_on_retry": False,
    "email": [os.getenv("ALERT_EMAIL", "haseebahamadnoul@gmail.com")],
}

MIN_DATA_ROWS = 500   # skip training if feature store has fewer than this


# ── Task callables ─────────────────────────────────────────────────────────────

def task_check_data_available(**context) -> bool:
    """Short-circuit: skip training if there's not enough data."""
    from utils.store_client import FeatureStore
    from src.config import (
        HOPSWORKS_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST,
        FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, DEFAULT_CITY,
    )

    store = FeatureStore(
        hopsworks_key=HOPSWORKS_KEY,
        hopsworks_project=HOPSWORKS_PROJECT,
        hopsworks_host=HOPSWORKS_HOST,
        fg_name=FEATURE_GROUP_NAME,
        fg_version=FEATURE_GROUP_VERSION,
    )
    df = store.read(city=DEFAULT_CITY)
    n_rows = len(df)
    log.info("Feature store row count for %s: %d", DEFAULT_CITY, n_rows)

    if n_rows < MIN_DATA_ROWS:
        log.warning(
            "Insufficient data (%d rows < %d minimum). "
            "Skipping training. Run backfill_pipeline first.",
            n_rows, MIN_DATA_ROWS,
        )
        return False

    context["ti"].xcom_push(key="n_rows", value=n_rows)
    return True


def task_train_models(**context) -> dict:
    """
    Load features, train RF + Ridge + LSTM, select best model.
    Returns metrics dict for the best model.
    """
    from src.training_pipeline import (
        load_data,
        prepare_dataset,
        temporal_train_val_test_split,
        train_random_forest,
        train_ridge,
        train_lstm,
        explain_with_shap,
        save_model_locally,
        register_model_hopsworks,
    )
    from src.config import MODELS_DIR
    import pandas as pd

    log.info("[Training DAG] Loading data from feature store...")
    df = load_data()

    log.info("[Training DAG] Preparing dataset...")
    X_flat, X_seq, y, feat_names = prepare_dataset(df)
    splits = temporal_train_val_test_split(X_flat, X_seq, y)

    results = {}
    trained = {}
    preds = {}
    stds = {}

    log.info("[Training DAG] Training Random Forest...")
    rf_model, rf_metrics, rf_pred, rf_std = train_random_forest(splits, feat_names)
    results["RandomForest"] = rf_metrics
    trained["RandomForest"] = rf_model
    preds["RandomForest"] = rf_pred
    stds["RandomForest"] = rf_std

    log.info("[Training DAG] Training Ridge Regression...")
    ridge_model, ridge_metrics, ridge_pred, ridge_std = train_ridge(splits)
    results["Ridge"] = ridge_metrics
    trained["Ridge"] = ridge_model
    preds["Ridge"] = ridge_pred
    stds["Ridge"] = ridge_std

    log.info("[Training DAG] Training LSTM...")
    lstm_model, lstm_metrics, lstm_pred, lstm_std = train_lstm(splits)
    results["LSTM"] = lstm_metrics
    trained["LSTM"] = lstm_model
    preds["LSTM"] = lstm_pred
    stds["LSTM"] = lstm_std

    # Select best model
    comparison = pd.DataFrame(results).T.sort_values("rmse")
    best_name = comparison.index[0]
    best_model = trained[best_name]
    best_metrics = results[best_name]
    best_std = stds[best_name]

    log.info(
        "[Training DAG] Best model: %s  RMSE=%.4f  MAE=%.4f  R²=%.4f",
        best_name, best_metrics["rmse"], best_metrics["mae"], best_metrics["r2"],
    )

    # SHAP explanation
    shap_path = MODELS_DIR / "shap_importance.png"
    explain_with_shap(best_model, best_name, splits["X_flat_test"], feat_names, shap_path)

    # Save model artifacts
    save_model_locally(best_model, best_name, best_metrics, best_std)

    # Register to Hopsworks / MLflow
    register_model_hopsworks(best_model, best_name, best_metrics)

    context["ti"].xcom_push(key="best_model_name", value=best_name)
    context["ti"].xcom_push(key="best_metrics", value=best_metrics)

    return {"model": best_name, "metrics": best_metrics}


def task_validate_model(**context) -> None:
    """
    Post-training validation:
    - Check that model artifacts exist on disk.
    - Log a warning if RMSE > threshold.
    """
    import joblib
    from src.config import MODELS_DIR

    RMSE_WARNING_THRESHOLD = 30.0

    info_path = MODELS_DIR / "best_model_info.pkl"
    if not info_path.exists():
        raise FileNotFoundError(f"Model info not found at {info_path}")

    info = joblib.load(info_path)
    rmse = info.get("metrics", {}).get("rmse", float("nan"))
    model_name = info.get("model_name", "Unknown")

    log.info("[Validation] Model=%s  RMSE=%.4f", model_name, rmse)

    if rmse > RMSE_WARNING_THRESHOLD:
        log.warning(
            "[Validation] RMSE %.4f exceeds warning threshold %.1f. "
            "Consider collecting more training data.",
            rmse, RMSE_WARNING_THRESHOLD,
        )

    model_path = info.get("model_path", "")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    log.info("[Validation] All artifacts verified ✓")


def task_alert_aqi_if_hazardous(**context) -> None:
    """
    Check if the current AQI is hazardous and log an alert.
    In production, this would send email/Slack notifications.
    """
    from utils.store_client import FeatureStore
    from src.config import (
        HOPSWORKS_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST,
        FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, DEFAULT_CITY,
        AQI_CATEGORIES,
    )

    store = FeatureStore(
        hopsworks_key=HOPSWORKS_KEY,
        hopsworks_project=HOPSWORKS_PROJECT,
        hopsworks_host=HOPSWORKS_HOST,
        fg_name=FEATURE_GROUP_NAME,
        fg_version=FEATURE_GROUP_VERSION,
    )
    df = store.read_latest(city=DEFAULT_CITY, n_hours=2)
    if df.empty or "aqi" not in df.columns:
        log.info("[Alert] No recent data available for AQI alert check")
        return

    current_aqi = float(df["aqi"].iloc[-1])

    def _category(aqi):
        for lo, hi, label, _ in AQI_CATEGORIES:
            if lo <= aqi <= hi:
                return label
        return "Hazardous"

    category = _category(current_aqi)
    log.info("[Alert] Current AQI for %s: %.0f (%s)", DEFAULT_CITY, current_aqi, category)

    if current_aqi > 150:
        log.warning(
            "🚨 AQI ALERT for %s: AQI=%.0f (%s) – Outdoor activity not recommended!",
            DEFAULT_CITY, current_aqi, category,
        )


# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="aqi_training_pipeline",
    description="Daily AQI model training pipeline for Faisalabad, Pakistan",
    default_args=default_args,
    schedule_interval="0 0 * * *",   # midnight UTC daily
    catchup=False,
    max_active_runs=1,
    tags=["aqi", "training-pipeline", "faisalabad", "ml"],
    doc_md=__doc__,
) as dag:

    check_data = ShortCircuitOperator(
        task_id="check_data_available",
        python_callable=task_check_data_available,
        execution_timeout=timedelta(minutes=5),
    )

    train = PythonOperator(
        task_id="train_models",
        python_callable=task_train_models,
        execution_timeout=timedelta(hours=2),
    )

    validate = PythonOperator(
        task_id="validate_model",
        python_callable=task_validate_model,
        execution_timeout=timedelta(minutes=5),
    )

    alert_check = PythonOperator(
        task_id="alert_aqi_if_hazardous",
        python_callable=task_alert_aqi_if_hazardous,
        trigger_rule="all_done",
        execution_timeout=timedelta(minutes=5),
    )

    check_data >> train >> validate >> alert_check
