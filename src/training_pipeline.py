"""
ML Training Pipeline – 3-day AQI Forecasting.

Trains three models (Random Forest, Ridge Regression, LSTM), evaluates them on
a held-out test set, explains the best model with SHAP, and registers it to
the model registry (Hopsworks or MLflow local).

Run:
    python -m src.training_pipeline
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

from src.config import (
    DATA_DIR,
    FEATURE_COLS,
    FORECAST_DAYS,
    HOPSWORKS_HOST,
    HOPSWORKS_KEY,
    HOPSWORKS_PROJECT,
    LOGS_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_LOOKBACK,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COLS,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from utils.feature_engineering import engineer_features
from utils.store_client import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "training.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load feature data from Hopsworks or local Parquet fallback."""
    store = FeatureStore(
        hopsworks_key=HOPSWORKS_KEY,
        hopsworks_project=HOPSWORKS_PROJECT,
        hopsworks_host=HOPSWORKS_HOST,
        fg_name=FEATURE_GROUP_NAME,
        fg_version=FEATURE_GROUP_VERSION,
    )
    df = store.read()

    if df.empty:
        # Try local parquet directly
        parquet_path = DATA_DIR / "aqi_features.parquet"
        if parquet_path.exists():
            log.info("Loading from local Parquet: %s", parquet_path)
            df = pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(
                f"No data found in feature store or at {parquet_path}. "
                "Run the backfill pipeline first."
            )

    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


# ── Feature preparation ───────────────────────────────────────────────────────

def prepare_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Engineer targets, clean data, and return arrays for training.

    Returns
    -------
    X_flat    : (N, n_features) – used by RF and Ridge
    X_seq     : (N, lookback, n_features) – used by LSTM
    y         : (N, 3) – targets [day1, day2, day3]
    feat_names: list of feature column names
    """
    log.info("Engineering targets and creating ML dataset...")
    df = engineer_features(df, add_targets=True, forecast_days=FORECAST_DAYS)

    # Drop rows with NaN in any feature or target
    all_cols = FEATURE_COLS + TARGET_COLS
    available = [c for c in all_cols if c in df.columns]
    df = df[["timestamp_utc", "city"] + available].dropna().reset_index(drop=True)

    if len(df) < MODEL_LOOKBACK + 72 + 10:
        raise ValueError(
            f"Insufficient data: {len(df)} rows. Need at least "
            f"{MODEL_LOOKBACK + 72 + 10} for training."
        )

    feat_names = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_names].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)

    # Create LSTM sequences
    X_seq_list: list[np.ndarray] = []
    X_flat_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for i in range(MODEL_LOOKBACK, len(X)):
        X_seq_list.append(X[i - MODEL_LOOKBACK:i])
        X_flat_list.append(X[i])
        y_list.append(y[i])

    X_seq  = np.array(X_seq_list,  dtype=np.float32)
    X_flat = np.array(X_flat_list, dtype=np.float32)
    y_arr  = np.array(y_list,      dtype=np.float32)

    log.info(
        "Dataset: X_flat=%s, X_seq=%s, y=%s",
        X_flat.shape, X_seq.shape, y_arr.shape,
    )
    return X_flat, X_seq, y_arr, feat_names


def temporal_train_val_test_split(
    X_flat: np.ndarray,
    X_seq: np.ndarray,
    y: np.ndarray,
) -> tuple:
    """70/15/15 temporal split – NO shuffling to prevent data leakage."""
    n = len(y)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    splits = {
        "X_flat_train": X_flat[:train_end],
        "X_flat_val":   X_flat[train_end:val_end],
        "X_flat_test":  X_flat[val_end:],
        "X_seq_train":  X_seq[:train_end],
        "X_seq_val":    X_seq[train_end:val_end],
        "X_seq_test":   X_seq[val_end:],
        "y_train":      y[:train_end],
        "y_val":        y[train_end:val_end],
        "y_test":       y[val_end:],
    }
    log.info(
        "Splits – train: %d, val: %d, test: %d",
        train_end, val_end - train_end, n - val_end,
    )
    return splits


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    y_true_f = y_true.ravel()
    y_pred_f = y_pred.ravel()
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true_f, y_pred_f))),
        "mae":  float(mean_absolute_error(y_true_f, y_pred_f)),
        "r2":   float(r2_score(y_true_f, y_pred_f)),
    }
    log.info("[%s] RMSE=%.3f  MAE=%.3f  R²=%.3f", name, *metrics.values())
    return metrics


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_random_forest(splits: dict, feat_names: list[str]) -> tuple:
    log.info("Training Random Forest...")
    X_tr, y_tr = splits["X_flat_train"], splits["y_train"]
    X_te, y_te = splits["X_flat_test"],  splits["y_test"]

    rf = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        n_jobs=-1,
    )

    # 5-fold CV on training set (flatten multi-output to single RMSE)
    kf = KFold(n_splits=5, shuffle=False)
    cv_rmse_scores: list[float] = []
    for tr_idx, val_idx in kf.split(X_tr):
        rf.fit(X_tr[tr_idx], y_tr[tr_idx])
        pred = rf.predict(X_tr[val_idx])
        cv_rmse_scores.append(float(np.sqrt(mean_squared_error(y_tr[val_idx].ravel(), pred.ravel()))))
    log.info("RF CV RMSE: %.3f ± %.3f", np.mean(cv_rmse_scores), np.std(cv_rmse_scores))

    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    metrics = evaluate(y_te, y_pred, "RandomForest")

    # Individual tree predictions for confidence intervals
    all_preds = np.stack(
        [est.predict(X_te) for est in rf.estimators_], axis=0
    )  # (n_estimators, n_test, 3)
    pred_std = all_preds.std(axis=0)

    return rf, metrics, y_pred, pred_std


# ── Ridge Regression ──────────────────────────────────────────────────────────

def train_ridge(splits: dict) -> tuple:
    log.info("Training Ridge Regression...")
    X_tr, y_tr = splits["X_flat_train"], splits["y_train"]
    X_te, y_te = splits["X_flat_test"],  splits["y_test"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  MultiOutputRegressor(Ridge(alpha=1.0, random_state=RANDOM_SEED))),
    ])

    # CV
    kf = KFold(n_splits=5, shuffle=False)
    cv_rmse_scores: list[float] = []
    for tr_idx, val_idx in kf.split(X_tr):
        pipe.fit(X_tr[tr_idx], y_tr[tr_idx])
        pred = pipe.predict(X_tr[val_idx])
        cv_rmse_scores.append(float(np.sqrt(mean_squared_error(y_tr[val_idx].ravel(), pred.ravel()))))
    log.info("Ridge CV RMSE: %.3f ± %.3f", np.mean(cv_rmse_scores), np.std(cv_rmse_scores))

    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    metrics = evaluate(y_te, y_pred, "Ridge")

    pred_std = np.full_like(y_pred, fill_value=np.std(y_te - y_pred))
    return pipe, metrics, y_pred, pred_std


# ── LSTM ──────────────────────────────────────────────────────────────────────

def build_lstm(seq_len: int, n_features: int, n_targets: int = 3) -> Sequential:
    model = Sequential([
        LSTM(128, input_shape=(seq_len, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(n_targets),
    ], name="aqi_lstm")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_lstm(splits: dict) -> tuple:
    log.info("Training LSTM...")
    X_tr, y_tr = splits["X_seq_train"], splits["y_train"]
    X_va, y_va = splits["X_seq_val"],   splits["y_val"]
    X_te, y_te = splits["X_seq_test"],  splits["y_test"]

    seq_len, n_feat = X_tr.shape[1], X_tr.shape[2]
    model = build_lstm(seq_len, n_feat, n_targets=len(TARGET_COLS))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=0,
    )

    y_pred = model.predict(X_te, verbose=0)
    metrics = evaluate(y_te, y_pred, "LSTM")

    # MC Dropout for uncertainty estimation
    mc_preds = np.stack(
        [model(X_te, training=True).numpy() for _ in range(30)], axis=0
    )
    pred_std = mc_preds.std(axis=0)

    return model, metrics, y_pred, pred_std


# ── SHAP explanation ──────────────────────────────────────────────────────────

def explain_with_shap(
    best_model,
    model_name: str,
    X_flat_test: np.ndarray,
    feat_names: list[str],
    output_path: Path,
) -> None:
    log.info("Computing SHAP values for %s...", model_name)

    if model_name == "RandomForest":
        # Use mean of estimators' feature importances (fast, reliable)
        importances = np.mean([
            est.feature_importances_
            for est in best_model.estimators_
        ], axis=0)
        shap_values = pd.Series(importances, index=feat_names).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        top = shap_values.head(20)
        ax.barh(top.index[::-1], top.values[::-1], color="#2196F3")
        ax.set_xlabel("Mean Feature Importance")
        ax.set_title(f"Feature Importance – {model_name}")
        plt.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        log.info("SHAP chart saved to %s", output_path)

    elif model_name == "Ridge":
        # Use absolute coefficient values
        coefs = np.abs(
            np.mean([est.coef_ for est in best_model.named_steps["ridge"].estimators_], axis=0)
        )
        shap_values = pd.Series(coefs, index=feat_names).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        top = shap_values.head(20)
        ax.barh(top.index[::-1], top.values[::-1], color="#4CAF50")
        ax.set_xlabel("Mean |Coefficient|")
        ax.set_title(f"Feature Importance – {model_name}")
        plt.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        log.info("Ridge importance chart saved to %s", output_path)

    elif model_name == "LSTM":
        # KernelExplainer on a subset (100 samples for speed)
        background = X_flat_test[:50]
        explainer = shap.KernelExplainer(
            lambda x: best_model.predict(
                x[:, np.newaxis, :].repeat(MODEL_LOOKBACK, axis=1), verbose=0
            ).mean(axis=1, keepdims=True),
            background,
        )
        sample = X_flat_test[:100]
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = np.mean([np.abs(s) for s in sv], axis=0)
        mean_abs = np.abs(sv).mean(axis=0)
        shap_series = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        top = shap_series.head(20)
        ax.barh(top.index[::-1], top.values[::-1], color="#FF5722")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"SHAP Feature Importance – {model_name}")
        plt.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        log.info("SHAP chart saved to %s", output_path)


# ── Model registry ────────────────────────────────────────────────────────────

def register_model_hopsworks(model, model_name: str, metrics: dict) -> None:
    try:
        import hopsworks  # type: ignore
        from src.config import HOPSWORKS_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST
        project = hopsworks.login(host=HOPSWORKS_HOST, api_key_value=HOPSWORKS_KEY, project=HOPSWORKS_PROJECT)
        mr = project.get_model_registry()

        save_dir = MODELS_DIR / f"hw_{model_name.lower()}"
        save_dir.mkdir(exist_ok=True)

        if model_name == "LSTM":
            model.save(str(save_dir / "model.keras"))
        else:
            joblib.dump(model, save_dir / "model.pkl")

        hw_model = mr.python.create_model(
            name=f"aqi_forecaster_{model_name.lower()}",
            metrics=metrics,
            description=f"3-day AQI forecaster: {model_name}",
        )
        hw_model.save(str(save_dir))
        log.info("Model registered to Hopsworks as aqi_forecaster_%s", model_name.lower())
    except Exception as exc:
        log.warning("Hopsworks model registration failed (%s) – falling back to MLflow", exc)
        register_model_mlflow(model, model_name, metrics)


def register_model_mlflow(model, model_name: str, metrics: dict) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model_type": model_name, "random_seed": RANDOM_SEED})
        mlflow.log_metrics(metrics)

        if model_name == "LSTM":
            save_path = str(MODELS_DIR / f"mlflow_{model_name.lower()}.keras")
            model.save(save_path)
            mlflow.log_artifact(save_path, artifact_path="model")
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"aqi_forecaster_{model_name.lower()}",
            )

        log.info(
            "Model registered to MLflow: %s  RMSE=%.3f",
            model_name, metrics["rmse"],
        )


def save_model_locally(model, model_name: str, metrics: dict, pred_std: np.ndarray) -> Path:
    """Always save a local copy for the dashboard to load."""
    artifact = {
        "model_name": model_name,
        "metrics": metrics,
        "pred_std_mean": float(pred_std.mean()),
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "lookback": MODEL_LOOKBACK,
    }
    if model_name == "LSTM":
        model_path = MODELS_DIR / "best_model.keras"
        model.save(str(model_path))
        artifact["model_path"] = str(model_path)
        artifact["model_type"] = "LSTM"
    else:
        model_path = MODELS_DIR / "best_model.pkl"
        joblib.dump(model, model_path)
        artifact["model_path"] = str(model_path)
        artifact["model_type"] = "sklearn"

    joblib.dump(artifact, MODELS_DIR / "best_model_info.pkl")
    log.info("Local model artifacts saved under %s", MODELS_DIR)
    return model_path


# ── Main ──────────────────────────────────────────────────────────────────────

def run_training() -> None:
    log.info("=== Training pipeline started ===")

    # ── Load & prepare ──────────────────────────────────────────────────────
    df = load_data()
    X_flat, X_seq, y, feat_names = prepare_dataset(df)
    splits = temporal_train_val_test_split(X_flat, X_seq, y)

    results: dict[str, dict] = {}
    trained: dict = {}
    preds:   dict = {}
    stds:    dict = {}

    # ── Train models ────────────────────────────────────────────────────────
    rf_model, rf_metrics, rf_pred, rf_std = train_random_forest(splits, feat_names)
    results["RandomForest"] = rf_metrics
    trained["RandomForest"] = rf_model
    preds["RandomForest"]   = rf_pred
    stds["RandomForest"]    = rf_std

    ridge_model, ridge_metrics, ridge_pred, ridge_std = train_ridge(splits)
    results["Ridge"] = ridge_metrics
    trained["Ridge"] = ridge_model
    preds["Ridge"]   = ridge_pred
    stds["Ridge"]    = ridge_std

    lstm_model, lstm_metrics, lstm_pred, lstm_std = train_lstm(splits)
    results["LSTM"] = lstm_metrics
    trained["LSTM"] = lstm_model
    preds["LSTM"]   = lstm_pred
    stds["LSTM"]    = lstm_std

    # ── Comparison table ────────────────────────────────────────────────────
    comparison = pd.DataFrame(results).T.rename_axis("Model")
    comparison = comparison.sort_values("rmse")
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON (test set)")
    print("=" * 55)
    print(comparison.to_string(float_format="%.4f"))
    print("=" * 55 + "\n")

    best_name = comparison.index[0]
    best_model = trained[best_name]
    best_metrics = results[best_name]
    best_std = stds[best_name]
    log.info("Best model: %s (RMSE=%.4f)", best_name, best_metrics["rmse"])

    # ── SHAP explanation ────────────────────────────────────────────────────
    shap_path = MODELS_DIR / "shap_importance.png"
    explain_with_shap(
        best_model, best_name,
        splits["X_flat_test"], feat_names, shap_path,
    )

    # ── Register ────────────────────────────────────────────────────────────
    save_model_locally(best_model, best_name, best_metrics, best_std)
    register_model_hopsworks(best_model, best_name, best_metrics)

    log.info("=== Training pipeline complete ===")


if __name__ == "__main__":
    try:
        run_training()
    except Exception:
        log.exception("Training pipeline failed")
        sys.exit(1)
