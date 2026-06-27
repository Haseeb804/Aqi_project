"""
Feature engineering transforms applied to both live and historical AQI data.

All functions accept and return pandas DataFrames, making them pipeline-safe.
The DataFrame must have columns: timestamp_utc, city, aqi, pm25, pm10, no2, o3, co, so2
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

POLLUTANTS = ["pm25", "pm10", "no2", "o3", "co", "so2"]
WINDOWS = [3, 6, 12]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, month, and season columns."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp_utc"])
    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek   # 0 = Monday
    df["month"]       = ts.dt.month

    # Season: 0=Winter (Dec-Feb), 1=Spring (Mar-May), 2=Summer (Jun-Aug), 3=Autumn (Sep-Nov)
    month_to_season = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                       6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df["season"] = df["month"].map(month_to_season)
    return df


def add_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean columns for each pollutant and AQI.
    Assumes the DataFrame is already sorted by timestamp_utc within each city group.
    """
    df = df.copy()
    df = df.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)

    for col in POLLUTANTS + ["aqi"]:
        if col not in df.columns:
            log.warning("Column %s missing, skipping rolling averages for it", col)
            continue
        for w in WINDOWS:
            col_name = f"{col}_{w}h_avg"
            df[col_name] = (
                df.groupby("city")[col]
                .transform(lambda s, w=w: s.rolling(window=w, min_periods=1).mean())
            )
    return df


def add_aqi_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add AQI_change_rate: percentage change of AQI relative to the previous row.
    Grouped by city so rate doesn't bleed across different cities.
    """
    df = df.copy()
    df = df.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)
    df["aqi_change_rate"] = (
        df.groupby("city")["aqi"]
        .transform(lambda s: s.pct_change() * 100)
        .fillna(0.0)
    )
    return df


def add_forecast_targets(df: pd.DataFrame, forecast_days: int = 3) -> pd.DataFrame:
    """
    Add target columns: AQI shifted N*24 hours into the future.
    Rows where targets would be NaN (end of the series) are dropped.
    """
    df = df.copy()
    df = df.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)
    for day in range(1, forecast_days + 1):
        shift = day * 24
        df[f"target_day{day}"] = (
            df.groupby("city")["aqi"]
            .transform(lambda s, sh=shift: s.shift(-sh))
        )
    # Drop rows where any target is NaN
    target_cols = [f"target_day{d}" for d in range(1, forecast_days + 1)]
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    return df


def engineer_features(
    df: pd.DataFrame,
    add_targets: bool = False,
    forecast_days: int = 3,
) -> pd.DataFrame:
    """
    Apply the full feature engineering chain to a raw DataFrame.

    Parameters
    ----------
    df           : Raw DataFrame with at minimum timestamp_utc, city, aqi + pollutants.
    add_targets  : If True, compute forecast target columns (used in training).
    forecast_days: How many days ahead to forecast.
    """
    if df.empty:
        log.warning("engineer_features received an empty DataFrame")
        return df

    df = add_time_features(df)
    df = add_rolling_averages(df)
    df = add_aqi_change_rate(df)

    if add_targets:
        df = add_forecast_targets(df, forecast_days=forecast_days)

    # Fill any remaining NaNs in feature columns with column median
    from src.config import FEATURE_COLS  # avoid circular import at module level
    for col in FEATURE_COLS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    log.info(
        "Feature engineering complete: %d rows, %d columns",
        len(df), len(df.columns),
    )
    return df


def merge_with_history(
    new_row: dict,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Append a new raw reading to the tail of a history DataFrame, then compute
    rolling features only for the new row. Returns the combined DataFrame.

    Used in the live hourly pipeline so rolling averages span real history.
    """
    row_df = pd.DataFrame([new_row])
    combined = pd.concat([history, row_df], ignore_index=True)
    combined = combined.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)
    combined = add_time_features(combined)
    combined = add_rolling_averages(combined)
    combined = add_aqi_change_rate(combined)
    return combined
