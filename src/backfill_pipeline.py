"""
Historical AQI backfill pipeline.

Fetches up to 2 years of hourly data from the OpenWeather Air Pollution History
API, applies the same feature engineering as the live pipeline, and stores the
result in the feature store (with tag="backfill") or as a local Parquet file.

Usage
-----
    python -m src.backfill_pipeline \\
        --city Faisalabad \\
        --start 2022-01-01 \\
        --end   2024-01-01

    # Override output path (skip feature store):
    python -m src.backfill_pipeline --start 2022-01-01 --end 2024-01-01 \\
        --output data/backfill.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import (
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
from utils.api_client import OpenWeatherClient
from utils.feature_engineering import engineer_features
from utils.store_client import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "backfill.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# OpenWeather free tier: max ~30 days per request to stay safe
CHUNK_DAYS = 25
MAX_RETRIES = 6
BASE_BACKOFF = 5  # seconds


def _to_unix(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def _fetch_chunk_with_backoff(
    client: OpenWeatherClient,
    lat: float,
    lon: float,
    start_unix: int,
    end_unix: int,
) -> list[dict]:
    """Fetch one time-chunk with exponential backoff on failure."""
    delay = BASE_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            records = client.fetch_history(lat, lon, start_unix, end_unix)
            return records
        except Exception as exc:
            if attempt == MAX_RETRIES:
                log.error("All %d retries exhausted for chunk [%s-%s]", MAX_RETRIES, start_unix, end_unix)
                raise
            log.warning(
                "Attempt %d/%d failed (%s). Backing off %.0fs...",
                attempt, MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 120)  # cap at 2 minutes
    return []


def fetch_history_in_chunks(
    client: OpenWeatherClient,
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    city: str,
) -> pd.DataFrame:
    """Iterate over the date range in CHUNK_DAYS chunks and collect all rows."""
    all_records: list[dict] = []
    chunk_start = start
    total_chunks = ((end - start).days // CHUNK_DAYS) + 1

    log.info(
        "Fetching %d days in ~%d chunks of %d days each",
        (end - start).days,
        total_chunks,
        CHUNK_DAYS,
    )

    chunk_idx = 0
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        chunk_idx += 1

        log.info(
            "[%d/%d] Fetching %s → %s",
            chunk_idx, total_chunks,
            chunk_start.date(), chunk_end.date(),
        )

        records = _fetch_chunk_with_backoff(
            client,
            lat,
            lon,
            _to_unix(chunk_start),
            _to_unix(chunk_end),
        )
        log.info("  → %d records returned", len(records))
        all_records.extend(records)

        chunk_start = chunk_end
        # Polite sleep to respect rate limits
        time.sleep(0.5)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["city"] = city
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.drop_duplicates(subset=["city", "timestamp_utc"])
    df = df.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame, output_path: str) -> None:
    print("\n" + "=" * 60)
    print("  BACKFILL SUMMARY")
    print("=" * 60)
    print(f"  Total rows added : {len(df):,}")
    if not df.empty:
        ts = df["timestamp_utc"]
        print(f"  Date coverage    : {ts.min()} → {ts.max()}")
        print(f"  Unique cities    : {df['city'].nunique()}")
        print(f"\n  Null counts per column:")
        nulls = df.isnull().sum()
        for col, n in nulls[nulls > 0].items():
            print(f"    {col:<25} {n:>6} ({n/len(df)*100:.1f}%)")
        if nulls.sum() == 0:
            print("    (none)")
    print(f"\n  Saved to         : {output_path}")
    print("=" * 60 + "\n")


def run_backfill(
    city: str,
    start: datetime,
    end: datetime,
    lat: float,
    lon: float,
    output_path: str | None = None,
) -> pd.DataFrame:
    log.info("=== Backfill started: %s  %s → %s ===", city, start.date(), end.date())

    client = OpenWeatherClient(api_key=OPENWEATHER_KEY)

    # ── Fetch raw data ─────────────────────────────────────────────────────
    df_raw = fetch_history_in_chunks(client, lat, lon, start, end, city)

    if df_raw.empty:
        log.error("No data fetched. Check API key and date range.")
        sys.exit(1)

    log.info("Total raw records: %d", len(df_raw))

    # ── Feature engineering ────────────────────────────────────────────────
    log.info("Running feature engineering...")
    df_feat = engineer_features(df_raw, add_targets=False)

    # ── Save ───────────────────────────────────────────────────────────────
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_parquet(output_path, index=False)
        log.info("Saved to %s", output_path)
        print_summary(df_feat, output_path)
    else:
        store = FeatureStore(
            hopsworks_key=HOPSWORKS_KEY,
            hopsworks_project=HOPSWORKS_PROJECT,
            hopsworks_host=HOPSWORKS_HOST,
            fg_name=FEATURE_GROUP_NAME,
            fg_version=FEATURE_GROUP_VERSION,
        )
        store.upsert(df_feat, tag="backfill")
        saved_path = str(Path(__file__).resolve().parent.parent / "data" / "aqi_features.parquet")
        print_summary(df_feat, saved_path)

    log.info("=== Backfill complete ===")
    return df_feat


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical AQI data for model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--city",
        default=DEFAULT_CITY,
        help="City name (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today UTC)",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=CITY_LAT,
        help="Latitude (default: %(default)s)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=CITY_LON,
        help="Longitude (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Local Parquet file path. If omitted, writes to feature store.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt   = datetime.strptime(args.end,   "%Y-%m-%d")
    except ValueError as exc:
        print(f"ERROR: Invalid date format – {exc}")
        sys.exit(1)

    if start_dt >= end_dt:
        print("ERROR: --start must be before --end")
        sys.exit(1)

    run_backfill(
        city=args.city,
        start=start_dt,
        end=end_dt,
        lat=args.lat,
        lon=args.lon,
        output_path=args.output,
    )
