"""
Feature store abstraction.

Tries Hopsworks first; falls back to local Parquet files transparently.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "aqi_features.parquet"


# ── Parquet (local) store ─────────────────────────────────────────────────────

class LocalParquetStore:
    def __init__(self, path: Path = _PARQUET_PATH) -> None:
        self.path = path

    def _read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.path)

    def upsert(self, df: pd.DataFrame, tag: str | None = None) -> None:
        """Append rows that don't already exist (keyed by city + timestamp_utc)."""
        df = df.copy()
        if tag:
            df["tag"] = tag

        existing = self._read()
        if existing.empty:
            df.to_parquet(self.path, index=False)
            log.info("Parquet: created new file with %d rows", len(df))
            return

        # Deduplicate on city + timestamp_utc
        key = ["city", "timestamp_utc"]
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        existing["timestamp_utc"] = pd.to_datetime(existing["timestamp_utc"])
        merged = pd.concat([existing, df]).drop_duplicates(subset=key, keep="last")
        merged = merged.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)
        merged.to_parquet(self.path, index=False)
        log.info("Parquet: upserted to %d total rows", len(merged))

    def read(
        self,
        city: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        df = self._read()
        if df.empty:
            return df
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        if city:
            df = df[df["city"] == city]
        if start:
            df = df[df["timestamp_utc"] >= pd.Timestamp(start)]
        if end:
            df = df[df["timestamp_utc"] <= pd.Timestamp(end)]
        return df.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)

    def read_latest(self, city: str, n_hours: int = 24) -> pd.DataFrame:
        df = self._read()
        if df.empty:
            return df
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        df = df[df["city"] == city]
        if df.empty:
            return df
        cutoff = df["timestamp_utc"].max() - timedelta(hours=n_hours)
        return df[df["timestamp_utc"] >= cutoff].sort_values("timestamp_utc").reset_index(drop=True)


# ── Hopsworks store ───────────────────────────────────────────────────────────

class HopsworksStore:
    def __init__(
        self,
        api_key: str,
        project_name: str,
        host: str,
        fg_name: str,
        fg_version: int,
    ) -> None:
        self.api_key = api_key
        self.project_name = project_name
        self.host = host
        self.fg_name = fg_name
        self.fg_version = fg_version
        self._project: Any = None
        self._fs: Any = None
        self._fg: Any = None

    def _connect(self) -> None:
        if self._project is not None:
            return
        import hopsworks  # type: ignore
        self._project = hopsworks.login(
            host=self.host,
            api_key_value=self.api_key,
            project=self.project_name,
        )
        self._fs = self._project.get_feature_store()
        log.info("Connected to Hopsworks project: %s", self.project_name)

    def _get_fg(self):
        self._connect()
        if self._fg is None:
            self._fg = self._fs.get_or_create_feature_group(
                name=self.fg_name,
                version=self.fg_version,
                primary_key=["city", "timestamp_utc"],
                event_time="timestamp_utc",
                description="Hourly AQI features for Faisalabad, Pakistan",
                online_enabled=False,
            )
        return self._fg

    def upsert(self, df: pd.DataFrame, tag: str | None = None) -> None:
        df = df.copy()
        if tag:
            df["tag"] = tag
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        fg = self._get_fg()
        fg.insert(df, write_options={"wait_for_job": True})
        log.info("Hopsworks: inserted %d rows into %s v%s", len(df), self.fg_name, self.fg_version)

    def read(
        self,
        city: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        fg = self._get_fg()
        df = fg.read()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        if city:
            df = df[df["city"] == city]
        if start:
            df = df[df["timestamp_utc"] >= pd.Timestamp(start)]
        if end:
            df = df[df["timestamp_utc"] <= pd.Timestamp(end)]
        return df.sort_values(["city", "timestamp_utc"]).reset_index(drop=True)

    def read_latest(self, city: str, n_hours: int = 24) -> pd.DataFrame:
        df = self.read(city=city)
        if df.empty:
            return df
        cutoff = df["timestamp_utc"].max() - timedelta(hours=n_hours)
        return df[df["timestamp_utc"] >= cutoff].sort_values("timestamp_utc").reset_index(drop=True)


# ── Auto-selecting facade ─────────────────────────────────────────────────────

class FeatureStore:
    """
    Wraps HopsworksStore with automatic fallback to LocalParquetStore.
    Call `store.upsert(df)` and `store.read(...)` without worrying which backend is live.
    """

    def __init__(
        self,
        hopsworks_key: str = "",
        hopsworks_project: str = "",
        hopsworks_host: str = "c.app.hopsworks.ai",
        fg_name: str = "aqi_features",
        fg_version: int = 1,
    ) -> None:
        self._local = LocalParquetStore()
        self._hw: HopsworksStore | None = None
        self._use_hw = False

        if hopsworks_key and hopsworks_project:
            try:
                hw = HopsworksStore(
                    api_key=hopsworks_key,
                    project_name=hopsworks_project,
                    host=hopsworks_host,
                    fg_name=fg_name,
                    fg_version=fg_version,
                )
                hw._connect()  # test connection eagerly
                self._hw = hw
                self._use_hw = True
                log.info("Feature store: using Hopsworks")
            except Exception as exc:
                log.warning("Hopsworks unavailable (%s) – falling back to local Parquet", exc)

        if not self._use_hw:
            log.info("Feature store: using local Parquet at %s", _PARQUET_PATH)

    def upsert(self, df: pd.DataFrame, tag: str | None = None) -> None:
        if self._use_hw and self._hw:
            try:
                self._hw.upsert(df, tag=tag)
                self._local.upsert(df, tag=tag)  # keep local copy in sync
                return
            except Exception as exc:
                log.error("Hopsworks upsert failed (%s), writing to local only", exc)
        self._local.upsert(df, tag=tag)

    def read(
        self,
        city: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        if self._use_hw and self._hw:
            try:
                return self._hw.read(city=city, start=start, end=end)
            except Exception as exc:
                log.error("Hopsworks read failed (%s), reading from local", exc)
        return self._local.read(city=city, start=start, end=end)

    def read_latest(self, city: str, n_hours: int = 24) -> pd.DataFrame:
        if self._use_hw and self._hw:
            try:
                return self._hw.read_latest(city=city, n_hours=n_hours)
            except Exception as exc:
                log.error("Hopsworks read_latest failed (%s), reading from local", exc)
        return self._local.read_latest(city=city, n_hours=n_hours)
