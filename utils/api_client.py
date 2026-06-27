"""
API clients for AQICN and OpenWeatherMap with retry logic.

AQICN  – real-time AQI (primary)
OpenWeather – historical + real-time fallback
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import requests
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)

# ── AQI calculation helpers ───────────────────────────────────────────────────

_PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
]
_PM10_BREAKPOINTS = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300),
    (425, 504, 301, 400), (505, 604, 401, 500),
]
_O3_BREAKPOINTS_PPM = [
    (0.000, 0.054, 0, 50), (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150), (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
]
_NO2_BREAKPOINTS_PPB = [
    (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
    (361, 649, 151, 200), (650, 1249, 201, 300),
    (1250, 1649, 301, 400), (1650, 2049, 401, 500),
]
_CO_BREAKPOINTS_PPM = [
    (0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500),
]
_SO2_BREAKPOINTS_PPB = [
    (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
    (186, 304, 151, 200), (305, 604, 201, 300),
    (605, 804, 301, 400), (805, 1004, 401, 500),
]


def _linear_interp(c: float, bp: list[tuple]) -> float:
    for c_lo, c_hi, i_lo, i_hi in bp:
        if c_lo <= c <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (c - c_lo) + i_lo
    return 500.0  # clamp to max


def calculate_aqi_from_pollutants(
    pm25: float | None = None,
    pm10: float | None = None,
    o3_ugm3: float | None = None,
    no2_ugm3: float | None = None,
    co_ugm3: float | None = None,
    so2_ugm3: float | None = None,
) -> float:
    """Return US EPA AQI from pollutant concentrations (μg/m³)."""
    candidates: list[float] = []

    if pm25 is not None and pm25 >= 0:
        candidates.append(_linear_interp(pm25, _PM25_BREAKPOINTS))
    if pm10 is not None and pm10 >= 0:
        candidates.append(_linear_interp(pm10, _PM10_BREAKPOINTS))
    if o3_ugm3 is not None and o3_ugm3 >= 0:
        # μg/m³ → ppm  (MW O3 = 48 g/mol, at STP 1 ppm ≈ 2.0 μg/m³)
        o3_ppm = o3_ugm3 / 2000.0
        candidates.append(_linear_interp(o3_ppm, _O3_BREAKPOINTS_PPM))
    if no2_ugm3 is not None and no2_ugm3 >= 0:
        # μg/m³ → ppb  (MW NO2 = 46 g/mol, at STP 1 ppb ≈ 1.88 μg/m³)
        no2_ppb = no2_ugm3 / 1.88
        candidates.append(_linear_interp(no2_ppb, _NO2_BREAKPOINTS_PPB))
    if co_ugm3 is not None and co_ugm3 >= 0:
        # μg/m³ → ppm  (MW CO = 28 g/mol, at STP 1 ppm ≈ 1145 μg/m³)
        co_ppm = co_ugm3 / 1145.0
        candidates.append(_linear_interp(co_ppm, _CO_BREAKPOINTS_PPM))
    if so2_ugm3 is not None and so2_ugm3 >= 0:
        # μg/m³ → ppb  (MW SO2 = 64 g/mol, at STP 1 ppb ≈ 2.62 μg/m³)
        so2_ppb = so2_ugm3 / 2.62
        candidates.append(_linear_interp(so2_ppb, _SO2_BREAKPOINTS_PPB))

    return float(max(candidates)) if candidates else float("nan")


# ── Retry decorator ───────────────────────────────────────────────────────────

def _make_retry():
    return retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((requests.RequestException, ValueError)),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )


# ── AQICN Client ──────────────────────────────────────────────────────────────

class AQICNClient:
    BASE_URL = "https://api.waqi.info"

    def __init__(self, api_key: str, timeout: int = 15) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    @_make_retry()
    def _get(self, endpoint: str) -> dict[str, Any]:
        url = f"{self.BASE_URL}/{endpoint}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            raise ValueError(f"AQICN error: {data.get('data', 'unknown')}")
        return data["data"]

    def fetch_current(self, station: str) -> dict[str, Any]:
        """Return parsed current AQI reading for a station or geo query."""
        raw = self._get(f"feed/{station}/?token={self.api_key}")
        return self._parse_current(raw)

    def _parse_current(self, raw: dict) -> dict[str, Any]:
        iaqi = raw.get("iaqi", {})
        ts_str = raw.get("time", {}).get("s", "")
        tz_str = raw.get("time", {}).get("tz", "+05:00")

        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc).replace(tzinfo=None)

        return {
            "timestamp_utc": ts,
            "city": raw.get("city", {}).get("name", "Unknown"),
            "aqi": float(raw.get("aqi", float("nan"))),
            "pm25": float(iaqi.get("pm25", {}).get("v", float("nan"))),
            "pm10": float(iaqi.get("pm10", {}).get("v", float("nan"))),
            "no2":  float(iaqi.get("no2",  {}).get("v", float("nan"))),
            "o3":   float(iaqi.get("o3",   {}).get("v", float("nan"))),
            "co":   float(iaqi.get("co",   {}).get("v", float("nan"))),
            "so2":  float(iaqi.get("so2",  {}).get("v", float("nan"))),
        }


# ── OpenWeather Client ────────────────────────────────────────────────────────

class OpenWeatherClient:
    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: str, timeout: int = 15) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    @_make_retry()
    def _get(self, endpoint: str, params: dict) -> dict[str, Any]:
        params["appid"] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_current(self, lat: float, lon: float) -> dict[str, Any]:
        """Real-time air pollution reading."""
        raw = self._get("air_pollution", {"lat": lat, "lon": lon})
        items = raw.get("list", [])
        if not items:
            raise ValueError("Empty response from OpenWeather current API")
        return self._parse_item(items[0])

    def fetch_history(
        self,
        lat: float,
        lon: float,
        start_unix: int,
        end_unix: int,
    ) -> list[dict[str, Any]]:
        """Historical hourly air pollution (free tier, back to Nov 2020)."""
        raw = self._get(
            "air_pollution/history",
            {"lat": lat, "lon": lon, "start": start_unix, "end": end_unix},
        )
        items = raw.get("list", [])
        if not items:
            log.warning("OpenWeather returned 0 rows for [%s, %s]", start_unix, end_unix)
            return []
        return [self._parse_item(it) for it in items]

    def _parse_item(self, item: dict) -> dict[str, Any]:
        comp = item.get("components", {})
        dt_unix = item.get("dt", 0)
        ts = datetime.utcfromtimestamp(dt_unix)

        pm25_val = float(comp.get("pm2_5", float("nan")))
        pm10_val = float(comp.get("pm10",  float("nan")))
        no2_val  = float(comp.get("no2",   float("nan")))
        o3_val   = float(comp.get("o3",    float("nan")))
        co_val   = float(comp.get("co",    float("nan")))
        so2_val  = float(comp.get("so2",   float("nan")))

        aqi = calculate_aqi_from_pollutants(
            pm25=pm25_val, pm10=pm10_val,
            o3_ugm3=o3_val, no2_ugm3=no2_val,
            co_ugm3=co_val, so2_ugm3=so2_val,
        )

        return {
            "timestamp_utc": ts,
            "city": "Faisalabad",
            "aqi": aqi,
            "pm25": pm25_val,
            "pm10": pm10_val,
            "no2":  no2_val,
            "o3":   o3_val,
            "co":   co_val,
            "so2":  so2_val,
        }


def fetch_current_with_fallback(
    aqicn_client: AQICNClient,
    ow_client: OpenWeatherClient,
    station: str,
    lat: float,
    lon: float,
) -> dict[str, Any]:
    """Try AQICN first; fall back to OpenWeather on any error."""
    try:
        return aqicn_client.fetch_current(station)
    except Exception as exc:
        log.warning("AQICN fetch failed (%s), falling back to OpenWeather", exc)
        return ow_client.fetch_current(lat, lon)
