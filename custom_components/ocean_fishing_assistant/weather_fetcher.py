"""Weather data fetcher - Open-Meteo REST-focused.

Features:
- Calls Open-Meteo REST endpoints (OM_BASE) directly for current & forecast.
- Uses the Open-Meteo Marine endpoint (OM_MARINE_BASE) for marine/wave fields.
- Global caching (shared across instances) for current & forecast.
- Robust normalization for many shapes (dicts, lists, objects).
- Unit coercion via unit_helpers (canonical m/s internal, output in km/h/mph/m/s).
- Forecast aggregation into flexible "periods" (default: 4 equal periods per day) or custom periods.
- Strict failure semantics: fails loudly when current/forecast critical fields are missing.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import aiohttp

from homeassistant.util import dt as dt_util
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import OM_BASE, OM_MARINE_BASE
from . import unit_helpers

_LOGGER = logging.getLogger(__name__)

# Global caches (shared across instances)
_GLOBAL_CACHE: Dict[str, Dict[str, Any]] = {}

# Defaults for diagnostics only (we fail loudly instead of silently using these)
DEFAULT_WEATHER_VALUES = {
    "temperature": 15.0,
    "wind_speed": 10.0,  # km/h (diagnostic)
    "wind_gust": 15.0,  # km/h (diagnostic)
    "cloud_cover": 50,  # percentage
    "precipitation_probability": 0,  # percentage
    "pressure": 1013.0,  # hPa
}

# hourly params to request from Open-Meteo
OM_PARAMS_HOURLY = ",".join(
    [
        "temperature_2m",
        "wind_speed_10m",
        "windgusts_10m",
        "cloudcover",
        "precipitation_probability",
        "pressure_msl",
    ]
)


def _safe_float(v: Any, default: float) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, bool):
            return float(v)
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int) -> int:
    try:
        if v is None:
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


class WeatherFetcher:
    """
    Fetch current weather and forecast via direct Open-Meteo REST calls.

    Aggregation periods:
      [
        {"name": "dawn", "start_hour": 4, "end_hour": 6},
        {"name": "day",  "start_hour": 6, "end_hour": 18},
        ...
      ]
    If None, a default 4-period split is used (00-06, 06-12, 12-18, 18-24).
    """

    def __init__(self, hass, latitude: float, longitude: float, speed_unit: Optional[str] = None) -> None:
        self.hass = hass
        self.latitude = round(float(latitude), 4)
        self.longitude = round(float(longitude), 4)

        # Determine default output speed unit: derive from HA units if possible
        if speed_unit:
            self.speed_unit = str(speed_unit).lower()
        else:
            # Try to detect HA system units (fallback to km/h)
            try:
                units_obj = getattr(self.hass.config, "units", None)
                if units_obj is not None and getattr(units_obj, "is_metric", True) is False:
                    self.speed_unit = "mph"
                else:
                    self.speed_unit = "km/h"
            except Exception:
                self.speed_unit = "km/h"

        self._cache_key = f"{self.latitude}_{self.longitude}_om"
        self._cache_duration = timedelta(minutes=30)

    # -----------------------
    # Unit helpers
    # -----------------------
    def _incoming_wind_to_m_s(self, value: Any, unit_hint: Optional[str] = None) -> Optional[float]:
        """Convert incoming wind value to m/s (internal canonical). If unit_hint provided, use it; otherwise assume m/s."""
        if value is None:
            return None
        try:
            v = float(value)
        except Exception:
            return None

        if unit_hint:
            u = str(unit_hint).strip().lower()
            if u in ("km/h", "kph", "kmh", "km per h"):
                return unit_helpers.kmh_to_m_s(v)
            if u in ("mph", "mi/h", "miles/h"):
                return unit_helpers.mph_to_m_s(v)
            # if hint says m/s:
            if u in ("m/s", "mps", "m s-1"):
                return v
        # default: assume m/s (Open-Meteo default)
        return v

    def _m_s_to_output(self, v: Optional[float]) -> Optional[float]:
        """Convert m/s value to configured output unit (km/h, mph, or m/s)."""
        if v is None:
            return None
        try:
            vf = float(v)
        except Exception:
            return None
        su = (self.speed_unit or "km/h").lower()
        if su in ("km/h", "kph", "kmh"):
            return unit_helpers.m_s_to_kmh(vf)
        if su in ("mph",):
            return unit_helpers.m_s_to_mph(vf)
        # default m/s
        return vf

    def _to_output_wind(self, value: Any, incoming_unit_hint: Optional[str] = None) -> Optional[float]:
        """Convenience: convert incoming value (with optional hint) to output unit."""
        m_s = self._incoming_wind_to_m_s(value, incoming_unit_hint)
        return self._m_s_to_output(m_s)

    # -----------------------
    # Public: current weather
    # -----------------------
    async def get_weather_data(self) -> Dict[str, Any]:
        """Get current normalized weather data via direct Open-Meteo REST call (no client fallback)."""
        now = dt_util.now()
        cache_entry = _GLOBAL_CACHE.get(self._cache_key)
        if cache_entry:
            cached_time = cache_entry.get("time")
            if isinstance(cached_time, datetime) and (now - cached_time) < self._cache_duration:
                _LOGGER.debug("Using cached weather data for %s", self._cache_key)
                return cache_entry["data"]

        result = await self.fetch_open_meteo_current_direct()
        if not result:
            _LOGGER.error("Unable to fetch current weather data from Open-Meteo for %s; aborting", self._cache_key)
            raise RuntimeError("Unable to fetch current weather data from Open-Meteo")

        # Normalize: prefer explicit current_weather block if present
        mapped = None
        if isinstance(result, dict) and "current_weather" in result and isinstance(result["current_weather"], dict):
            mapped = self._map_to_current_shape(result["current_weather"])
        else:
            mapped = self._extract_current_from_rest_result(result)

        if not mapped or mapped.get("temperature") is None or mapped.get("wind_speed") is None:
            _LOGGER.error("Open-Meteo returned incomplete current-weather data: %s", result)
            raise RuntimeError("Incomplete weather data from Open-Meteo")

        _LOGGER.info("Fetched current weather from Open-Meteo REST for %s", self._cache_key)
        _GLOBAL_CACHE[self._cache_key] = {"data": mapped, "time": now}
        return mapped

    async def fetch_open_meteo_current_direct(self) -> Optional[Dict[str, Any]]:
        """Call OM_BASE with current_weather=true and hourly as fallback."""
        try:
            session: aiohttp.ClientSession = async_get_clientsession(self.hass)
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "timezone": "UTC",
                "current_weather": "true",
                "hourly": OM_PARAMS_HOURLY,
                "forecast_days": 1,
            }
            async with session.get(OM_BASE, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return data
        except Exception:
            _LOGGER.exception("Open-Meteo current REST fetch failed for %s,%s", self.latitude, self.longitude)
            return None

    def _extract_current_from_rest_result(self, rest_result: Any) -> Optional[Dict[str, Any]]:
        """Extract nearest-hour-like entry from Open-Meteo REST result (hourly arrays) and map to current shape."""
        try:
            if isinstance(rest_result, dict) and "hourly" in rest_result and isinstance(rest_result["hourly"], dict):
                hourly = rest_result["hourly"]
                times = hourly.get("time") or []
                if not times:
                    return None
                now = dt_util.now()
                idx = 0
                try:
                    for i, t_raw in enumerate(times):
                        t = dt_util.parse_datetime(str(t_raw)) if t_raw is not None else None
                        if t and t.tzinfo is None:
                            t = t.replace(tzinfo=timezone.utc)
                        if t and abs((t - now).total_seconds()) < 3600:
                            idx = i
                            break
                except Exception:
                    idx = 0
                candidate = {}
                for k, arr in hourly.items():
                    if k == "time":
                        continue
                    if isinstance(arr, list) and len(arr) > idx:
                        candidate[k] = arr[idx]
                    else:
                        candidate[k] = None
                return self._map_to_current_shape(candidate)
            # If dict keyed by date or direct mapping, try mapping directly
            if isinstance(rest_result, dict):
                return self._map_to_current_shape(rest_result)
            # If list, pick nearest hour-like entry
            if isinstance(rest_result, list):
                now = dt_util.now()
                best = None
                best_delta = None
                for item in rest_result:
                    if not isinstance(item, dict):
                        continue
                    t_raw = item.get("time") or item.get("datetime") or item.get("timestamp")
                    try:
                        t = dt_util.parse_datetime(str(t_raw)) if t_raw is not None else None
                    except Exception:
                        t = None
                    if t is None:
                        continue
                    if t.tzinfo is None:
                        t = t.replace(tzinfo=timezone.utc)
                    delta = abs((t - now).total_seconds())
                    if best is None or delta < best_delta:
                        best = item
                        best_delta = delta
                if best:
                    return self._map_to_current_shape(best)
            # else not recognized
            return None
        except Exception:
            _LOGGER.exception("Error extracting current from REST result")
            return None

    def _map_to_current_shape(self, v: Any) -> Optional[Dict[str, Any]]:
        """Normalize various shapes into expected current-weather dict.

        Returned wind values are normalized to the configured output unit.
        """
        if isinstance(v, dict):
            d = v
        else:
            # Try attribute access
            d = {
                "temperature": getattr(v, "temperature", None),
                "temp": getattr(v, "temp", None),
                "wind_speed": getattr(v, "wind_speed", None),
                "wind_gust": getattr(v, "wind_gust", None),
                "cloud_cover": getattr(v, "cloud_cover", None),
                "precipitation_probability": getattr(v, "precipitation_probability", None),
                "pressure": getattr(v, "pressure", None),
                "units": getattr(v, "units", None),
                "wind_unit": getattr(v, "wind_unit", None),
            }

        def pick(*keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        temp_raw = pick("temperature", "temp", "temperature_2m", "air_temperature", None)
        wind_raw = pick("wind_speed", "wind_kph", "wind_km_h", "wind", "windspeed", None)
        wind_gust_raw = pick("wind_gust", "gust", None) or wind_raw
        cloud_raw = pick("cloud_cover", "clouds", "clouds_percent", "cloud_coverage", None)
        precip_raw = pick("precipitation_probability", "pop", "precipitation", "precip", None)
        pressure_raw = pick("pressure", "air_pressure", "pressure_msl", None)

        # Critical fields check (allow caller to decide to fail if missing)
        if temp_raw is None or wind_raw is None:
            _LOGGER.debug("Mapping produced no temperature or wind (temp=%s wind=%s); returning None", temp_raw, wind_raw)
            return None

        # unit hint if present
        wind_unit_hint = pick("wind_unit", "wind_speed_unit", None)

        try:
            temp_f = _safe_float(temp_raw, DEFAULT_WEATHER_VALUES["temperature"])

            # Interpret incoming wind into m/s, then convert to configured output
            wind_m_s = self._incoming_wind_to_m_s(wind_raw, wind_unit_hint)
            wind_gust_m_s = self._incoming_wind_to_m_s(wind_gust_raw, wind_unit_hint) if wind_gust_raw is not None else wind_m_s

            wind_out = self._m_s_to_output(wind_m_s) if wind_m_s is not None else _safe_float(DEFAULT_WEATHER_VALUES["wind_speed"], DEFAULT_WEATHER_VALUES["wind_speed"])
            wind_gust_out = self._m_s_to_output(wind_gust_m_s) if wind_gust_m_s is not None else _safe_float(DEFAULT_WEATHER_VALUES["wind_gust"], DEFAULT_WEATHER_VALUES["wind_gust"])

            cloud_i = _safe_int(cloud_raw, DEFAULT_WEATHER_VALUES["cloud_cover"]) if cloud_raw is not None else None
            precip_i = _safe_int(precip_raw, DEFAULT_WEATHER_VALUES["precipitation_probability"]) if precip_raw is not None else None
            pressure_f = _safe_float(pressure_raw, DEFAULT_WEATHER_VALUES["pressure"]) if pressure_raw is not None else None

        except Exception as exc:
            _LOGGER.exception("Error coercing Open-Meteo current values: %s", exc)
            return None

        out: Dict[str, Any] = {
            "temperature": temp_f,
            "wind_speed": wind_out,
            "wind_gust": wind_gust_out,
            "wind_unit": self.speed_unit,
        }
        if cloud_i is not None:
            out["cloud_cover"] = cloud_i
        if precip_i is not None:
            out["precipitation_probability"] = precip_i
        if pressure_f is not None:
            out["pressure"] = pressure_f
        return out

    # -----------------------
    # Public: forecast
    # -----------------------
    async def get_forecast(self, days: int = 7, aggregation_periods: Optional[Sequence[Dict[str, int]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get forecast summarized into date->period mappings.
        aggregation_periods: optional sequence of {"name": str, "start_hour": int, "end_hour": int}
          If None, default 4 equal periods are used: 00-06,06-12,12-18,18-24.
        """
        now = dt_util.now()
        forecast_cache_key = f"{self._cache_key}_forecast_{days}_{str(aggregation_periods)}"
        cache_entry = _GLOBAL_CACHE.get(forecast_cache_key)
        if cache_entry:
            cached_time = cache_entry.get("time")
            if isinstance(cached_time, datetime) and (now - cached_time) < self._cache_duration:
                _LOGGER.debug("Using cached forecast data for %s", forecast_cache_key)
                return cache_entry["data"]

        result = await self.fetch_open_meteo_forecast_direct(days)
        if not result:
            _LOGGER.error("Unable to fetch forecast from Open-Meteo for %s; aborting", self._cache_key)
            raise RuntimeError("Unable to fetch forecast from Open-Meteo")

        # If aggregation requested, attempt to aggregate hourly -> periods (helper handles many shapes)
        if aggregation_periods:
            normalized = await self._ensure_period_aggregation(result, days, aggregation_periods)
            if not normalized:
                _LOGGER.error("Open-Meteo returned no usable forecast data for aggregation")
                raise RuntimeError("Unable to fetch forecast from Open-Meteo")
            _GLOBAL_CACHE[forecast_cache_key] = {"data": normalized, "time": now}
            return normalized

        # No aggregation requested — normalize into date->daily-summary
        normalized: Optional[Dict[str, Dict[str, Any]]] = None

        if isinstance(result, dict) and "hourly" in result and isinstance(result.get("hourly"), dict):
            hourly = result.get("hourly")
            times = hourly.get("time") or []
            items: List[Dict[str, Any]] = []
            for idx, t in enumerate(times):
                row = {"time": t}
                for k, arr in hourly.items():
                    if k == "time":
                        continue
                    if isinstance(arr, list) and idx < len(arr):
                        row[k] = arr[idx]
                    else:
                        row[k] = None
                items.append(row)
            normalized = self._normalize_hourly_list_to_daily(items, days)
        elif isinstance(result, dict):
            # Dict keyed by dates or similar shape
            normalized = await self.hass.async_add_executor_job(self._call_sync_normalize_dict, result, days)
        elif isinstance(result, list):
            # list of hourly/daily entries
            first = next((it for it in result if isinstance(it, dict)), None)
            if first:
                if "time" in first or "datetime" in first or "timestamp" in first:
                    normalized = self._normalize_hourly_list_to_daily(result, days)
                else:
                    normalized = self._normalize_forecast_list(result, days)

        if not normalized:
            _LOGGER.error("Open-Meteo REST returned no usable forecast data")
            raise RuntimeError("Unable to fetch forecast from Open-Meteo")

        _LOGGER.info("Fetched forecast from Open-Meteo REST for %s", self._cache_key)
        _GLOBAL_CACHE[forecast_cache_key] = {"data": normalized, "time": now}
        return normalized

    async def fetch_open_meteo_forecast_direct(self, days: int) -> Optional[Dict[str, Any]]:
        """Call OM_BASE for forecast (hourly arrays)."""
        try:
            session: aiohttp.ClientSession = async_get_clientsession(self.hass)
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "timezone": "UTC",
                "hourly": OM_PARAMS_HOURLY,
                "forecast_days": int(days),
            }
            async with session.get(OM_BASE, params=params, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return data
        except Exception:
            _LOGGER.exception("Open-Meteo forecast REST fetch failed for %s,%s", self.latitude, self.longitude)
            return None

    async def fetch(self, latitude: float, longitude: float, mode: str = "hourly", days: int = 5) -> Dict[str, Any]:
        """
        Unified fetch method expected by the coordinator.

        - latitude, longitude: coordinates to request (coordinator passes these)
        - mode: currently only 'hourly' is supported (keeps the same signature for extensibility)
        - days: forecast_days to request from Open-Meteo

        Returns a dict with:
          - 'hourly' (the original Open-Meteo hourly dict, if present)
          - 'timestamps' (list of ISO strings)
          - canonical arrays useful for DataFormatter / scoring:
                temperature_c, wind_m_s, wind_gust_m_s, pressure_hpa,
                cloud_cover, precipitation_probability
        """
        if mode != "hourly":
            _LOGGER.warning("WeatherFetcher.fetch called with unsupported mode=%s; treating as 'hourly'", mode)
        try:
            session: aiohttp.ClientSession = async_get_clientsession(self.hass)
            params = {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "timezone": "UTC",
                "hourly": OM_PARAMS_HOURLY,
                "forecast_days": int(days),
            }
            async with session.get(OM_BASE, params=params, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception:
            _LOGGER.exception("Open-Meteo fetch failed for %s,%s", latitude, longitude)
            raise RuntimeError("Open-Meteo fetch failed")

        out: Dict[str, Any] = {}
        # Preserve original hourly structure if present
        if isinstance(data, dict) and "hourly" in data and isinstance(data.get("hourly"), dict):
            hourly = data.get("hourly") or {}
            # timestamps - Open-Meteo uses 'time' for hourly times
            times = hourly.get("time") or []
            out["hourly"] = dict(hourly)  # shallow copy to avoid mutating original
            out["timestamps"] = list(times)

            # helper to safely copy arrays (coerce to simple lists)
            def _as_list(key: str) -> List[Any]:
                v = hourly.get(key)
                if isinstance(v, list):
                    return list(v)
                return []

            # Canonical arrays expected by scoring/formatter
            # Open-Meteo keys we requested: temperature_2m, wind_speed_10m, windgusts_10m, pressure_msl, cloudcover, precipitation_probability
            out["temperature_c"] = _as_list("temperature_2m")
            out["wind_m_s"] = _as_list("wind_speed_10m")
            out["wind_gust_m_s"] = _as_list("windgusts_10m")
            # pressure_msl from Open-Meteo is in hPa already; expose as pressure_hpa for scoring
            out["pressure_hpa"] = _as_list("pressure_msl")
            # normalize a few keys to expected names
            out["cloud_cover"] = _as_list("cloudcover")
            out["precipitation_probability"] = _as_list("precipitation_probability")
        else:
            # If data isn't hourly-shaped, attempt the best-effort normalization used elsewhere
            # Keep original payload and try to expose timestamps if possible
            out = data or {}
            # attempt to set timestamps key if not present
            if "timestamps" not in out:
                if isinstance(out, dict) and "time" in out:
                    out["timestamps"] = out.get("time")
                elif isinstance(out, dict) and "hourly" in out and isinstance(out.get("hourly"), dict):
                    out["timestamps"] = out.get("hourly", {}).get("time", [])
                else:
                    out.setdefault("timestamps", [])

        _LOGGER.info("WeatherFetcher.fetch: returning forecast payload for %s,%s (days=%s)", latitude, longitude, days)
        return out

    def _call_sync_normalize_dict(self, d: Dict[str, Any], days: int) -> Optional[Dict[str, Dict[str, Any]]]:
        """Helper for sync normalization paths executed in executor."""
        try:
            final: Dict[str, Dict[str, Any]] = {}
            for date_key in sorted(d.keys())[:days]:
                entry = d.get(date_key) or {}
                if not isinstance(entry, dict):
                    continue
                try:
                    temp = _safe_float(entry.get("temperature") or entry.get("temp"), DEFAULT_WEATHER_VALUES["temperature"])
                    wind_m_s = _safe_float(entry.get("wind_speed") or entry.get("wind"), None)
                    gust_m_s = _safe_float(entry.get("wind_gust") or entry.get("gust") or wind_m_s, None)
                except Exception:
                    _LOGGER.debug("Skipping invalid sync-normalize entry for %s", date_key)
                    continue
                cloud = _safe_int(entry.get("cloud_cover") or entry.get("clouds"), DEFAULT_WEATHER_VALUES["cloud_cover"]) if (entry.get("cloud_cover") or entry.get("clouds")) is not None else None
                pop = _safe_int(entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation"), DEFAULT_WEATHER_VALUES["precipitation_probability"]) if (entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation")) is not None else None
                pressure = _safe_float(entry.get("pressure") or entry.get("pressure_msl"), DEFAULT_WEATHER_VALUES["pressure"]) if (entry.get("pressure") or entry.get("pressure_msl")) is not None else None

                wind_out = self._m_s_to_output(wind_m_s) if wind_m_s is not None else None
                gust_out = self._m_s_to_output(gust_m_s) if gust_m_s is not None else None

                final[date_key] = {"temperature": temp, "wind_speed": wind_out, "wind_gust": gust_out}
                if cloud is not None:
                    final[date_key]["cloud_cover"] = cloud
                if pop is not None:
                    final[date_key]["precipitation_probability"] = pop
                if pressure is not None:
                    final[date_key]["pressure"] = pressure
            return final or None
        except Exception:
            return None

    # -----------------------
    # Helpers: normalize various shapes
    # -----------------------
    def _normalize_forecast_list(self, lst: List[Any], days: int) -> Dict[str, Dict[str, Any]]:
        """Normalize list of dicts into date-keyed daily summaries (expects daily entries)."""
        final: Dict[str, Dict[str, Any]] = {}
        for item in lst:
            if not isinstance(item, dict):
                continue
            # find a date key
            date_key = None
            for k in ("date", "time", "datetime", "day"):
                if k in item and item.get(k):
                    v = item.get(k)
                    try:
                        if isinstance(v, (int, float)):
                            dt = datetime.fromtimestamp(float(v), tz=timezone.utc)
                            date_key = dt.date().isoformat()
                        else:
                            s = str(v).split("T")[0]
                            datetime.strptime(s, "%Y-%m-%d")
                            date_key = s
                    except Exception:
                        date_key = None
                if date_key:
                    break
            if not date_key:
                continue
            try:
                temp = _safe_float(item.get("temperature") or item.get("temp") or item.get("temperature_2m"), DEFAULT_WEATHER_VALUES["temperature"])
                wind_m_s = _safe_float(item.get("wind_speed") or item.get("wind") or item.get("wind_speed_10m"), None)
                gust_m_s = _safe_float(item.get("wind_gust") or item.get("gust") or wind_m_s, None)
            except Exception:
                _LOGGER.debug("Skipping invalid forecast list item for %s", date_key)
                continue
            cloud = _safe_int(item.get("cloud_cover") or item.get("clouds"), DEFAULT_WEATHER_VALUES["cloud_cover"]) if (item.get("cloud_cover") or item.get("clouds")) is not None else None
            pop = _safe_int(item.get("precipitation_probability") or item.get("pop") or item.get("precipitation") or item.get("precip"), DEFAULT_WEATHER_VALUES["precipitation_probability"]) if (item.get("precipitation_probability") or item.get("pop") or item.get("precipitation") or item.get("precip")) is not None else None
            pressure = _safe_float(item.get("pressure") or item.get("pressure_msl"), DEFAULT_WEATHER_VALUES["pressure"]) if (item.get("pressure") or item.get("pressure_msl")) is not None else None

            final[date_key] = {
                "temperature": temp,
                "wind_speed": self._m_s_to_output(wind_m_s) if wind_m_s is not None else None,
                "wind_gust": self._m_s_to_output(gust_m_s) if gust_m_s is not None else None,
            }
            if cloud is not None:
                final[date_key]["cloud_cover"] = cloud
            if pop is not None:
                final[date_key]["precipitation_probability"] = pop
            if pressure is not None:
                final[date_key]["pressure"] = pressure
            if len(final) >= days:
                break
        return final

    def _normalize_hourly_list_to_daily(self, hourly_list: List[Any], days: int) -> Dict[str, Dict[str, Any]]:
        """
        Convert list of hourly entries (each with time + metrics) into daily summaries:
        - temperature: mean
        - wind_speed: mean
        - wind_gust: max
        - cloud_cover: mean (rounded)
        - precipitation_probability: max
        - pressure: mean
        Returned wind values are in configured output unit.
        """
        per_date: Dict[str, Dict[str, Any]] = {}
        for entry in hourly_list:
            if not isinstance(entry, dict):
                continue
            t_raw = entry.get("time") or entry.get("datetime") or entry.get("timestamp")
            if t_raw is None:
                continue
            try:
                t = dt_util.parse_datetime(str(t_raw)) if t_raw is not None else None
            except Exception:
                t = None
            if t is None:
                try:
                    tnum = float(t_raw)
                    if tnum > 1e12:
                        tnum = tnum / 1000.0
                    t = datetime.fromtimestamp(tnum, tz=timezone.utc)
                except Exception:
                    continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            date_key = t.date().isoformat()
            try:
                temp = _safe_float(entry.get("temperature") or entry.get("temp") or entry.get("temperature_2m"), DEFAULT_WEATHER_VALUES["temperature"])
                wind_m_s = _safe_float(entry.get("wind_speed") or entry.get("wind") or entry.get("wind_speed_10m"), DEFAULT_WEATHER_VALUES["wind_speed"])  # assume m/s
                gust_m_s = _safe_float(entry.get("wind_gust") or entry.get("gust"), wind_m_s or DEFAULT_WEATHER_VALUES["wind_gust"])
            except Exception:
                continue
            cloud = _safe_int(entry.get("cloud_cover") or entry.get("clouds") or entry.get("cloudcover"), DEFAULT_WEATHER_VALUES["cloud_cover"]) if (entry.get("cloud_cover") or entry.get("clouds") or entry.get("cloudcover")) is not None else None
            pop = _safe_int(entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation") or entry.get("precip"), DEFAULT_WEATHER_VALUES["precipitation_probability"]) if (entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation") or entry.get("precip")) is not None else None
            pressure = _safe_float(entry.get("pressure") or entry.get("pressure_msl"), DEFAULT_WEATHER_VALUES["pressure"]) if (entry.get("pressure") or entry.get("pressure_msl")) is not None else None

            agg = per_date.get(date_key)
            if not agg:
                per_date[date_key] = {
                    "temperature_sum": temp,
                    "wind_speed_sum": wind_m_s or 0.0,
                    "pressure_sum": pressure or 0.0,
                    "cloud_sum": cloud or 0,
                    "precip_max": pop or 0,
                    "gust_max": gust_m_s,
                    "count": 1,
                }
            else:
                agg["temperature_sum"] += temp
                agg["wind_speed_sum"] += (wind_m_s or 0.0)
                agg["pressure_sum"] += (pressure or 0.0)
                agg["cloud_sum"] += (cloud or 0)
                agg["precip_max"] = max(agg["precip_max"], pop or 0)
                agg["gust_max"] = max(agg["gust_max"], gust_m_s)
                agg["count"] += 1

        if not per_date:
            return {}

        final: Dict[str, Dict[str, Any]] = {}
        for date_key in sorted(per_date.keys())[:days]:
            agg = per_date[date_key]
            cnt = agg.get("count", 1) or 1
            try:
                mean_wind_m_s = float(agg["wind_speed_sum"]) / cnt
                gust_m_s = float(agg["gust_max"])
                final[date_key] = {
                    "temperature": float(agg["temperature_sum"]) / cnt,
                    "wind_speed": self._m_s_to_output(mean_wind_m_s),
                    "wind_gust": self._m_s_to_output(gust_m_s),
                    "cloud_cover": int(round(float(agg["cloud_sum"]) / cnt)) if agg.get("cloud_sum") is not None else None,
                    "precipitation_probability": int(round(float(agg["precip_max"]))),
                    "pressure": float(agg["pressure_sum"]) / cnt if agg.get("pressure_sum") is not None else None,
                }
            except Exception:
                _LOGGER.debug("Failed to aggregate hourly data for %s; skipping", date_key)
                continue
        return final

    async def _ensure_period_aggregation(self, data: Dict[str, Any], days: int, aggregation_periods: Sequence[Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
        """
        If `data` is already date-keyed summaries, return as-is (truncated to days).
        If `data` is hourly list under special keys, attempt to aggregate into requested periods.
        For simplicity this helper accepts the REST-returned dict (possibly hourly arrays) and tries
        to make a best-effort aggregation.
        """
        # If already date keyed with metrics, truncate and return
        if all(isinstance(v, dict) and ("temperature" in v or "wind_speed" in v) for v in data.values()):
            # ensure wind units present (if numeric assumed m/s -> convert)
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in list(data.items())[:days]:
                if isinstance(v, dict) and v.get("wind_speed") is not None:
                    # assume v["wind_speed"] is in m/s if it looks small (< 50) or no unit provided
                    # We conservatively assume incoming daily summaries are already in m/s and convert
                    try:
                        wind_m_s = float(v.get("wind_speed"))
                        gust_m_s = float(v.get("wind_gust")) if v.get("wind_gust") is not None else None
                        vv = dict(v)
                        vv["wind_speed"] = self._m_s_to_output(wind_m_s)
                        vv["wind_gust"] = self._m_s_to_output(gust_m_s) if gust_m_s is not None else None
                        vv["wind_unit"] = self.speed_unit
                        out[k] = vv
                    except Exception:
                        out[k] = v
                else:
                    out[k] = v
            return out

        # If dict has 'hourly' arrays, convert to list first
        hourly_list = None
        if isinstance(data, dict) and "hourly" in data and isinstance(data.get("hourly"), dict):
            hourly = data.get("hourly")
            times = hourly.get("time") or []
            items: List[Dict[str, Any]] = []
            for idx, t in enumerate(times):
                row = {"time": t}
                for key, arr in hourly.items():
                    if key == "time":
                        continue
                    if isinstance(arr, list) and idx < len(arr):
                        row[key] = arr[idx]
                    else:
                        row[key] = None
                items.append(row)
            hourly_list = items
        # If dict is a mapping of date->list, or contains an 'entries' key, try to find an hourly list
        if hourly_list is None:
            values = [v for v in data.values() if isinstance(v, list)]
            if values:
                first = values[0]
                if first and isinstance(first[0], dict) and ("time" in first[0] or "datetime" in first[0]):
                    hourly_list = first

        if hourly_list is None and isinstance(data, list):
            hourly_list = data

        if hourly_list is None:
            return self._normalize_forecast_list(list(data.values()) if isinstance(data, dict) else [], days)

        # Aggregate hourly_list into periods per day
        return self._aggregate_hourly_into_periods(hourly_list, days, aggregation_periods)

    def _aggregate_hourly_into_periods(self, hourly_list: List[Dict[str, Any]], days: int, aggregation_periods: Optional[Sequence[Dict[str, int]]]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate hourly data into named periods. aggregation_periods is a sequence of dicts:
          {"name": str, "start_hour": int, "end_hour": int}
        Hours are in 0-24 with end exclusive. If None, default four 6-hour periods are used.
        Aggregation rules:
          - temperature: mean
          - wind_speed: mean
          - wind_gust: max
          - cloud_cover: mean (rounded)
          - precipitation_probability: max
          - pressure: mean
        Returns dict keyed by date -> { period_name: {metrics...}, ... }
        """
        if not aggregation_periods:
            aggregation_periods = [
                {"name": "period_00_06", "start_hour": 0, "end_hour": 6},
                {"name": "period_06_12", "start_hour": 6, "end_hour": 12},
                {"name": "period_12_18", "start_hour": 12, "end_hour": 18},
                {"name": "period_18_24", "start_hour": 18, "end_hour": 24},
            ]

        per_date_periods: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for entry in hourly_list:
            if not isinstance(entry, dict):
                continue
            t_raw = entry.get("time") or entry.get("datetime") or entry.get("timestamp")
            if t_raw is None:
                continue
            try:
                t = dt_util.parse_datetime(str(t_raw)) if t_raw is not None else None
            except Exception:
                t = None
            if t is None:
                try:
                    tnum = float(t_raw)
                    if tnum > 1e12:
                        tnum = tnum / 1000.0
                    t = datetime.fromtimestamp(tnum, tz=timezone.utc)
                except Exception:
                    continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            date_key = t.date().isoformat()
            hour = t.hour

            # find matching period
            for p in aggregation_periods:
                start = int(p["start_hour"])
                end = int(p["end_hour"])
                if start <= hour < end:
                    pname = p["name"]
                    per_date_periods.setdefault(date_key, {}).setdefault(pname, {
                        "temperature_sum": 0.0,
                        "wind_speed_sum": 0.0,
                        "pressure_sum": 0.0,
                        "cloud_sum": 0,
                        "precip_max": 0,
                        "gust_max": 0,
                        "count": 0,
                    })
                    agg = per_date_periods[date_key][pname]
                    try:
                        temp = _safe_float(entry.get("temperature") or entry.get("temp") or entry.get("temperature_2m"), DEFAULT_WEATHER_VALUES["temperature"])
                        wind_m_s = _safe_float(entry.get("wind_speed") or entry.get("wind") or entry.get("wind_speed_10m"), DEFAULT_WEATHER_VALUES["wind_speed"])
                        gust_m_s = _safe_float(entry.get("wind_gust") or entry.get("gust") or entry.get("wind") or DEFAULT_WEATHER_VALUES["wind_gust"])
                    except Exception:
                        continue
                    cloud = _safe_int(entry.get("cloud_cover") or entry.get("clouds") or entry.get("cloudcover"), DEFAULT_WEATHER_VALUES["cloud_cover"]) if (entry.get("cloud_cover") or entry.get("clouds") or entry.get("cloudcover")) is not None else None
                    pop = _safe_int(entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation") or entry.get("precip"), DEFAULT_WEATHER_VALUES["precipitation_probability"]) if (entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation") or entry.get("precip")) is not None else None
                    pressure = _safe_float(entry.get("pressure") or entry.get("pressure_msl"), DEFAULT_WEATHER_VALUES["pressure"]) if (entry.get("pressure") or entry.get("pressure_msl")) is not None else None

                    agg["temperature_sum"] += temp
                    agg["wind_speed_sum"] += (wind_m_s or 0.0)
                    agg["pressure_sum"] += (pressure or 0.0)
                    agg["cloud_sum"] += (cloud or 0)
                    agg["precip_max"] = max(agg["precip_max"], pop or 0)
                    agg["gust_max"] = max(agg["gust_max"], gust_m_s)
                    agg["count"] += 1
                    break

        # finalize into requested shape: date -> period_name -> metrics
        final: Dict[str, Dict[str, Any]] = {}
        for date_key in sorted(per_date_periods.keys())[:days]:
            final[date_key] = {}
            for pname, agg in per_date_periods[date_key].items():
                cnt = agg.get("count", 1) or 1
                try:
                    mean_wind_m_s = float(agg["wind_speed_sum"]) / cnt
                    gust_m_s = float(agg["gust_max"])
                    final[date_key][pname] = {
                        "temperature": float(agg["temperature_sum"]) / cnt,
                        "wind_speed": self._m_s_to_output(mean_wind_m_s),
                        "wind_gust": self._m_s_to_output(gust_m_s),
                        "cloud_cover": int(round(float(agg["cloud_sum"]) / cnt)) if agg.get("cloud_sum") is not None else None,
                        "precipitation_probability": int(round(float(agg["precip_max"]))),
                        "pressure": float(agg["pressure_sum"]) / cnt if agg.get("pressure_sum") is not None else None,
                    }
                except Exception:
                    _LOGGER.debug("Failed to finalize aggregation for %s %s; skipping", date_key, pname)
                    continue

        return final

    # -----------------------
    # Marine endpoint helper
    # -----------------------
    async def fetch_marine_direct(self, days: int = 5) -> Optional[Dict[str, Any]]:
        """
        Attempt to fetch marine-specific variables directly from the Open-Meteo Marine endpoint.
        Returns normalized structure similar to Open-Meteo client output (hourly dicts).
        """
        try:
            session: aiohttp.ClientSession = async_get_clientsession(self.hass)
            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "timezone": "UTC",
                # request common marine fields — callers may merge as needed
                "hourly": ",".join(["wave_height", "wave_direction", "wave_period", "swell_height", "swell_period"]),
                "forecast_days": int(days),
            }
            async with session.get(OM_MARINE_BASE, params=params, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
            # leave raw structure (caller will extract hourly arrays)
            return data
        except Exception:
            _LOGGER.exception("Direct Marine API fetch failed for %s,%s", self.latitude, self.longitude)
            return None

    # -----------------------
    # End
    # -----------------------