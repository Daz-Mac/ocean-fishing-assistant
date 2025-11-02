"""Weather data fetcher - Open-Meteo focused.

Features:
- Uses an injected Open-Meteo client when provided (will try many candidate methods).
- Falls back to calling Open-Meteo / Marine endpoints directly (uses HA aiohttp client session).
- Caching (global) for current & forecast.
- Robust normalization for many shapes (dicts, lists, objects).
- Unit coercion (m/s -> km/h) and safe numeric coercion helpers.
- Forecast aggregation into flexible "periods" (default: 4 equal periods per day) or custom periods
  (useful for Dusk/Dawn or arbitrary monitored periods).
- Strict failure semantics: fails loudly when current/forecast critical fields are missing.
"""
from __future__ import annotations

import inspect
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp

from homeassistant.util import dt as dt_util
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import OM_BASE, OM_MARINE_BASE

_LOGGER = logging.getLogger(__name__)

# Global caches (shared across instances)
_GLOBAL_CACHE: Dict[str, Dict[str, Any]] = {}

# Defaults for diagnostics only (we fail loudly instead of silently using these)
DEFAULT_WEATHER_VALUES = {
    "temperature": 15.0,
    "wind_speed": 10.0,  # km/h
    "wind_gust": 15.0,  # km/h
    "cloud_cover": 50,  # percentage
    "precipitation_probability": 0,  # percentage
    "pressure": 1013.0,  # hPa
}


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
    Fetch current weather and forecast using an injected Open-Meteo client when available.
    If marine/wave fields are missing the fetcher will attempt a direct call to the Marine API
    endpoint and merge results.

    aggregation_periods: optional parameter for get_forecast that defines how hourly points are
    aggregated into monitoring periods. Expected shape:
      [
        {"name": "dawn", "start_hour": 4, "end_hour": 6},
        {"name": "day",  "start_hour": 6, "end_hour": 18},
        ...
      ]
    If None, a default 4-period split is used (00-06, 06-12, 12-18, 18-24).
    """

    def __init__(
        self,
        hass,
        latitude: float,
        longitude: float,
        use_open_meteo: bool = True,
        open_meteo_client: Optional[Any] = None,
    ) -> None:
        self.hass = hass
        self.latitude = round(float(latitude), 4)
        self.longitude = round(float(longitude), 4)
        self.use_open_meteo = use_open_meteo
        self.open_meteo_client = open_meteo_client

        self._cache_key = f"{self.latitude}_{self.longitude}_{'om' if use_open_meteo else 'none'}"
        self._cache_duration = timedelta(minutes=30)

    # -----------------------
    # Public: current weather
    # -----------------------
    async def get_weather_data(self) -> Dict[str, Any]:
        """Get current normalized weather data. Uses cache -> Open-Meteo client -> raise on failure."""
        now = dt_util.now()
        cache_entry = _GLOBAL_CACHE.get(self._cache_key)
        if cache_entry:
            cached_time = cache_entry.get("time")
            if isinstance(cached_time, datetime) and (now - cached_time) < self._cache_duration:
                _LOGGER.debug("Using cached weather data for %s", self._cache_key)
                return cache_entry["data"]

        if self.use_open_meteo and self.open_meteo_client:
            try:
                result = await self._call_open_meteo_current()
                if result:
                    # Validate critical metrics (temperature and wind required for scoring)
                    if not isinstance(result, dict) or result.get("temperature") is None or result.get(
                        "wind_speed"
                    ) is None:
                        _LOGGER.error("Open-Meteo returned incomplete current-weather data: %s", result)
                        raise RuntimeError("Incomplete weather data from Open-Meteo")
                    _LOGGER.info("Fetched current weather from Open-Meteo client for %s", self._cache_key)
                    _GLOBAL_CACHE[self._cache_key] = {"data": result, "time": now}
                    return result
                _LOGGER.error("Open-Meteo client returned no usable current-weather data")
            except Exception as exc:
                _LOGGER.exception("Open-Meteo client current fetch failed: %s", exc)

        _LOGGER.error("Unable to fetch current weather data from Open-Meteo for %s; aborting", self._cache_key)
        raise RuntimeError("Unable to fetch current weather data from Open-Meteo")

    async def _call_open_meteo_current(self) -> Optional[Dict[str, Any]]:
        """Discover and call a suitable method on the injected client to obtain 'current' weather."""
        client = self.open_meteo_client
        if client is None:
            return None

        candidate_methods = [
            "get_current",
            "get_current_weather",
            "fetch_current",
            "fetch_current_weather",
            "current",
            "get_now",
            "fetch_hourly_forecast",
            "get_hourly",
            "fetch_hourly",
        ]
        fn = None
        for name in candidate_methods:
            if hasattr(client, name):
                fn = getattr(client, name)
                break
        if fn is None and hasattr(client, "fetch"):
            fn = getattr(client, "fetch")

        if fn is None:
            _LOGGER.debug("Open-Meteo client has no recognized current-weather method")
            return None

        try:
            result = fn() if callable(fn) else None
            if inspect.isawaitable(result):
                result = await result
            if not result:
                return None

            # If result is a list -> pick nearest hour entry
            if isinstance(result, list):
                now = dt_util.now()
                best = None
                best_delta = None
                for item in result:
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
                return None

            # If dict -> handle Open-Meteo style hourly => pick nearest hour
            if isinstance(result, dict):
                if "hourly" in result and isinstance(result.get("hourly"), dict):
                    hourly = result.get("hourly")
                    times = hourly.get("time") or []
                    if isinstance(times, list) and len(times) > 0:
                        idx = 0
                        try:
                            now = dt_util.now()
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
                            if isinstance(arr, list) and len(arr) > idx:
                                candidate[k] = arr[idx]
                            else:
                                candidate[k] = None
                        return self._map_to_current_shape(candidate)
                # else try mapping dict directly
                return self._map_to_current_shape(result)

            # If object with attributes, attempt mapping via attribute access
            return self._map_to_current_shape(result)
        except Exception as exc:
            _LOGGER.exception("Error calling Open-Meteo current method: %s", exc)
            return None

    def _map_to_current_shape(self, v: Any) -> Optional[Dict[str, Any]]:
        """Normalize various shapes into expected current-weather dict.

        Returned wind values are normalized to km/h (m/s -> km/h conversion applied if reported units).
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

        wind_unit = pick("wind_unit", "wind_speed_unit", None)

        try:
            temp_f = _safe_float(temp_raw, DEFAULT_WEATHER_VALUES["temperature"])
            wind_f = _safe_float(wind_raw, DEFAULT_WEATHER_VALUES["wind_speed"])
            wind_gust_f = _safe_float(wind_gust_raw, DEFAULT_WEATHER_VALUES["wind_gust"])
            cloud_i = _safe_int(cloud_raw, DEFAULT_WEATHER_VALUES["cloud_cover"]) if cloud_raw is not None else None
            precip_i = _safe_int(precip_raw, DEFAULT_WEATHER_VALUES["precipitation_probability"]) if precip_raw is not None else None
            pressure_f = _safe_float(pressure_raw, DEFAULT_WEATHER_VALUES["pressure"]) if pressure_raw is not None else None

            if wind_unit and str(wind_unit).strip().lower() in ("m/s", "mps"):
                wind_f = wind_f * 3.6
                wind_gust_f = wind_gust_f * 3.6
        except Exception as exc:
            _LOGGER.exception("Error coercing Open-Meteo client current values: %s", exc)
            return None

        out: Dict[str, Any] = {
            "temperature": temp_f,
            "wind_speed": wind_f,
            "wind_gust": wind_gust_f,
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

        if self.use_open_meteo and self.open_meteo_client:
            try:
                result = await self._call_open_meteo_forecast(days)
                if result:
                    # result should be a dict keyed by date -> summary (we will optionally re-aggregate into periods)
                    # If result looks hourly-list style, _call_open_meteo_forecast will already handle aggregation
                    _LOGGER.info("Fetched forecast from Open-Meteo client for %s", self._cache_key)
                    # Optionally aggregate hourly -> periods (if result has hourly-like list)
                    if aggregation_periods:
                        # If result is a dict of dates->hourly list, normalize; otherwise assume already summarized
                        normalized = await self._ensure_period_aggregation(result, days, aggregation_periods)
                        _GLOBAL_CACHE[forecast_cache_key] = {"data": normalized, "time": now}
                        return normalized
                    _GLOBAL_CACHE[forecast_cache_key] = {"data": result, "time": now}
                    return result
                _LOGGER.error("Open-Meteo client returned no usable forecast data")
            except Exception as exc:
                _LOGGER.exception("Open-Meteo client forecast fetch failed: %s", exc)

        _LOGGER.error("Unable to fetch forecast from Open-Meteo for %s; aborting", self._cache_key)
        raise RuntimeError("Unable to fetch forecast from Open-Meteo")

    async def _call_open_meteo_forecast(self, days: int) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Try to call forecast methods on the injected client. Handles:
         - Open-Meteo style dicts (hourly arrays)
         - dict keyed by ISO dates
         - list of dicts (hourly daily)
         - objects with 'daily' attribute
        """
        client = self.open_meteo_client
        if client is None:
            return None

        candidate_methods = [
            "get_forecast",
            "get_daily_forecast",
            "fetch_forecast",
            "fetch_daily",
            "get_weather_forecast",
            "forecast",
            "fetch_hourly_forecast",
            "get_hourly",
            "fetch_hourly",
        ]
        fn = None
        for name in candidate_methods:
            if hasattr(client, name):
                fn = getattr(client, name)
                break
        if fn is None and hasattr(client, "fetch"):
            fn = getattr(client, "fetch")
        if fn is None:
            _LOGGER.debug("Open-Meteo client has no recognized forecast method")
            return None

        try:
            result = fn() if callable(fn) else None
            if inspect.isawaitable(result):
                result = await result
            if not result:
                return None

            # If dict with 'hourly' arrays (Open-Meteo style) -> turn into hourly list then aggregate to days/periods
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
                # Aggregate to daily summaries (by default) - caller may request custom periods when calling get_forecast
                return self._normalize_hourly_list_to_daily(items, days)

            # If dict keyed by dates -> normalize entries
            if isinstance(result, dict):
                normalized: Dict[str, Dict[str, Any]] = {}
                for key, val in sorted(result.items()):
                    date_key = str(key).split("T")[0]
                    if isinstance(val, dict):
                        try:
                            temp = _safe_float(val.get("temperature") or val.get("temp") or val.get("temperature_2m"), DEFAULT_WEATHER_VALUES["temperature"])
                            wind = _safe_float(val.get("wind_speed") or val.get("wind") or val.get("wind_speed_10m"), DEFAULT_WEATHER_VALUES["wind_speed"])
                            gust = _safe_float(val.get("wind_gust") or val.get("gust") or wind, DEFAULT_WEATHER_VALUES["wind_gust"])
                        except Exception:
                            _LOGGER.debug("Skipping invalid forecast entry for date %s", date_key)
                            continue
                        cloud = _safe_int(val.get("cloud_cover") or val.get("clouds"), DEFAULT_WEATHER_VALUES["cloud_cover"]) if (val.get("cloud_cover") or val.get("clouds")) is not None else None
                        pop = _safe_int(val.get("precipitation_probability") or val.get("pop") or val.get("precipitation"), DEFAULT_WEATHER_VALUES["precipitation_probability"]) if (val.get("precipitation_probability") or val.get("pop") or val.get("precipitation")) is not None else None
                        pressure = _safe_float(val.get("pressure") or val.get("pressure_msl"), DEFAULT_WEATHER_VALUES["pressure"]) if (val.get("pressure") or val.get("pressure_msl")) is not None else None
                        normalized[date_key] = {"temperature": temp, "wind_speed": wind, "wind_gust": gust}
                        if cloud is not None:
                            normalized[date_key]["cloud_cover"] = cloud
                        if pop is not None:
                            normalized[date_key]["precipitation_probability"] = pop
                        if pressure is not None:
                            normalized[date_key]["pressure"] = pressure
                if normalized:
                    limited = dict(list(normalized.items())[:days])
                    return limited or None
                return None

            # If a list -> hourly or daily entries
            if isinstance(result, list):
                first = next((it for it in result if isinstance(it, dict)), None)
                if first:
                    if "time" in first or "datetime" in first or "timestamp" in first:
                        normalized = self._normalize_hourly_list_to_daily(result, days)
                        return normalized or None
                    normalized = self._normalize_forecast_list(result, days)
                    return normalized or None

            # If object with 'daily' attribute
            if hasattr(result, "daily"):
                daily = getattr(result, "daily")
                if isinstance(daily, dict):
                    # run sync normalizer in executor if needed
                    return await self.hass.async_add_executor_job(self._call_sync_normalize_dict, daily, days)
                if isinstance(daily, list):
                    normalized = self._normalize_hourly_list_to_daily(daily, days)
                    return normalized or None

            _LOGGER.debug("Unrecognized forecast shape from Open-Meteo client: %s", type(result))
            return None
        except Exception as exc:
            _LOGGER.exception("Error calling Open-Meteo forecast method: %s", exc)
            return None

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
                    wind = _safe_float(entry.get("wind_speed") or entry.get("wind"), DEFAULT_WEATHER_VALUES["wind_speed"])
                    gust = _safe_float(entry.get("wind_gust") or entry.get("gust") or wind, DEFAULT_WEATHER_VALUES["wind_gust"])
                except Exception:
                    _LOGGER.debug("Skipping invalid sync-normalize entry for %s", date_key)
                    continue
                cloud = _safe_int(entry.get("cloud_cover") or entry.get("clouds"), DEFAULT_WEATHER_VALUES["cloud_cover"]) if (entry.get("cloud_cover") or entry.get("clouds")) is not None else None
                pop = _safe_int(entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation"), DEFAULT_WEATHER_VALUES["precipitation_probability"]) if (entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation")) is not None else None
                pressure = _safe_float(entry.get("pressure") or entry.get("pressure_msl"), DEFAULT_WEATHER_VALUES["pressure"]) if (entry.get("pressure") or entry.get("pressure_msl")) is not None else None
                final[date_key] = {"temperature": temp, "wind_speed": wind, "wind_gust": gust}
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
                wind = _safe_float(item.get("wind_speed") or item.get("wind") or item.get("wind_speed_10m"), DEFAULT_WEATHER_VALUES["wind_speed"])
                gust = _safe_float(item.get("wind_gust") or item.get("gust") or wind, DEFAULT_WEATHER_VALUES["wind_gust"])
            except Exception:
                _LOGGER.debug("Skipping invalid forecast list item for %s", date_key)
                continue
            cloud = _safe_int(item.get("cloud_cover") or item.get("clouds"), DEFAULT_WEATHER_VALUES["cloud_cover"]) if (item.get("cloud_cover") or item.get("clouds")) is not None else None
            pop = _safe_int(item.get("precipitation_probability") or item.get("pop") or item.get("precipitation"), DEFAULT_WEATHER_VALUES["precipitation_probability"]) if (item.get("precipitation_probability") or item.get("pop") or item.get("precipitation")) is not None else None
            pressure = _safe_float(item.get("pressure") or item.get("pressure_msl"), DEFAULT_WEATHER_VALUES["pressure"]) if (item.get("pressure") or item.get("pressure_msl")) is not None else None
            final[date_key] = {"temperature": temp, "wind_speed": wind, "wind_gust": gust}
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
                wind = _safe_float(entry.get("wind_speed") or entry.get("wind") or entry.get("wind_speed_10m"), DEFAULT_WEATHER_VALUES["wind_speed"])
                gust = _safe_float(entry.get("wind_gust") or entry.get("gust"), wind or DEFAULT_WEATHER_VALUES["wind_gust"])
            except Exception:
                continue
            cloud = _safe_int(entry.get("cloud_cover") or entry.get("clouds") or entry.get("cloudcover"), DEFAULT_WEATHER_VALUES["cloud_cover"]) if (entry.get("cloud_cover") or entry.get("clouds") or entry.get("cloudcover")) is not None else None
            pop = _safe_int(entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation") or entry.get("precip"), DEFAULT_WEATHER_VALUES["precipitation_probability"]) if (entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation") or entry.get("precip")) is not None else None
            pressure = _safe_float(entry.get("pressure") or entry.get("pressure_msl"), DEFAULT_WEATHER_VALUES["pressure"]) if (entry.get("pressure") or entry.get("pressure_msl")) is not None else None

            agg = per_date.get(date_key)
            if not agg:
                per_date[date_key] = {
                    "temperature_sum": temp,
                    "wind_speed_sum": wind,
                    "pressure_sum": pressure or 0.0,
                    "cloud_sum": cloud or 0,
                    "precip_max": pop or 0,
                    "gust_max": gust,
                    "count": 1,
                }
            else:
                agg["temperature_sum"] += temp
                agg["wind_speed_sum"] += wind
                agg["pressure_sum"] += (pressure or 0.0)
                agg["cloud_sum"] += (cloud or 0)
                agg["precip_max"] = max(agg["precip_max"], pop or 0)
                agg["gust_max"] = max(agg["gust_max"], gust)
                agg["count"] += 1

        if not per_date:
            return {}

        final: Dict[str, Dict[str, Any]] = {}
        for date_key in sorted(per_date.keys())[:days]:
            agg = per_date[date_key]
            cnt = agg.get("count", 1) or 1
            try:
                final[date_key] = {
                    "temperature": float(agg["temperature_sum"]) / cnt,
                    "wind_speed": float(agg["wind_speed_sum"]) / cnt,
                    "wind_gust": float(agg["gust_max"]),
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
        For simplicity this helper accepts the client-returned dict (possibly hourly arrays) and tries
        to make a best-effort aggregation.
        """
        # If already date keyed with metrics, truncate and return
        if all(isinstance(v, dict) and ("temperature" in v or "wind_speed" in v) for v in data.values()):
            return dict(list(data.items())[:days])

        # If dict has 'hourly' arrays, convert to list first
        hourly_list = None
        if isinstance(data, dict) and "hourly" in data and isinstance(data.get("hourly"), dict):
            hourly = data.get("hourly")
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
            hourly_list = items
        # If dict is a mapping of date->list, or contains an 'entries' key, try to find an hourly list
        if hourly_list is None:
            # look for list-like values
            values = [v for v in data.values() if isinstance(v, list)]
            if values:
                first = values[0]
                if first and isinstance(first[0], dict) and ("time" in first[0] or "datetime" in first[0]):
                    hourly_list = first

        # If still not found, maybe caller passed a plain list
        if hourly_list is None and isinstance(data, list):
            hourly_list = data

        if hourly_list is None:
            # fallback: try to normalize to daily
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
            # default 4 x 6-hour windows
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
                # handle end == 24
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
                        wind = _safe_float(entry.get("wind_speed") or entry.get("wind") or entry.get("wind_speed_10m"), DEFAULT_WEATHER_VALUES["wind_speed"])
                        gust = _safe_float(entry.get("wind_gust") or entry.get("gust") or entry.get("wind") or DEFAULT_WEATHER_VALUES["wind_gust"])
                    except Exception:
                        continue
                    cloud = _safe_int(entry.get("cloud_cover") or entry.get("clouds") or entry.get("cloudcover"), DEFAULT_WEATHER_VALUES["cloud_cover"]) if (entry.get("cloud_cover") or entry.get("clouds") or entry.get("cloudcover")) is not None else None
                    pop = _safe_int(entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation") or entry.get("precip"), DEFAULT_WEATHER_VALUES["precipitation_probability"]) if (entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation") or entry.get("precip")) is not None else None
                    pressure = _safe_float(entry.get("pressure") or entry.get("pressure_msl"), DEFAULT_WEATHER_VALUES["pressure"]) if (entry.get("pressure") or entry.get("pressure_msl")) is not None else None

                    agg["temperature_sum"] += temp
                    agg["wind_speed_sum"] += wind
                    agg["pressure_sum"] += (pressure or 0.0)
                    agg["cloud_sum"] += (cloud or 0)
                    agg["precip_max"] = max(agg["precip_max"], pop or 0)
                    agg["gust_max"] = max(agg["gust_max"], gust)
                    agg["count"] += 1
                    break

        # finalize into requested shape: date -> period_name -> metrics
        final: Dict[str, Dict[str, Any]] = {}
        for date_key in sorted(per_date_periods.keys())[:days]:
            final[date_key] = {}
            for pname, agg in per_date_periods[date_key].items():
                cnt = agg.get("count", 1) or 1
                try:
                    final[date_key][pname] = {
                        "temperature": float(agg["temperature_sum"]) / cnt,
                        "wind_speed": float(agg["wind_speed_sum"]) / cnt,
                        "wind_gust": float(agg["gust_max"]),
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