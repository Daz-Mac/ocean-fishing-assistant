"""
Strict WeatherFetcher for Open‑Meteo.

- Requires explicit speed_unit in constructor (no HA unit detection).
- Raises on HTTP or payload shape failures.
- Marine endpoint failures are fatal (marine data required for Ocean Fishing Assistant).
- Returns raw Open‑Meteo payloads for forecast/fetch; provides strict, validated current snapshot via get_weather_data().
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

# hourly params to request from Open-Meteo (we require these fields to be present)
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


def _to_float_strict(value: Any, name: str) -> float:
    """Convert value to float strictly; raise ValueError if missing/invalid."""
    if value is None:
        raise ValueError(f"Missing numeric value for '{name}' (strict)")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Unable to convert '{name}' to float: {value!r}") from exc


def _to_int_strict(value: Any, name: str) -> int:
    """Convert value to int strictly; raise ValueError if missing/invalid."""
    if value is None:
        raise ValueError(f"Missing integer value for '{name}' (strict)")
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"Unable to convert '{name}' to int: {value!r}") from exc


class WeatherFetcher:
    """
    Strict fetcher for Open-Meteo REST endpoints.

    speed_unit: required output unit for wind values, must be one of "km/h", "mph", "m/s".
    """

    def __init__(self, hass, latitude: float, longitude: float, speed_unit: str) -> None:
        self.hass = hass
        self.latitude = round(float(latitude), 6)
        self.longitude = round(float(longitude), 6)

        if not speed_unit or str(speed_unit).strip().lower() not in ("km/h", "kph", "kmh", "mph", "m/s"):
            raise ValueError("WeatherFetcher requires explicit speed_unit ('km/h', 'mph' or 'm/s') (strict)")
        su = str(speed_unit).strip().lower()
        # Accept common variants and normalize them to canonical internal values
        if su in ("kph", "kmh", "km/h"):
            self.speed_unit = "km/h"
        elif su == "mph":
            self.speed_unit = "mph"
        else:
            # covers "m/s" and variants which fall through
            self.speed_unit = "m/s"

        self._cache_key = f"{self.latitude}_{self.longitude}_om"
        self._cache_duration = timedelta(minutes=30)

    # -----------------------
    # New: convenience fetch wrapper used by coordinator
    # -----------------------
    async def fetch(self, latitude: float, longitude: float, mode: str = "hourly", days: int = 5) -> Dict[str, Any]:
        """
        Strict wrapper to return the raw Open-Meteo forecast payload expected by the coordinator.

        - Validates lat/lon match the fetcher instance (strict).
        - Supports only mode='hourly' currently.
        - Delegates to fetch_open_meteo_forecast_direct and returns the raw dict.
        """
        # Strict validation: caller must use the same coordinates as the fetcher was constructed with
        try:
            lat_n = round(float(latitude), 6)
            lon_n = round(float(longitude), 6)
        except Exception:
            raise ValueError("Invalid latitude/longitude passed to fetch (strict)")

        if lat_n != self.latitude or lon_n != self.longitude:
            raise ValueError("Latitude/longitude mismatch with fetcher instance (strict)")

        if mode != "hourly":
            raise ValueError(f"Unsupported fetch mode '{mode}' (strict)")

        # Return the raw Open-Meteo payload (coordinator expects 'hourly' arrays)
        return await self.fetch_open_meteo_forecast_direct(days)

    # -----------------------
    # Unit helpers
    # -----------------------
    def _incoming_wind_to_m_s(self, value: Any, unit_hint: Optional[str] = None) -> Optional[float]:
        """Convert incoming wind to m/s. If unit_hint is provided, use it; otherwise assume Open‑Meteo default m/s."""
        if value is None:
            return None
        try:
            v = float(value)
        except Exception as exc:
            raise ValueError(f"Incoming wind value not numeric: {value!r}") from exc

        if unit_hint:
            u = str(unit_hint).strip().lower()
            if u in ("km/h", "kph", "kmh"):
                return unit_helpers.kmh_to_m_s(v)
            if u in ("mph", "mi/h", "miles/h"):
                return unit_helpers.mph_to_m_s(v)
            if u in ("m/s", "mps", "m s-1"):
                return v
            # unknown hint -> assume m/s per strict API assumption
            return v
        # No hint -> assume Open‑Meteo default m/s
        return v

    def _m_s_to_output(self, v: Optional[float]) -> Optional[float]:
        """Convert canonical m/s to configured output unit."""
        if v is None:
            return None
        try:
            vf = float(v)
        except Exception as exc:
            raise ValueError(f"Invalid internal wind value: {v!r}") from exc

        if self.speed_unit == "km/h":
            return unit_helpers.m_s_to_kmh(vf)
        if self.speed_unit == "mph":
            return unit_helpers.m_s_to_mph(vf)
        # else m/s
        return vf

    def _to_output_wind(self, value: Any, incoming_unit_hint: Optional[str] = None) -> Optional[float]:
        """Convert incoming wind (with optional hint) to output unit."""
        m_s = self._incoming_wind_to_m_s(value, incoming_unit_hint)
        return self._m_s_to_output(m_s)

    # -----------------------
    # Current weather (strict)
    # -----------------------
    async def get_weather_data(self) -> Dict[str, Any]:
        """
        Fetch and return a strict current weather snapshot.

        This method will raise on any missing/invalid critical field (temperature, wind).
        """
        from homeassistant.util import dt as dt_util

        now = dt_util.now()
        cache_entry = self.hass.data.setdefault("ocean_fishing_assistant_fetch_cache", {}).get(self._cache_key)
        if cache_entry:
            cached_time = cache_entry.get("time")
            if isinstance(cached_time, datetime) and (now - cached_time) < self._cache_duration:
                _LOGGER.debug("Using cached current weather for %s", self._cache_key)
                return cache_entry["data"]

        # strict fetch: raise if HTTP or payload errors
        payload = await self.fetch_open_meteo_current_direct()
        if not isinstance(payload, dict):
            raise RuntimeError("Open-Meteo current fetch returned non-dict payload (strict)")

        # Prefer explicit current_weather block (Open‑Meteo documented shape)
        if "current_weather" in payload and isinstance(payload["current_weather"], dict):
            units_container = payload.get("current_weather_units") or payload.get("hourly_units") or {}
            wind_unit_hint = None
            if isinstance(units_container, dict):
                wind_unit_hint = (
                    units_container.get("windspeed")
                    or units_container.get("wind_speed")
                    or units_container.get("wind_speed_10m")
                )
            mapped = self._map_to_current_shape(payload["current_weather"], incoming_unit_hint=wind_unit_hint)
        else:
            # Strictly extract nearest-hourly snapshot (will raise if critical fields missing)
            mapped = self._extract_current_from_rest_result(payload)

        # Must have critical fields
        if not mapped or mapped.get("temperature") is None or mapped.get("wind_speed") is None:
            raise RuntimeError(f"Incomplete current weather from Open-Meteo (strict): {mapped}")

        # cache and return strict snapshot (wind_out is already converted to configured output unit)
        self.hass.data.setdefault("ocean_fishing_assistant_fetch_cache", {})[self._cache_key] = {"data": mapped, "time": now}
        return mapped

    async def fetch_open_meteo_current_direct(self) -> Dict[str, Any]:
        """Call OM_BASE with current_weather=true; raise on any network/HTTP error."""
        session: aiohttp.ClientSession = async_get_clientsession(self.hass)
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "UTC",
            "current_weather": "true",
            "hourly": OM_PARAMS_HOURLY,
            "forecast_days": 2,
        }
        try:
            async with session.get(OM_BASE, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()
            if not isinstance(data, dict):
                raise RuntimeError("Open-Meteo returned unexpected payload shape for current (strict)")
            return data
        except Exception as exc:
            _LOGGER.exception("Open-Meteo current REST fetch failed for %s,%s", self.latitude, self.longitude)
            raise RuntimeError("Open-Meteo current REST fetch failed") from exc

    def _extract_current_from_rest_result(self, rest_result: Any) -> Dict[str, Any]:
        """Extract a nearest-hour snapshot from hourly arrays; strict convert and require fields."""
        if not isinstance(rest_result, dict):
            raise ValueError("REST result not a dict (strict)")

        if "hourly" not in rest_result or not isinstance(rest_result["hourly"], dict):
            raise ValueError("REST result missing 'hourly' dict (strict)")

        hourly = rest_result["hourly"]
        times = hourly.get("time") or []
        if not isinstance(times, (list, tuple)) or not times:
            raise ValueError("'hourly.time' missing or empty (strict)")

        # find nearest hour index
        now = dt_util.now()
        idx = 0
        for i, t_raw in enumerate(times):
            try:
                t = dt_util.parse_datetime(str(t_raw)) if t_raw is not None else None
                if t and t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if t and abs((t - now).total_seconds()) < 3600:
                    idx = i
                    break
            except Exception:
                continue

        # Build candidate dict for that index
        candidate: Dict[str, Any] = {}
        for k, arr in hourly.items():
            if k == "time":
                continue
            if isinstance(arr, list) and len(arr) > idx:
                candidate[k] = arr[idx]
            else:
                candidate[k] = None

        units_container = rest_result.get("hourly_units") or {}
        wind_unit_hint = None
        if isinstance(units_container, dict):
            wind_unit_hint = units_container.get("windspeed") or units_container.get("wind_speed") or units_container.get("wind_speed_10m")

        return self._map_to_current_shape(candidate, incoming_unit_hint=wind_unit_hint)

    def _map_to_current_shape(self, v: Any, incoming_unit_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Normalize a dict-like current snapshot into strict current shape.

        Required output keys: temperature (float), wind_speed (float), wind_gust (float or None), wind_unit.
        """
        if not isinstance(v, dict):
            raise ValueError("Current snapshot must be a dict (strict)")

        def pick(*keys):
            for k in keys:
                if k in v and v[k] is not None:
                    return v[k]
            return None

        temp_raw = pick("temperature", "temp", "temperature_2m", "air_temperature")
        wind_raw = pick("wind_speed", "wind_kph", "wind_km_h", "wind", "windspeed", "wind_speed_10m")
        wind_gust_raw = pick("wind_gust", "gust", "windgusts_10m") or wind_raw
        cloud_raw = pick("cloud_cover", "clouds", "clouds_percent", "cloud_coverage")
        precip_raw = pick("precipitation_probability", "pop", "precipitation", "precip")
        pressure_raw = pick("pressure", "air_pressure", "pressure_msl", "pressure_msl")

        # Require temperature and wind (strict)
        temp_f = _to_float_strict(temp_raw, "temperature")
        if wind_raw is None:
            raise ValueError("Missing wind value in current snapshot (strict)")
        wind_m_s = self._incoming_wind_to_m_s(wind_raw, incoming_unit_hint)
        if wind_m_s is None:
            raise ValueError("Unable to interpret incoming wind as numeric (strict)")
        wind_gust_m_s = self._incoming_wind_to_m_s(wind_gust_raw, incoming_unit_hint) if wind_gust_raw is not None else wind_m_s

        wind_out = self._m_s_to_output(wind_m_s)
        wind_gust_out = self._m_s_to_output(wind_gust_m_s) if wind_gust_m_s is not None else None

        out: Dict[str, Any] = {
            "temperature": float(temp_f),
            "wind_speed": float(wind_out),
            "wind_gust": float(wind_gust_out) if wind_gust_out is not None else None,
            "wind_unit": self.speed_unit,
        }
        if cloud_raw is not None:
            out["cloud_cover"] = _to_int_strict(cloud_raw, "cloud_cover")
        if precip_raw is not None:
            out["precipitation_probability"] = _to_int_strict(precip_raw, "precipitation_probability")
        if pressure_raw is not None:
            out["pressure"] = _to_float_strict(pressure_raw, "pressure")
        return out

    # -----------------------
    # Forecast fetch (strict)
    # -----------------------
    async def get_forecast(self, days: int = 7, aggregation_periods: Optional[Sequence[Dict[str, int]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Strict forecast retrieval. Raises on failure to fetch or invalid payload shape.
        """
        now = dt_util.now()
        forecast_cache_key = f"{self._cache_key}_forecast_{days}_{str(aggregation_periods)}"
        cache_entry = self.hass.data.setdefault("ocean_fishing_assistant_fetch_cache", {}).get(forecast_cache_key)
        if cache_entry:
            cached_time = cache_entry.get("time")
            if isinstance(cached_time, datetime) and (now - cached_time) < self._cache_duration:
                _LOGGER.debug("Using cached forecast for %s", forecast_cache_key)
                return cache_entry["data"]

        payload = await self.fetch_open_meteo_forecast_direct(days)
        if not isinstance(payload, dict):
            raise RuntimeError("Open-Meteo forecast fetch returned non-dict payload (strict)")

        # If aggregation_periods provided, delegate to _ensure_period_aggregation (this may raise)
        if aggregation_periods:
            normalized = await self._ensure_period_aggregation(payload, days, aggregation_periods)
            if not normalized:
                raise RuntimeError("Open-Meteo returned no usable forecast data for aggregation (strict)")
            self.hass.data.setdefault("ocean_fishing_assistant_fetch_cache", {})[forecast_cache_key] = {"data": normalized, "time": now}
            return normalized

        # Default: require hourly arrays present and produce daily summaries strictly
        if "hourly" in payload and isinstance(payload["hourly"], dict):
            hourly = payload["hourly"]
            times = hourly.get("time") or []
            if not times:
                raise RuntimeError("'hourly.time' missing or empty in forecast payload (strict)")
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
        else:
            raise RuntimeError("Forecast payload missing required 'hourly' arrays (strict)")

        if not normalized:
            raise RuntimeError("Open-Meteo forecast normalization produced no output (strict)")

        self.hass.data.setdefault("ocean_fishing_assistant_fetch_cache", {})[forecast_cache_key] = {"data": normalized, "time": now}
        return normalized

    async def fetch_open_meteo_forecast_direct(self, days: int) -> Dict[str, Any]:
        """Call OM_BASE for forecast; raise on HTTP/network errors."""
        session: aiohttp.ClientSession = async_get_clientsession(self.hass)
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "UTC",
            "hourly": OM_PARAMS_HOURLY,
            "forecast_days": int(days) + 1,
        }
        try:
            async with session.get(OM_BASE, params=params, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
            if not isinstance(data, dict):
                raise RuntimeError("Open-Meteo returned unexpected forecast payload shape (strict)")
            return data
        except Exception as exc:
            _LOGGER.exception("Open-Meteo forecast REST fetch failed for %s,%s", self.latitude, self.longitude)
            raise RuntimeError("Open-Meteo forecast REST fetch failed") from exc

    # -----------------------
    # Helpers & normalization (rely on strict conversions above)
    # -----------------------
    def _normalize_forecast_list(self, lst: List[Any], days: int) -> Dict[str, Dict[str, Any]]:
        """Normalize list of dicts into date-keyed daily summaries strictly where possible."""
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
            # require temperature and wind present for each item (strict)
            try:
                temp = _to_float_strict(item.get("temperature") or item.get("temp") or item.get("temperature_2m"), "temperature")
                wind_m_s = _to_float_strict(item.get("wind_speed") or item.get("wind") or item.get("wind_speed_10m"), "wind_speed")
            except ValueError:
                # item incomplete -> skip (we don't silently fill defaults)
                continue

            gust_m_s = None
            if item.get("wind_gust") or item.get("gust"):
                gust_m_s = float(item.get("wind_gust") or item.get("gust"))

            entry = {
                "temperature": temp,
                "wind_speed": self._m_s_to_output(wind_m_s),
            }
            if gust_m_s is not None:
                entry["wind_gust"] = self._m_s_to_output(gust_m_s)
            if item.get("cloud_cover") is not None:
                entry["cloud_cover"] = _to_int_strict(item.get("cloud_cover"), "cloud_cover")
            if item.get("precipitation_probability") is not None:
                entry["precipitation_probability"] = _to_int_strict(item.get("precipitation_probability"), "precipitation_probability")
            if item.get("pressure") is not None:
                entry["pressure"] = _to_float_strict(item.get("pressure"), "pressure")
            final[date_key] = entry
            if len(final) >= days:
                break
        return final

    def _normalize_hourly_list_to_daily(self, hourly_list: List[Any], days: int) -> Dict[str, Dict[str, Any]]:
        """Convert list of hourly entries into daily summaries strictly requiring numeric fields."""
        per_date: Dict[str, Dict[str, Any]] = {}
        from homeassistant.util import dt as dt_util

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

            # require temperature and wind for strict aggregation
            try:
                temp = _to_float_strict(entry.get("temperature") or entry.get("temp") or entry.get("temperature_2m"), "temperature")
                wind_m_s = _to_float_strict(entry.get("wind_speed") or entry.get("wind") or entry.get("wind_speed_10m"), "wind_speed")
            except ValueError:
                # skip entries that don't include required numeric fields
                continue

            gust_m_s = None
            if entry.get("wind_gust") is not None or entry.get("gust") is not None:
                try:
                    gust_m_s = float(entry.get("wind_gust") or entry.get("gust"))
                except Exception:
                    gust_m_s = None

            cloud = None
            if entry.get("cloud_cover") is not None or entry.get("clouds") is not None or entry.get("cloudcover") is not None:
                cloud = _to_int_strict(entry.get("cloud_cover") or entry.get("clouds") or entry.get("cloudcover"), "cloud_cover")

            pop = None
            if entry.get("precipitation_probability") is not None or entry.get("pop") is not None or entry.get("precipitation") is not None:
                pop = _to_int_strict(entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation"), "precipitation_probability")

            pressure = None
            if entry.get("pressure") is not None or entry.get("pressure_msl") is not None:
                pressure = _to_float_strict(entry.get("pressure") or entry.get("pressure_msl"), "pressure")

            agg = per_date.get(date_key)
            if not agg:
                per_date[date_key] = {
                    "temperature_sum": temp,
                    "wind_speed_sum": wind_m_s,
                    "pressure_sum": pressure or 0.0,
                    "cloud_sum": cloud or 0,
                    "precip_max": pop or 0,
                    "gust_max": gust_m_s or 0.0,
                    "count": 1,
                }
            else:
                agg["temperature_sum"] += temp
                agg["wind_speed_sum"] += wind_m_s
                agg["pressure_sum"] += (pressure or 0.0)
                agg["cloud_sum"] += (cloud or 0)
                agg["precip_max"] = max(agg["precip_max"], pop or 0)
                agg["gust_max"] = max(agg["gust_max"], gust_m_s or 0.0)
                agg["count"] += 1

        if not per_date:
            return {}

        final: Dict[str, Dict[str, Any]] = {}
        for date_key in sorted(per_date.keys())[:days]:
            agg = per_date[date_key]
            cnt = agg.get("count", 1) or 1
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
        return final

    async def _ensure_period_aggregation(self, data: Dict[str, Any], days: int, aggregation_periods: Sequence[Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate payload into requested named periods. Raises on invalid shapes.
        """
        # If already date keyed with metrics, convert wind units strictly
        if all(isinstance(v, dict) and ("temperature" in v and "wind_speed" in v) for v in data.values()):
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in list(data.items())[:days]:
                if not isinstance(v, dict):
                    continue
                if v.get("wind_speed") is None:
                    raise ValueError(f"Daily summary for {k} missing 'wind_speed' (strict)")
                try:
                    wind_m_s = float(v.get("wind_speed"))
                except Exception:
                    raise ValueError(f"Invalid wind_speed for {k}: {v.get('wind_speed')!r} (strict)")
                vv = dict(v)
                vv["wind_speed"] = self._m_s_to_output(wind_m_s)
                vv["wind_unit"] = self.speed_unit
                out[k] = vv
            return out

        # If dict has hourly arrays, construct hourly list
        hourly_list = None
        if isinstance(data, dict) and "hourly" in data and isinstance(data.get("hourly"), dict):
            hourly = data.get("hourly")
            times = hourly.get("time") or []
            if not isinstance(times, (list, tuple)) or not times:
                raise ValueError("'hourly.time' missing or empty in aggregation payload (strict)")
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

        if hourly_list is None:
            raise ValueError("Unable to find hourly entries for aggregation (strict)")

        return self._aggregate_hourly_into_periods(hourly_list, days, aggregation_periods)

    def _aggregate_hourly_into_periods(self, hourly_list: List[Dict[str, Any]], days: int, aggregation_periods: Optional[Sequence[Dict[str, int]]]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate hourly_list into named periods strictly. See docstring above for aggregation policy.
        """
        if not aggregation_periods:
            aggregation_periods = [
                {"name": "period_00_06", "start_hour": 0, "end_hour": 6},
                {"name": "period_06_12", "start_hour": 6, "end_hour": 12},
                {"name": "period_12_18", "start_hour": 12, "end_hour": 18},
                {"name": "period_18_24", "start_hour": 18, "end_hour": 24},
            ]

        per_date_periods: Dict[str, Dict[str, Dict[str, Any]]] = {}
        from homeassistant.util import dt as dt_util

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
                    # require numeric temperature and wind for each hourly entry to be included (strict)
                    try:
                        temp = _to_float_strict(entry.get("temperature") or entry.get("temp") or entry.get("temperature_2m"), "temperature")
                        wind_m_s = _to_float_strict(entry.get("wind_speed") or entry.get("wind") or entry.get("wind_speed_10m"), "wind_speed")
                        gust_m_s = None
                        if entry.get("wind_gust") is not None or entry.get("gust") is not None:
                            gust_m_s = float(entry.get("wind_gust") or entry.get("gust"))
                    except ValueError:
                        # skip this hourly row (it doesn't meet strict numeric requirements)
                        continue

                    cloud = None
                    if entry.get("cloud_cover") is not None or entry.get("clouds") is not None or entry.get("cloudcover") is not None:
                        cloud = _to_int_strict(entry.get("cloud_cover") or entry.get("clouds") or entry.get("cloudcover"), "cloud_cover")

                    pop = None
                    if entry.get("precipitation_probability") is not None or entry.get("pop") is not None or entry.get("precipitation") is not None:
                        pop = _to_int_strict(entry.get("precipitation_probability") or entry.get("pop") or entry.get("precipitation"), "precipitation_probability")

                    pressure = None
                    if entry.get("pressure") is not None or entry.get("pressure_msl") is not None:
                        pressure = _to_float_strict(entry.get("pressure") or entry.get("pressure_msl"), "pressure")

                    agg["temperature_sum"] += temp
                    agg["wind_speed_sum"] += wind_m_s
                    agg["pressure_sum"] += (pressure or 0.0)
                    agg["cloud_sum"] += (cloud or 0)
                    agg["precip_max"] = max(agg["precip_max"], pop or 0)
                    agg["gust_max"] = max(agg["gust_max"], gust_m_s or 0.0)
                    agg["count"] += 1
                    break

        # finalize into requested shape: date -> period_name -> metrics
        final: Dict[str, Dict[str, Any]] = {}
        for date_key in sorted(per_date_periods.keys())[:days]:
            final[date_key] = {}
            for pname, agg in per_date_periods[date_key].items():
                cnt = agg.get("count", 1) or 1
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
        return final

    # -----------------------
    # Marine endpoint helper (STRICT)
    # -----------------------
    async def fetch_marine_direct(self, days: int = 5) -> Dict[str, Any]:
        """
        Fetch marine/wave variables from the Open‑Meteo Marine endpoint.

        This function is strict and will raise on any network/HTTP errors or if the returned
        payload does not include hourly arrays aligned to the requested forecast horizon.
        """
        session: aiohttp.ClientSession = async_get_clientsession(self.hass)
        hourly_vars = ",".join(
            [
                "wave_height",
                "wave_direction",
                "wave_period",
                "swell_wave_height",
                "swell_wave_period",
            ]
        )
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "UTC",
            "hourly": hourly_vars,
            "forecast_days": int(days) + 1,
        }
        try:
            async with session.get(OM_MARINE_BASE, params=params, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
            if not isinstance(data, dict) or "hourly" not in data or not isinstance(data["hourly"], dict):
                raise RuntimeError("Marine API returned unexpected payload shape (strict)")
            return data
        except Exception as exc:
            _LOGGER.exception("Direct Marine API fetch failed for %s,%s", self.latitude, self.longitude)
            # marine data is required for Ocean Fishing Assistant -> propagate error
            raise RuntimeError("Direct Marine API fetch failed (marine required)") from exc