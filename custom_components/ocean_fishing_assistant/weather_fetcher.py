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
from typing import Any, Dict, List, Optional

import aiohttp

from homeassistant.util import dt as dt_util
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import OM_BASE, OM_MARINE_BASE, WEATHER_FETCHER_CACHE_TTL_DEFAULT

from . import unit_helpers

_LOGGER = logging.getLogger(__name__)

# hourly params to request from Open-Meteo (we require these fields to be present)
OM_PARAMS_HOURLY = ",".join(
    [
        "temperature_2m",
        "wind_speed_10m",
        "wind_direction_10m",  # wind direction in degrees (0-359)
        "windgusts_10m",
        "cloudcover",
        "precipitation_probability",
        "pressure_msl",
        "visibility",  # <-- added strict visibility parameter
    ]
)

# marine-specific hourly parameters we attach into the raw['hourly'] payload
OM_MARINE_PARAMS_HOURLY = ",".join(
    [
        "wave_height",
        "wave_direction",
        "wave_period",
        "swell_wave_height",
        "swell_wave_period",
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
    cache_ttl_seconds: controls in-memory cache duration for get_weather_data()/get_forecast() (seconds).
    """

    def __init__(self, hass, latitude: float, longitude: float, speed_unit: str, cache_ttl_seconds: Optional[int] = None) -> None:
        self.hass = hass
        # Use centralized rounding precision for stored instance coords
        lat_r, lon_r = unit_helpers.round_coords(latitude, longitude)
        self.latitude = lat_r
        self.longitude = lon_r

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

        # canonical, deterministic cache key using centralized formatting (string)
        self._cache_key = unit_helpers.format_coord_key(self.latitude, self.longitude) + "_om"
        # Allow external override via constructor; otherwise use integration default
        if cache_ttl_seconds is None:
            self._cache_duration = timedelta(seconds=WEATHER_FETCHER_CACHE_TTL_DEFAULT)
        else:
            try:
                self._cache_duration = timedelta(seconds=int(cache_ttl_seconds))
            except Exception:
                self._cache_duration = timedelta(seconds=WEATHER_FETCHER_CACHE_TTL_DEFAULT)

    # -----------------------
    # Convenience async fetch wrapper used by coordinator
    # -----------------------
    async def fetch(self, latitude: float, longitude: float, mode: str = "hourly", days: int = 5) -> Dict[str, Any]:
        """
        Strict wrapper to return the raw Open-Meteo forecast payload expected by the coordinator.

        - Validates lat/lon match the fetcher instance (strict).
        - Supports only mode='hourly' currently.
        - Delegates to fetch_open_meteo_forecast_direct and returns the raw dict.
        """
        try:
            lat_n, lon_n = unit_helpers.round_coords(latitude, longitude)
        except Exception:
            raise ValueError("Invalid latitude/longitude passed to fetch (strict)")

        if lat_n != self.latitude or lon_n != self.longitude:
            raise ValueError("Latitude/longitude mismatch with fetcher instance (strict)")

        if mode != "hourly":
            raise ValueError(f"Unsupported fetch mode '{mode}' (strict)")

        # Fetch raw Open-Meteo forecast (coordinator expects 'hourly' arrays)
        session: aiohttp.ClientSession = async_get_clientsession(self.hass)
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "UTC",
            "hourly": OM_PARAMS_HOURLY,
            "forecast_days": int(days) + 1,
        }
        try:
            async with session.get(OM_BASE, params=params, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
            if not isinstance(data, dict):
                raise RuntimeError("Open-Meteo returned unexpected payload shape (strict)")
            return data
        except Exception as exc:
            _LOGGER.exception("Open-Meteo fetch failed for %s,%s", latitude, longitude)
            raise RuntimeError("Open-Meteo fetch failed") from exc

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

        This method will raise on any missing/invalid critical field (temperature, wind, gust, cloud, pop, pressure).

        NOTE: Always constructs the current snapshot from hourly arrays (nearest-hour).
        It will also attempt to fetch marine hourly arrays and merge in current wave/swell values (strict).
        """
        now = dt_util.now()
        cache_entry = self.hass.data.setdefault("ocean_fishing_assistant_fetch_cache", {}).get(self._cache_key)
        if cache_entry:
            cached_time = cache_entry.get("time")
            if isinstance(cached_time, datetime) and (now - cached_time) < self._cache_duration:
                _LOGGER.debug("Using cached current weather for %s", self._cache_key)
                return cache_entry["data"]

        # strict fetch: raise if HTTP or payload errors
        # We fetch both the main hourly payload and marine endpoint to build a complete strict snapshot.
        payload = await self.fetch_open_meteo_current_direct()
        marine_payload = await self.fetch_marine_direct(days=2)

        if not isinstance(payload, dict):
            raise RuntimeError("Open-Meteo current fetch returned non-dict payload (strict)")
        if not isinstance(marine_payload, dict) or "hourly" not in marine_payload:
            raise RuntimeError("Open-Meteo marine fetch returned invalid payload (strict)")

        # Always derive current snapshot from hourly arrays (strict), merging marine values where available
        mapped = self._extract_current_from_rest_result(payload, marine_payload=marine_payload)

        # Must have critical fields
        required = ("temperature", "wind_speed", "wind_gust", "cloud_cover", "precipitation_probability", "pressure")
        missing = [k for k in required if mapped.get(k) is None]
        if missing:
            raise RuntimeError(f"Incomplete current weather from Open-Meteo (strict), missing: {missing}; mapped={mapped}")

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

    def _extract_current_from_rest_result(self, rest_result: Any, marine_payload: Optional[Any] = None) -> Dict[str, Any]:
        """Extract a nearest-hour snapshot from hourly arrays; strict convert and require fields.
        If marine_payload provided, merge corresponding marine values at the same nearest-hour index.
        """
        if not isinstance(rest_result, dict):
            raise ValueError("REST result not a dict (strict)")

        if "hourly" not in rest_result or not isinstance(rest_result["hourly"], dict):
            raise ValueError("REST result missing 'hourly' dict (strict)")

        hourly = rest_result["hourly"]
        times = hourly.get("time") or []
        if not isinstance(times, (list, tuple)) or not times:
            raise ValueError("'hourly.time' missing or empty (strict)")

        # find nearest hour index (prefer exact nearest timestamp within 1 hour)
        now = dt_util.now()
        idx = 0
        best_delta = None
        for i, t_raw in enumerate(times):
            try:
                t = dt_util.parse_datetime(str(t_raw)) if t_raw is not None else None
                if t and t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if t:
                    delta = abs((t - now).total_seconds())
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        idx = i
            except Exception:
                continue

        # Build candidate dict for that index from main hourly payload
        candidate: Dict[str, Any] = {}
        for k, arr in hourly.items():
            if k == "time":
                continue
            if isinstance(arr, list) and len(arr) > idx:
                candidate[k] = arr[idx]
            else:
                candidate[k] = None

        # Merge marine values at same index if marine payload provided and aligned
        if marine_payload and isinstance(marine_payload, dict) and isinstance(marine_payload.get("hourly"), dict):
            m_hourly = marine_payload["hourly"]
            # if marine timestamps exists and matches length, prefer picking by same index; otherwise conservatively pick if present
            for mk in ("wave_height", "wave_period", "swell_wave_height", "swell_wave_period"):
                m_arr = m_hourly.get(mk)
                if isinstance(m_arr, list) and len(m_arr) > idx:
                    candidate[mk] = m_arr[idx]
                else:
                    # leave candidate[mk] as-is (do not fallback); if absent, it's fine — not required strictly
                    pass

        units_container = rest_result.get("hourly_units") or {}
        wind_unit_hint = None
        if isinstance(units_container, dict):
            # Open-Meteo uses "windspeed_10m" or similar labels in hourly_units; be permissive but strict about presence
            wind_unit_hint = units_container.get("windspeed") or units_container.get("wind_speed") or units_container.get("wind_speed_10m") or units_container.get("wind_speed_10m_aggregate")

        return self._map_to_current_shape(candidate, incoming_unit_hint=wind_unit_hint)

    def _map_to_current_shape(self, v: Any, incoming_unit_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Normalize a dict-like current snapshot into strict current shape.

        Required output keys: temperature (float), wind_speed (float), wind_gust (float), cloud_cover (int),
        precipitation_probability (int), pressure (float), wind_unit.
        Additional optional keys: visibility_km, wave_height, wave_period, swell_wave_height, swell_wave_period.
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
        # strict: do NOT fallback to wind_raw for gust — gust must be present
        wind_gust_raw = pick("wind_gust", "gust", "windgusts_10m")
        cloud_raw = pick("cloud_cover", "cloudcover", "clouds", "clouds_percent", "cloud_coverage")
        precip_raw = pick("precipitation_probability", "pop", "precipitation", "precip")
        pressure_raw = pick("pressure", "air_pressure", "pressure_msl", "pressure_mean_sea_level", "msl_pressure")

        # Visibility (optional but strictly converted if present). Unit hint may be available under 'visibility' in hourly_units.
        visibility_raw = pick("visibility", "visibility_m", "visibility_km")

        # Marine current candidates (optional)
        wave_height_raw = pick("wave_height", "wave_height_m")
        wave_period_raw = pick("wave_period", "wave_period_s")
        swell_height_raw = pick("swell_wave_height", "swell_height_m")
        swell_period_raw = pick("swell_wave_period", "swell_period_s")

        # Require temperature and wind (strict)
        temp_f = _to_float_strict(temp_raw, "temperature")
        if wind_raw is None:
            raise ValueError("Missing wind value in current snapshot (strict)")
        wind_m_s = self._incoming_wind_to_m_s(wind_raw, incoming_unit_hint)
        if wind_m_s is None:
            raise ValueError("Unable to interpret incoming wind as numeric (strict)")

        # Require gust (strict)
        if wind_gust_raw is None:
            raise ValueError("Missing wind_gust in current snapshot (strict)")
        wind_gust_m_s = self._incoming_wind_to_m_s(wind_gust_raw, incoming_unit_hint)
        if wind_gust_m_s is None:
            raise ValueError("Unable to interpret incoming wind_gust as numeric (strict)")

        # Require cloud cover (strict)
        if cloud_raw is None:
            raise ValueError("Missing cloud_cover in current snapshot (strict)")
        cloud_i = _to_int_strict(cloud_raw, "cloud_cover")

        # Require precipitation_probability (strict)
        if precip_raw is None:
            raise ValueError("Missing precipitation_probability in current snapshot (strict)")
        precip_i = _to_int_strict(precip_raw, "precipitation_probability")

        # Require pressure (strict)
        if pressure_raw is None:
            raise ValueError("Missing pressure in current snapshot (strict)")
        pressure_f = _to_float_strict(pressure_raw, "pressure")

        wind_out = self._m_s_to_output(wind_m_s)
        wind_gust_out = self._m_s_to_output(wind_gust_m_s)

        out: Dict[str, Any] = {
            "temperature": float(temp_f),
            "wind_speed": float(wind_out),
            "wind_gust": float(wind_gust_out),
            "wind_unit": self.speed_unit,
            "cloud_cover": int(cloud_i),
            "precipitation_probability": int(precip_i),
            "pressure": float(pressure_f),
        }

        # Strictly convert visibility into km if present. If unit cannot be determined, assume meters (strict convert).
        if visibility_raw is not None:
            try:
                vis_val = float(visibility_raw)
                # If caller supplied a 'visibility' with unit hint in the candidate dict it's not available here;
                # we make a strict assumption: Open-Meteo visibility is meters. Convert to km.
                # (DataFormatter enforces visibility units for hourly arrays; here we are strict but practical.)
                vis_km = vis_val / 1000.0
                out["visibility_km"] = float(vis_km)
            except Exception:
                # do not fallback silently; raise to indicate mapping failure if present but invalid
                raise ValueError(f"Invalid visibility value in current snapshot: {visibility_raw!r}")

        # Map marine current values into snapshot if present (do not require them strictly)
        if wave_height_raw is not None:
            try:
                out["wave_height"] = float(wave_height_raw)
            except Exception:
                raise ValueError(f"Invalid wave_height in current snapshot: {wave_height_raw!r}")
        if wave_period_raw is not None:
            try:
                out["wave_period"] = float(wave_period_raw)
            except Exception:
                raise ValueError(f"Invalid wave_period in current snapshot: {wave_period_raw!r}")
        if swell_height_raw is not None:
            try:
                out["swell_wave_height"] = float(swell_height_raw)
            except Exception:
                raise ValueError(f"Invalid swell_wave_height in current snapshot: {swell_height_raw!r}")
        if swell_period_raw is not None:
            try:
                out["swell_wave_period"] = float(swell_period_raw)
            except Exception:
                raise ValueError(f"Invalid swell_wave_period in current snapshot: {swell_period_raw!r}")

        return out

    # -----------------------
    # Marine fetch (strict)
    # -----------------------
    async def fetch_marine_direct(self, days: int = 5) -> Dict[str, Any]:
        """
        Fetch marine-specific hourly arrays (waves/swell) from the marine endpoint.

        Raises on HTTP/network errors or if payload shape is invalid.
        """
        session: aiohttp.ClientSession = async_get_clientsession(self.hass)
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "UTC",
            "hourly": OM_MARINE_PARAMS_HOURLY,
            "forecast_days": int(days) + 1,
        }
        try:
            async with session.get(OM_MARINE_BASE, params=params, timeout=60) as resp:
                resp.raise_for_status()
                data = await resp.json()
            if not isinstance(data, dict):
                raise RuntimeError("Open-Meteo marine endpoint returned unexpected payload shape (strict)")
            # Expect 'hourly' dict with arrays
            if "hourly" not in data or not isinstance(data["hourly"], dict):
                raise RuntimeError("Open-Meteo marine payload missing required 'hourly' dict (strict)")
            return data
        except Exception as exc:
            _LOGGER.exception("Open-Meteo marine REST fetch failed for %s,%s", self.latitude, self.longitude)
            raise RuntimeError("Open-Meteo marine REST fetch failed") from exc

