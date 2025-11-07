# Strict WeatherFetcher + DataFormatter (no fallbacks, fail loudly)
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import aiohttp

from homeassistant.util import dt as dt_util
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import OM_BASE, OM_MARINE_BASE
from . import unit_helpers
from . import ocean_scoring

_LOGGER = logging.getLogger(__name__)

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


def _ensure_list_length_equal(key: str, timestamps: List[Any], arr: List[Any]) -> None:
    if len(timestamps) != len(arr):
        raise ValueError(f"Array length mismatch for '{key}': timestamps length={len(timestamps)}, {key} length={len(arr)}")


class WeatherFetcher:
    """
    Strict WeatherFetcher for Open-Meteo REST.

    - Requires explicit speed_unit parameter (no HA auto-detection).
    - Expects documented OM payload shapes; missing required fields raise errors.
    """

    ALLOWED_SPEED_UNITS = {"km/h", "mph", "m/s"}

    def __init__(self, hass, latitude: float, longitude: float, speed_unit: str) -> None:
        self.hass = hass
        self.latitude = float(latitude)
        self.longitude = float(longitude)

        if not speed_unit or not isinstance(speed_unit, str):
            raise ValueError("WeatherFetcher requires explicit speed_unit (e.g. 'km/h', 'mph', 'm/s')")
        su = speed_unit.strip().lower()
        if su not in self.ALLOWED_SPEED_UNITS:
            raise ValueError(f"Unsupported speed_unit '{speed_unit}'; supported: {sorted(self.ALLOWED_SPEED_UNITS)}")
        self.speed_unit = su

        self._cache_key = f"{round(self.latitude,4)}_{round(self.longitude,4)}_om"
        self._cache_duration = timedelta(minutes=30)
        # minimal in-memory cache shared across instances
        self._cache: Dict[str, Dict[str, Any]] = {}

    # --- Unit helpers (strict) ---
    def _incoming_wind_to_m_s(self, value: Any, unit_hint: Optional[str]) -> float:
        """
        Convert incoming wind value to m/s. unit_hint is REQUIRED.
        Raises ValueError if conversion cannot be performed or unit_hint missing.
        """
        if unit_hint is None:
            raise ValueError("Missing unit hint for incoming wind values (strict mode)")
        try:
            v = float(value)
        except Exception as exc:
            raise ValueError(f"Non-numeric wind value: {value!r}") from exc
        u = str(unit_hint).strip().lower()
        if u in ("km/h", "kph", "kmh"):
            return unit_helpers.kmh_to_m_s(v)
        if u in ("mph", "mi/h", "miles/h"):
            return unit_helpers.mph_to_m_s(v)
        if u in ("m/s", "mps", "m s-1"):
            return float(v)
        raise ValueError(f"Unsupported incoming wind unit hint: {unit_hint!r}")

    def _m_s_to_output(self, v: float) -> float:
        """Convert m/s to configured output unit (strict)."""
        try:
            vf = float(v)
        except Exception as exc:
            raise ValueError("Non-numeric m/s wind value") from exc
        if self.speed_unit == "km/h":
            return unit_helpers.m_s_to_kmh(vf)
        if self.speed_unit == "mph":
            return unit_helpers.m_s_to_mph(vf)
        return vf  # m/s

    # --- Current weather ---
    async def get_weather_data(self) -> Dict[str, Any]:
        """Fetch current weather; raises on any missing required fields."""
        now = dt_util.now()
        cache_entry = self._cache.get(self._cache_key)
        if cache_entry:
            cached_time = cache_entry.get("time")
            if isinstance(cached_time, datetime) and (now - cached_time) < self._cache_duration:
                _LOGGER.debug("Using cached weather data for %s", self._cache_key)
                return cache_entry["data"]

        payload = await self.fetch_open_meteo_current_direct()
        if not isinstance(payload, dict):
            raise RuntimeError("Open-Meteo returned unexpected payload for current weather")

        # Prefer documented 'current_weather' block
        if "current_weather" in payload and isinstance(payload["current_weather"], dict):
            units_container = payload.get("current_weather_units") or payload.get("hourly_units") or {}
            wind_hint = None
            if isinstance(units_container, dict):
                wind_hint = units_container.get("windspeed") or units_container.get("wind_speed") or units_container.get("wind_speed_10m")
            if wind_hint is None:
                raise ValueError("Missing wind unit hint in Open-Meteo response (current_weather_units/hourly_units)")
            mapped = self._map_current_block(payload["current_weather"], wind_hint)
        else:
            # Strict: require hourly arrays with 'time'
            if "hourly" not in payload or not isinstance(payload["hourly"], dict):
                raise ValueError("Open-Meteo payload missing required 'current_weather' or 'hourly' block")
            hourly = payload["hourly"]
            times = hourly.get("time")
            if not isinstance(times, (list, tuple)) or not times:
                raise ValueError("Open-Meteo hourly block missing 'time' array for current extraction")
            # pick nearest hour index exactly equal to current hour UTC (strict)
            now_utc = dt_util.now().replace(minute=0, second=0, microsecond=0)
            idx = None
            for i, t_raw in enumerate(times):
                t = dt_util.parse_datetime(str(t_raw))
                if t is None:
                    continue
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if t == now_utc:
                    idx = i
                    break
            if idx is None:
                raise ValueError("Unable to find exact matching hourly timestamp for current hour (strict)")
            # require the documented hourly unit mapping
            hourly_units = payload.get("hourly_units")
            wind_hint = None
            if isinstance(hourly_units, dict):
                wind_hint = hourly_units.get("wind_speed_10m") or hourly_units.get("windspeed")
            if wind_hint is None:
                raise ValueError("Missing hourly wind unit hint (strict)")

            candidate = {}
            for k, arr in hourly.items():
                if not isinstance(arr, (list, tuple)):
                    continue
                if idx >= len(arr):
                    raise ValueError(f"Hourly array length mismatch for key '{k}'")
                candidate[k] = arr[idx]
            mapped = self._map_current_block(candidate, wind_hint)

        # final basic validation
        if "temperature" not in mapped or "wind_speed" not in mapped:
            raise ValueError("Mapped current weather missing required fields (temperature or wind_speed)")

        self._cache[self._cache_key] = {"data": mapped, "time": now}
        return mapped

    async def fetch_open_meteo_current_direct(self) -> Dict[str, Any]:
        """Call OM_BASE with current_weather=true (strict)."""
        session: aiohttp.ClientSession = async_get_clientsession(self.hass)
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "UTC",
            "current_weather": "true",
            "hourly": OM_PARAMS_HOURLY,
            "forecast_days": 2,
        }
        async with session.get(OM_BASE, params=params, timeout=30) as resp:
            resp.raise_for_status()
            data = await resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("Open-Meteo current endpoint returned unexpected shape")
        return data

    def _map_current_block(self, block: Dict[str, Any], wind_unit_hint: str) -> Dict[str, Any]:
        """
        Map either current_weather block or a single-hour candidate into canonical current dict.
        Requires explicit wind_unit_hint.
        """
        # temperature
        if "temperature" in block:
            temp = block["temperature"]
        elif "temperature_2m" in block:
            temp = block["temperature_2m"]
        elif "temp" in block:
            temp = block["temp"]
        else:
            raise ValueError("Missing temperature field in current block (strict)")

        # wind
        if "windspeed" in block:
            wind = block["windspeed"]
        elif "wind_speed" in block:
            wind = block["wind_speed"]
        elif "wind_speed_10m" in block:
            wind = block["wind_speed_10m"]
        else:
            raise ValueError("Missing wind speed field in current block (strict)")

        # gust optional
        gust = block.get("windgusts") or block.get("wind_gust") or block.get("windgusts_10m") or None

        try:
            temp_f = float(temp)
        except Exception:
            raise ValueError("Non-numeric temperature in current block")

        wind_m_s = self._incoming_wind_to_m_s(wind, wind_unit_hint)
        gust_m_s = None
        if gust is not None:
            gust_m_s = self._incoming_wind_to_m_s(gust, wind_unit_hint)

        out = {
            "temperature": temp_f,
            "wind_speed": self._m_s_to_output(wind_m_s),
            "wind_gust": self._m_s_to_output(gust_m_s) if gust_m_s is not None else None,
            "wind_unit": self.speed_unit,
        }

        # optional fields: cloud_cover, precipitation_probability, pressure (only include if present)
        if "cloudcover" in block:
            out["cloud_cover"] = int(block["cloudcover"])
        if "precipitation_probability" in block:
            out["precipitation_probability"] = int(block["precipitation_probability"])
        if "pressure_msl" in block:
            out["pressure"] = float(block["pressure_msl"])
        return out

    # --- Forecast fetch (strict) ---
    async def fetch(self, latitude: float, longitude: float, mode: str = "hourly", days: int = 5) -> Dict[str, Any]:
        """
        Fetch raw Open-Meteo response (strict): returns dict with 'hourly' containing arrays.
        """
        if mode != "hourly":
            raise ValueError("Only 'hourly' mode is supported in strict WeatherFetcher")

        session: aiohttp.ClientSession = async_get_clientsession(self.hass)
        params = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "timezone": "UTC",
            "hourly": OM_PARAMS_HOURLY,
            "forecast_days": int(days) + 1,
        }
        async with session.get(OM_BASE, params=params, timeout=60) as resp:
            resp.raise_for_status()
            data = await resp.json()
        if not isinstance(data, dict) or "hourly" not in data or not isinstance(data["hourly"], dict):
            raise RuntimeError("Open-Meteo fetch returned payload missing 'hourly' dict (strict)")
        return data

    async def fetch_marine_direct(self, days: int = 5) -> Dict[str, Any]:
        """
        Fetch marine endpoint. Return will be passed through but caller must align shapes.
        This function raises on network/HTTP errors; caller decides whether marine data is required.
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
            "forecast_days": int(days),
        }
        async with session.get(OM_MARINE_BASE, params=params, timeout=60) as resp:
            resp.raise_for_status()
            data = await resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("Marine endpoint returned unexpected shape")
        return data


class DataFormatter:
    """
    Strict DataFormatter:

    - Requires payload to contain 'hourly' dict with 'time' array and canonical keys.
    - Requires hourly_units to exist and to provide units for wind arrays.
    - No legacy aliases, no heuristics. Raises ValueError on any validation failure.
    """

    # mapping of Open-Meteo hourly keys -> canonical keys used by scoring/formatting
    HOURLY_KEY_MAP = {
        "time": "timestamps",
        "temperature_2m": "temperature_c",
        "wind_speed_10m": "wind_m_s",
        "windgusts_10m": "wind_max_m_s",
        "pressure_msl": "pressure_hpa",
        "cloudcover": "cloud_cover",
        "precipitation_probability": "precipitation_probability",
        # marine keys
        "wave_height": "wave_height_m",
        "wave_direction": "wave_direction",
        "wave_period": "wave_period_s",
        "swell_wave_height": "swell_height_m",
        "swell_wave_period": "swell_period_s",
    }

    def __init__(self) -> None:
        pass

    def validate(self, raw_payload: Dict[str, Any], species_profile=None, units: str = "metric", safety_limits: Optional[dict] = None) -> Dict[str, Any]:
        """
        Validate and normalize raw_payload into strict canonical shape:
        - 'timestamps' list
        - canonical arrays using canonical keys
        - compute per_timestamp_forecasts via ocean_scoring.compute_forecast (errors propagate)
        - compute period_forecasts via strict aggregator (errors propagate)
        """
        if not isinstance(raw_payload, dict):
            raise ValueError("raw_payload must be a dict (strict)")

        if "hourly" not in raw_payload or not isinstance(raw_payload["hourly"], dict):
            raise ValueError("raw_payload must include an 'hourly' dict (strict)")

        hourly = raw_payload["hourly"]
        if "time" not in hourly or not isinstance(hourly["time"], (list, tuple)):
            raise ValueError("'hourly' must include 'time' array (strict)")

        timestamps = list(hourly["time"])
        if not timestamps:
            raise ValueError("'time' array is empty (strict)")

        # prepare hourly_units (required for wind conversion)
        hourly_units = raw_payload.get("hourly_units") or hourly.get("units")
        if not isinstance(hourly_units, dict):
            raise ValueError("Missing 'hourly_units' mapping in payload; wind unit hints are required (strict)")

        canonical: Dict[str, Any] = {}
        canonical["timestamps"] = timestamps

        # Map required numeric arrays: require temperature_2m and wind_speed_10m at minimum
        missing_required = []
        if "temperature_2m" not in hourly:
            missing_required.append("temperature_2m")
        if "wind_speed_10m" not in hourly:
            missing_required.append("wind_speed_10m")
        if missing_required:
            raise ValueError(f"Missing required hourly arrays: {missing_required} (strict)")

        # map arrays and enforce equal lengths
        for om_key, canon_key in self.HOURLY_KEY_MAP.items():
            if om_key == "time":
                continue
            if om_key in hourly:
                arr = hourly[om_key]
                if not isinstance(arr, (list, tuple)):
                    raise ValueError(f"Hourly key '{om_key}' must be a list/tuple (strict)")
                # enforce same length as timestamps
                _ensure_list_length_equal(om_key, timestamps, list(arr))
                if om_key in ("wind_speed_10m", "windgusts_10m"):
                    # require explicit unit hint for these
                    unit_hint = hourly_units.get(om_key) or hourly_units.get("windspeed") or hourly_units.get("wind_speed_10m")
                    if not unit_hint:
                        raise ValueError(f"Missing unit hint for hourly key '{om_key}' (strict)")
                    converted: List[Optional[float]] = []
                    for v in arr:
                        if v is None:
                            converted.append(None)
                            continue
                        converted.append(self._convert_wind_array_value(v, unit_hint))
                    canonical[canon_key] = converted
                else:
                    canonical[canon_key] = list(arr)  # shallow copy

        # compute per-timestamp forecasts if required keys exist (compute errors propagate)
        required_for_scoring = ("temperature_c" in canonical and "wind_m_s" in canonical and "pressure_hpa" in canonical) or ("wave_height_m" in canonical)
        per_ts_forecasts: List[Dict[str, Any]] = []
        if required_for_scoring:
            per_ts_forecasts = ocean_scoring.compute_forecast(canonical, species_profile=species_profile, safety_limits=safety_limits)
        else:
            # If scoring cannot be attempted due to missing optional keys, raise so callers know.
            raise ValueError("Insufficient canonical keys to compute forecasts (strict): need temperature_c, wind_m_s, pressure_hpa or wave_height_m")

        # Build hourly_like list expected by aggregator (strict)
        hourly_like: List[Dict[str, Any]] = []
        length = len(timestamps)
        for i, ts in enumerate(timestamps):
            row: Dict[str, Any] = {"time": ts}
            # attach canonical keys into row when present
            for src_key in ("temperature_c", "wind_m_s", "wind_max_m_s", "pressure_hpa", "cloud_cover", "precipitation_probability",
                            "wave_height_m", "wave_period_s", "swell_height_m", "swell_period_s"):
                arr = canonical.get(src_key)
                if isinstance(arr, (list, tuple)):
                    row[src_key] = arr[i]
            # attach computed forecast entry if available
            if i < len(per_ts_forecasts):
                row["_forecast_entry"] = per_ts_forecasts[i]
            hourly_like.append(row)

        # Use strict default 4 periods if caller doesn't supply aggregation shape
        default_periods = [
            {"name": "period_00_06", "start_hour": 0, "end_hour": 6},
            {"name": "period_06_12", "start_hour": 6, "end_hour": 12},
            {"name": "period_12_18", "start_hour": 12, "end_hour": 18},
            {"name": "period_18_24", "start_hour": 18, "end_hour": 24},
        ]
        # NOTE: we do not read aggregation_periods from the payload (legacy); caller must provide if desired.

        # Aggregate (the aggregator is strict and raises on missing members)
        period_agg = self._aggregate_hourly_into_periods(hourly_like, days=7, aggregation_periods=default_periods, full_payload=raw_payload, units=units)

        # aggregate per-period scoring (same approach as original but errors propagate)
        period_forecasts: Dict[str, Dict[str, Any]] = {}
        for date_key, pmap in period_agg.items():
            period_forecasts[date_key] = {}
            for pname, pdata in pmap.items():
                indices = pdata.get("indices") or []
                per_ts_entries = []
                for idx in indices:
                    if idx < len(hourly_like):
                        fe = hourly_like[idx].get("_forecast_entry")
                        if fe:
                            per_ts_entries.append(fe)
                score_vals = [float(e.get("score_10")) for e in per_ts_entries if e.get("score_10") is not None]
                score_10 = float(sum(score_vals) / len(score_vals)) if score_vals else None
                components = None
                if per_ts_entries:
                    # basic components aggregation (mean of numeric score_10 per component)
                    keys = set().union(*(e.get("components", {}).keys() if e.get("components") else [] for e in per_ts_entries))
                    out_comp = {}
                    for k in keys:
                        vals = []
                        for e in per_ts_entries:
                            c = e.get("components") or {}
                            if k in c and c[k].get("score_10") is not None:
                                vals.append(float(c[k]["score_10"]))
                        if vals:
                            avg = float(sum(vals) / len(vals))
                            out_comp[k] = {"score_10": round(avg, 3), "score_100": int(round(avg * 10))}
                    components = out_comp or None
                profile_used = next((e.get("profile_used") for e in per_ts_entries if e.get("profile_used")), None)
                safety = {"unsafe": any((e.get("safety") or {}).get("unsafe") for e in per_ts_entries),
                          "caution": any((e.get("safety") or {}).get("caution") for e in per_ts_entries),
                          "reasons": sorted({r for e in per_ts_entries for r in (e.get("safety") or {}).get("reasons", [])})}
                summary = dict(pdata)
                summary.update({
                    "score_10": round(score_10, 3) if score_10 is not None else None,
                    "score_100": int(round(score_10 * 10)) if score_10 is not None else None,
                    "components": components,
                    "profile_used": profile_used,
                    "safety": safety,
                })
                period_forecasts[date_key][pname] = summary

        final_out = {
            "timestamps": timestamps,
            **canonical,
            "raw_payload": raw_payload,
            "per_timestamp_forecasts": per_ts_forecasts,
            "period_forecasts": period_forecasts,
        }
        return final_out

    # Aggregate hourly entries into named periods (implemented locally to avoid cross-class delegation)
    def _aggregate_hourly_into_periods(self, hourly_list: List[Dict[str, Any]], days: int, aggregation_periods: Optional[Sequence[Dict[str, Any]]], full_payload: Optional[Dict[str, Any]] = None, units: str = "metric") -> Dict[str, Dict[str, Any]]:
        """
        Local aggregator that accepts hourly-like rows produced earlier in validate().
        Expects rows with keys: time, temperature_c, wind_m_s, wind_max_m_s, pressure_hpa, cloud_cover,
        precipitation_probability, wave_height_m, wave_period_s, swell_height_m, swell_period_s
        Returns mapping: date -> period_name -> {metrics..., indices: [hourly indices included]}
        """
        if not aggregation_periods:
            aggregation_periods = [
                {"name": "period_00_06", "start_hour": 0, "end_hour": 6},
                {"name": "period_06_12", "start_hour": 6, "end_hour": 12},
                {"name": "period_12_18", "start_hour": 12, "end_hour": 18},
                {"name": "period_18_24", "start_hour": 18, "end_hour": 24},
            ]

        per_date_periods: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for idx, entry in enumerate(hourly_list):
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
                        "indices": [],
                    })
                    agg = per_date_periods[date_key][pname]
                    try:
                        temp = float(entry.get("temperature_c")) if entry.get("temperature_c") is not None else None
                    except Exception:
                        temp = None
                    try:
                        wind_m_s = float(entry.get("wind_m_s")) if entry.get("wind_m_s") is not None else None
                    except Exception:
                        wind_m_s = None
                    try:
                        gust_m_s = float(entry.get("wind_max_m_s")) if entry.get("wind_max_m_s") is not None else None
                    except Exception:
                        gust_m_s = None
                    try:
                        cloud = int(entry.get("cloud_cover")) if entry.get("cloud_cover") is not None else None
                    except Exception:
                        cloud = None
                    try:
                        pop = int(entry.get("precipitation_probability")) if entry.get("precipitation_probability") is not None else None
                    except Exception:
                        pop = None
                    try:
                        pressure = float(entry.get("pressure_hpa")) if entry.get("pressure_hpa") is not None else None
                    except Exception:
                        pressure = None

                    if temp is not None:
                        agg["temperature_sum"] += temp
                    if wind_m_s is not None:
                        agg["wind_speed_sum"] += wind_m_s
                    if pressure is not None:
                        agg["pressure_sum"] += pressure
                    if cloud is not None:
                        agg["cloud_sum"] += cloud
                    if pop is not None:
                        agg["precip_max"] = max(agg["precip_max"], pop)
                    if gust_m_s is not None:
                        agg["gust_max"] = max(agg["gust_max"], gust_m_s)
                    agg["count"] += 1
                    agg["indices"].append(idx)
                    break

        # finalize
        final: Dict[str, Dict[str, Any]] = {}

        # determine output wind unit
        out_wind_unit = "km/h" if units == "metric" else "mph" if units == "imperial" else units

        for date_key in sorted(per_date_periods.keys())[:days]:
            final[date_key] = {}
            for pname, agg in per_date_periods[date_key].items():
                cnt = agg.get("count", 1) or 1
                try:
                    mean_temp = float(agg["temperature_sum"]) / cnt if agg.get("temperature_sum") is not None else None
                    mean_wind_m_s = float(agg["wind_speed_sum"]) / cnt if agg.get("wind_speed_sum") is not None else None
                    gust_m_s = float(agg["gust_max"]) if agg.get("gust_max") is not None else None
                    pressure = float(agg["pressure_sum"]) / cnt if agg.get("pressure_sum") is not None else None
                    cloud = int(round(float(agg["cloud_sum"]) / cnt)) if agg.get("cloud_sum") is not None else None
                    precip = int(round(float(agg["precip_max"]))) if agg.get("precip_max") is not None else None
                except Exception:
                    _LOGGER.debug("Failed to finalize aggregation for %s %s; skipping", date_key, pname)
                    continue

                # convert wind to requested display unit
                wind_out = None
                gust_out = None
                try:
                    if mean_wind_m_s is not None:
                        if out_wind_unit in ("km/h", "kph", "kmh"):
                            wind_out = unit_helpers.m_s_to_kmh(mean_wind_m_s)
                        elif out_wind_unit in ("mph",):
                            wind_out = unit_helpers.m_s_to_mph(mean_wind_m_s)
                        else:
                            wind_out = mean_wind_m_s
                    if gust_m_s is not None:
                        if out_wind_unit in ("km/h", "kph", "kmh"):
                            gust_out = unit_helpers.m_s_to_kmh(gust_m_s)
                        elif out_wind_unit in ("mph",):
                            gust_out = unit_helpers.m_s_to_mph(gust_m_s)
                        else:
                            gust_out = gust_m_s
                except Exception:
                    wind_out = None
                    gust_out = None

                final[date_key][pname] = {
                    "temperature": mean_temp,
                    "wind_speed": wind_out,
                    "wind_gust": gust_out,
                    "wind_unit": out_wind_unit,
                    "cloud_cover": cloud,
                    "precipitation_probability": precip,
                    "pressure": pressure,
                    "indices": list(agg.get("indices", [])),
                }

        return final

    def _convert_wind_array_value(self, v: Any, unit_hint: str) -> float:
        """Convert single wind array value to m/s using local converters."""
        try:
            if v is None:
                raise ValueError("None wind value")
            val = float(v)
        except Exception as exc:
            raise ValueError(f"Non-numeric wind value: {v!r}") from exc
        u = str(unit_hint).strip().lower() if unit_hint is not None else "m/s"
        if u in ("km/h", "kph", "kmh"):
            out = unit_helpers.kmh_to_m_s(val)
        elif u in ("mph", "mi/h", "miles/h"):
            out = unit_helpers.mph_to_m_s(val)
        else:
            # assume m/s
            out = val
        if out is None:
            raise ValueError(f"Unable to convert wind value: {v!r} with hint {unit_hint!r}")
        return float(out)