"""
Simplified OFA sensor (aggressive mode).

This sensor assumes strict canonical payloads produced by DataFormatter and
Skyfield. It removes historical aliases and many fallback heuristics so that
missing or malformed upstream data surfaces as clear errors.

Canonical assumptions (from DataFormatter / ocean_scoring):
 - coordinator.data is a dict with keys:
    - "timestamps" : list[str ISOZ]
    - "per_timestamp_forecasts": list[dict] (entries produced by ocean_scoring.compute_forecast)
    - "period_forecasts": dict (optional; strict shape)
    - "raw_payload": original raw payload (optional)
    - "tide": tide dict (optional, canonical fields if present)
    - "moon_phase": list or scalar (optional)
 - Each per_timestamp_forecasts entry contains:
    - "timestamp" (ISOZ), "index" (int), "score_100" (int or None), "components" (dict or None),
      "forecast_raw": { "formatted_weather": {...}, "score_calc": {...} }
"""
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
import logging

from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.const import ATTR_ATTRIBUTION
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import DOMAIN, CONF_NAME
from . import unit_helpers

_LOGGER = logging.getLogger(__name__)

ATTRIBUTION = "Data provided by Open-Meteo"


def _parse_dt_isoz(isoz: str) -> Optional[datetime]:
    """Parse ISOZ timestamp (expects ISO with trailing 'Z') into aware UTC datetime."""
    if isoz is None:
        return None
    if isinstance(isoz, datetime):
        return isoz.astimezone(timezone.utc) if isoz.tzinfo else isoz.replace(tzinfo=timezone.utc)
    try:
        # handle trailing Z -> +00:00 for fromisoformat
        s = str(isoz)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        parsed = datetime.fromisoformat(s)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _iso_z(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _round_opt(v: Optional[float], ndigits: int = 3) -> Optional[float]:
    if v is None:
        return None
    return round(float(v), ndigits)


def _m_s_to_display(w_m_s: Optional[float], entry_units: str) -> (Optional[float], Optional[str]):
    """Convert canonical m/s wind value to display units (km/h for metric, mph for imperial)."""
    if w_m_s is None:
        return None, None
    try:
        if entry_units == "metric":
            return _round_opt(unit_helpers.m_s_to_kmh(w_m_s), 2), "km/h"
        if entry_units == "imperial":
            return _round_opt(unit_helpers.m_s_to_mph(w_m_s), 2), "mph"
        return _round_opt(w_m_s, 2), "m/s"
    except Exception:
        return None, None


def _augment_components_with_values_simple(components: Optional[Dict[str, Any]], score_calc_raw: Optional[Dict[str, Any]], entry_units: str) -> Optional[Dict[str, Any]]:
    """
    Simplified augmentation:
      - Remove per-component 'score_10'
      - Inject the canonical numeric value used by scoring when available:
         wind -> wind_speed_m_s
         tide -> tide_height_m
         waves -> wave_height_m
         pressure -> pressure_delta_hpa
         moon -> moon_phase
         temperature -> temperature_c
    Expects `score_calc_raw` to be the canonical scoring raw dict (ocean_scoring.compute_score -> 'raw'),
    or an aggregated dict prepared by the caller for period summaries.
    """
    if components is None:
        return None
    if not isinstance(components, dict):
        return components

    out: Dict[str, Any] = {}
    for cname, cobj in components.items():
        if not isinstance(cobj, dict):
            out[cname] = cobj
            continue
        cc = dict(cobj)
        cc.pop("score_10", None)
        # pull numeric values from score_calc_raw.raw when present (canonical names)
        try:
            raw = score_calc_raw or {}
            if cname == "wind":
                if raw.get("wind") is not None:
                    # store canonical m/s value here; presentation layer may convert elsewhere
                    cc["wind_speed_m_s"] = _round_opt(raw.get("wind"), 3)
            elif cname == "tide":
                if raw.get("tide") is not None:
                    cc["tide_height_m"] = _round_opt(raw.get("tide"), 3)
            elif cname == "waves":
                if raw.get("wave") is not None:
                    cc["wave_height_m"] = _round_opt(raw.get("wave"), 3)
            elif cname == "pressure":
                if raw.get("pressure_delta") is not None:
                    cc["pressure_delta_hpa"] = _round_opt(raw.get("pressure_delta"), 2)
            elif cname == "moon":
                if raw.get("moon_phase") is not None:
                    cc["moon_phase"] = _round_opt(raw.get("moon_phase"), 6)
            elif cname == "temperature":
                if raw.get("temperature") is not None:
                    cc["temperature_c"] = _round_opt(raw.get("temperature"), 1)
        except Exception:
            # intentionally let missing pieces be absent rather than adding fallbacks
            pass
        out[cname] = cc
    return out


def _collect_safety_values(score_calc_raw: Optional[Dict[str, Any]], entry_units: str) -> Dict[str, Any]:
    """Collect minimal safety-related values from canonical score_calc.raw."""
    out: Dict[str, Any] = {}
    if not score_calc_raw or not isinstance(score_calc_raw, dict):
        return out
    raw = score_calc_raw
    try:
        if raw.get("wind_gust") is not None:
            # score_calc.raw.wind_gust is canonical m/s
            val = raw.get("wind_gust")
            if entry_units == "metric":
                out["wind_gust"] = _round_opt(unit_helpers.m_s_to_kmh(val), 2)
                out["wind_gust_unit"] = "km/h"
            elif entry_units == "imperial":
                out["wind_gust"] = _round_opt(unit_helpers.m_s_to_mph(val), 2)
                out["wind_gust_unit"] = "mph"
            else:
                out["wind_gust"] = _round_opt(val, 2)
                out["wind_gust_unit"] = "m/s"
        if raw.get("visibility_km") is not None:
            out["visibility_km"] = _round_opt(raw.get("visibility_km"), 2)
        if raw.get("swell_period_s") is not None:
            out["swell_period_s"] = _round_opt(raw.get("swell_period_s"), 1)
        if raw.get("precipitation_probability") is not None:
            try:
                out["precipitation_probability"] = int(round(float(raw.get("precipitation_probability"))))
            except Exception:
                pass
    except Exception:
        pass
    return out


class OFASensor(CoordinatorEntity):
    """
    Simplified CoordinatorEntity for Ocean Fishing Assistant.
    Reads `expose_raw` only from entry.options (no historical fallbacks).
    """

    def __init__(self, coordinator, name: str, expose_raw: bool = False):
        if not name:
            raise RuntimeError("Sensor name must be provided (strict simplified mode)")

        super().__init__(coordinator)

        prefix = "ocean_fishing_assistant"
        if not name.startswith(prefix):
            name = f"{prefix}_{name}"

        self._attr_name = name
        self._expose_raw = bool(expose_raw)

        # Unique id best-effort
        try:
            safe_name = name.replace(" ", "_")
            self._attr_unique_id = f"{prefix}_{getattr(coordinator, 'entry_id', 'noentry')}_{safe_name}"
        except Exception:
            self._attr_unique_id = None

    @property
    def available(self) -> bool:
        return bool(self.coordinator.last_update_success and self.coordinator.data)

    def _get_current_forecast(self) -> Optional[Dict[str, Any]]:
        """
        Return the per-timestamp forecast for the current UTC hour (floored).
        Simple canonical behavior: use per_timestamp_forecasts list and match 'timestamp'.
        """
        data = self.coordinator.data
        if not data or "per_timestamp_forecasts" not in data:
            raise RuntimeError("Coordinator data missing 'per_timestamp_forecasts' (strict)")

        forecasts = data["per_timestamp_forecasts"]
        if not isinstance(forecasts, list) or not forecasts:
            return None

        now_utc = dt_util.utcnow()
        floored = now_utc.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

        # find exact match first
        for entry in forecasts:
            ts = entry.get("timestamp")
            if ts is None:
                continue
            dt = _parse_dt_isoz(ts)
            if dt == floored:
                return entry

        # choose the first future entry (timestamp >= floored)
        for entry in forecasts:
            dt = _parse_dt_isoz(entry.get("timestamp"))
            if dt and dt >= floored:
                return entry

        # fallback to first (strict list should be ordered)
        return forecasts[0]

    @property
    def state(self) -> Optional[int]:
        data = self.coordinator.data
        if not data:
            raise RuntimeError("Coordinator data missing (strict)")

        forecast = self._get_current_forecast()
        if not forecast:
            raise RuntimeError("No per_timestamp_forecasts available (strict)")

        sc = forecast.get("score_100")
        if sc is None:
            raise RuntimeError("Current forecast missing score_100 (strict)")
        return int(sc)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """
        Build a compact attribute set assuming canonical inputs.
        - current_forecast: sanitized view of current per-timestamp forecast (components augmented)
        - remainder_of_today_periods & next_5_day_periods: sanitized period entries built from canonical period_forecasts
        - per_timestamp_forecasts & period_forecasts only when expose_raw True
        - raw_payload only when expose_raw True
        """
        data = self.coordinator.data
        if not data:
            raise RuntimeError("Coordinator data missing when building attributes (strict)")

        attrs: Dict[str, Any] = {}
        entry_units = getattr(self.coordinator, "units", "metric") or "metric"

        # current forecast
        current = self._get_current_forecast()
        if current is None:
            raise RuntimeError("Unable to locate current per-timestamp forecast (strict)")

        # Use forecast_raw.score_calc.raw as canonical numeric raw values used for scoring
        score_calc = (current.get("forecast_raw") or {}).get("score_calc") or {}
        score_calc_raw = score_calc.get("raw") if isinstance(score_calc, dict) else None

        # Sanitize current forecast: remove heavy raw blocks unless expose_raw True
        current_copy = dict(current)
        if not self._expose_raw:
            current_copy.pop("forecast_raw", None)
            # hide index and score_10 when raw is disabled
            current_copy.pop("index", None)
            current_copy.pop("score_10", None)

        # augment components: remove per-component score_10 and inject numeric values from score_calc_raw when present
        comps = current_copy.get("components")
        current_copy["components"] = _augment_components_with_values_simple(comps, score_calc_raw, entry_units)

        # attach grouped safety values derived from canonical score_calc_raw.raw
        safety_vals = _collect_safety_values(score_calc_raw, entry_units)
        if safety_vals:
            current_copy["safety_values"] = safety_vals

        attrs["current_forecast"] = current_copy

        # --- Grouped period views (remainder_of_today_periods, next_5_day_periods) ---
        period_forecasts = data.get("period_forecasts", {}) or {}
        timestamps = data.get("timestamps", []) or []
        per_ts_forecasts = data.get("per_timestamp_forecasts", []) or []

        # compute local now and today
        now_local = dt_util.now()
        today_local = now_local.date()

        remainder_of_today: Dict[str, Any] = {}
        next_5_days: Dict[str, Any] = {}

        # period_forecasts is canonical mapping: date -> period_name -> { "indices": [..], ... }
        for date_key, pmap in period_forecasts.items():
            for pname, pdata in (pmap or {}).items():
                indices = pdata.get("indices") or []
                if not isinstance(indices, list) or not indices:
                    continue
                # determine whether any index is later today (local)
                include_today = False
                include_next_days = False
                touched_dates = set()
                for idx in indices:
                    if not isinstance(idx, int):
                        continue
                    if idx < 0 or idx >= len(timestamps):
                        continue
                    ts = timestamps[idx]
                    dt_utc = _parse_dt_isoz(ts)
                    if dt_utc is None:
                        continue
                    dt_local = dt_util.as_local(dt_utc)
                    if dt_local.date() == today_local and dt_local >= now_local:
                        include_today = True
                    # next 5 days: pick dates > today and <= today+5
                    if today_local < dt_local.date() <= (today_local + timedelta(days=5)):
                        include_next_days = True
                        touched_dates.add(dt_local.date().isoformat())

                # sanitized period summary (remove heavy arrays)
                sanitized = dict(pdata)
                sanitized.pop("indices", None)
                # hide top-level score_10 when raw not exposed
                if not self._expose_raw:
                    sanitized.pop("score_10", None)

                # Prepare aggregated raw values for this period by averaging canonical raw values
                # collected from per_timestamp_forecasts[indices]. This enables injecting numeric
                # component values similar to current_forecast.
                raw_agg: Dict[str, float] = {}
                counts: Dict[str, int] = {}
                try:
                    for idx in indices:
                        if not isinstance(idx, int):
                            continue
                        if idx < 0 or idx >= len(per_ts_forecasts):
                            continue
                        fe = per_ts_forecasts[idx] or {}
                        score_calc = (fe.get("forecast_raw") or {}).get("score_calc") or {}
                        raw = score_calc.get("raw") if isinstance(score_calc, dict) else None
                        if not raw:
                            continue
                        # keys of interest: wind (m/s), tide (m), wave (m), pressure_delta (hPa),
                        # moon_phase, temperature
                        for k, keyname in (("wind", "wind"), ("tide", "tide"), ("wave", "wave"),
                                           ("pressure_delta", "pressure_delta"), ("moon_phase", "moon_phase"),
                                           ("temperature", "temperature")):
                            if raw.get(keyname) is None:
                                continue
                            try:
                                v = float(raw.get(keyname))
                            except Exception:
                                continue
                            if k not in raw_agg:
                                raw_agg[k] = v
                                counts[k] = 1
                            else:
                                raw_agg[k] += v
                                counts[k] += 1
                    # finalize averages
                    for k in list(raw_agg.keys()):
                        c = counts.get(k, 1) or 1
                        raw_agg[k] = raw_agg[k] / c
                except Exception:
                    # on any failure, leave raw_agg empty so augmentation just won't inject values
                    raw_agg = {}

                # augment components similarly (period-level components are aggregated; provide aggregated raw)
                comps = sanitized.get("components")
                # map raw_agg keys to the names _augment expects (it just reads raw.get("wind"), raw.get("tide"), ...)
                sanitized["components"] = _augment_components_with_values_simple(comps, raw_agg or None, entry_units)

                # add safety_values from period-level keys if present (pressure, wind_gust, precipitation_probability)
                period_safety = {}
                if sanitized.get("wind_gust") is not None:
                    # DataFormatter produced wind_gust in display units already under period entries
                    period_safety["wind_gust"] = _round_opt(sanitized.get("wind_gust"), 2)
                    if sanitized.get("wind_unit"):
                        period_safety["wind_gust_unit"] = sanitized.get("wind_unit")
                if sanitized.get("precipitation_probability") is not None:
                    try:
                        period_safety["precipitation_probability"] = int(round(float(sanitized.get("precipitation_probability"))))
                    except Exception:
                        pass
                if period_safety:
                    sanitized["safety_values"] = period_safety

                if include_today:
                    remainder_of_today[pname] = sanitized
                if include_next_days:
                    for d in touched_dates:
                        next_5_days.setdefault(d, {})[pname] = sanitized

        attrs["remainder_of_today_periods"] = remainder_of_today
        attrs["next_5_day_periods"] = next_5_days

        # expose per_timestamp_forecasts and period_forecasts only when explicitly requested
        if self._expose_raw:
            attrs["per_timestamp_forecasts"] = data.get("per_timestamp_forecasts")
            attrs["period_forecasts"] = period_forecasts
            attrs["raw_payload"] = data.get("raw_payload") or data
        attrs["raw_output_enabled"] = bool(self._expose_raw)

        # Top-level summary fields (simple, canonical)
        attrs["profile_used"] = current.get("profile_used")
        attrs["units"] = entry_units

        # Tide & moon: prefer canonical tide block if present
        tide_obj = data.get("tide")
        if tide_obj and isinstance(tide_obj, dict):
            next_high = tide_obj.get("next_high")
            next_low = tide_obj.get("next_low")
            attrs["next_high_tide"] = {"timestamp": next_high} if next_high else None
            attrs["next_low_tide"] = {"timestamp": next_low} if next_low else None

        # moon_phase: prefer score_calc_raw.raw.moon_phase then canonical top-level moon_phase array
        moon_numeric = None
        if score_calc_raw and isinstance(score_calc_raw, dict) and score_calc_raw.get("moon_phase") is not None:
            moon_numeric = score_calc_raw.get("moon_phase")
        elif "moon_phase" in data:
            mp = data.get("moon_phase")
            if isinstance(mp, (list, tuple)):
                # try to align using current index
                idx = current.get("index")
                if isinstance(idx, int) and idx < len(mp):
                    moon_numeric = mp[idx]
                else:
                    moon_numeric = mp[0] if mp else None
            else:
                moon_numeric = mp
        attrs["moon_phase"] = _round_opt(moon_numeric, 6) if moon_numeric is not None else None

        # Top-level current metrics (use formatted_weather produced by scoring)
        formatted = (current.get("forecast_raw") or {}).get("formatted_weather") or {}
        # Temperature (°C)
        if formatted.get("temperature") is not None:
            attrs["current_temperature"] = _round_opt(formatted.get("temperature"), 1)
        else:
            attrs["current_temperature"] = None

        # Wind (convert canonical m/s to display)
        # ocean_scoring's formatted_weather.wind is canonical m/s
        wind_m_s = formatted.get("wind")
        w_val, w_unit = _m_s_to_display(wind_m_s, entry_units)
        attrs["current_wind_speed"] = w_val
        attrs["current_wind_unit"] = w_unit

        # Gust (if provided)
        gust_m_s = formatted.get("wind_gust")
        g_val, g_unit = _m_s_to_display(gust_m_s, entry_units)
        attrs["current_wind_gust"] = g_val
        attrs["current_wind_gust_unit"] = g_unit

        # Pressure (hPa)
        attrs["current_pressure_hpa"] = _round_opt(formatted.get("pressure_hpa"), 1) if formatted.get("pressure_hpa") is not None else None

        # Waves / swell (canonical units: meter / seconds)
        attrs["current_wave_height_m"] = _round_opt(formatted.get("wave_height_m"), 3) if formatted.get("wave_height_m") is not None else None
        attrs["current_wave_period_s"] = _round_opt(formatted.get("wave_period_s"), 2) if formatted.get("wave_period_s") is not None else None
        attrs["current_swell_period_s"] = _round_opt(formatted.get("swell_period_s"), 1) if formatted.get("swell_period_s") is not None else None

        # Attribution and return
        attrs["attribution"] = ATTRIBUTION
        attrs[ATTR_ATTRIBUTION] = ATTRIBUTION
        return attrs


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities):
    """
    Setup entry. In this simplified/aggressive mode we read expose_raw only from entry.options.
    """
    coordinator = hass.data[DOMAIN].get(entry.entry_id) if hass.data.get(DOMAIN) else None
    if coordinator is None:
        raise RuntimeError("Coordinator not found in hass.data for this config entry (strict)")

    sensor_name = entry.data.get(CONF_NAME)
    if not sensor_name:
        raise RuntimeError("Missing required name in config entry data (strict)")

    expose_raw = bool(entry.options.get("expose_raw", False)) if isinstance(entry.options, dict) else False

    prefix = "ocean_fishing_assistant"
    final_name = sensor_name if sensor_name.startswith(prefix) else f"{prefix}_{sensor_name}"

    async_add_entities([OFASensor(coordinator, name=final_name, expose_raw=expose_raw)], update_before_add=True)