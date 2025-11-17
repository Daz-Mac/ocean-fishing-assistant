# Strict sensor entity - surfaces errors loudly so callers see misconfiguration
from typing import Optional, Dict, Any, List, Tuple, Set
from datetime import datetime, timezone, timedelta
import math
import logging
from statistics import mean

from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.const import ATTR_ATTRIBUTION
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import DOMAIN, CONF_NAME
from . import unit_helpers

_ATTR_LOGGER = logging.getLogger(__name__)

ATTRIBUTION = "Data provided by Open-Meteo"


def _parse_dt(v: Any) -> Optional[datetime]:
    """Parse a timestamp value into an aware UTC datetime, or None."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        # numeric epoch (secs or ms)
        if isinstance(v, (int, float)):
            val = float(v)
            if val > 1e12:  # probably milliseconds
                val = val / 1000.0
            return datetime.fromtimestamp(val, tz=timezone.utc)
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        parsed = datetime.fromisoformat(s)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _iso_z(dt: Optional[datetime]) -> str:
    """Return ISO string ending with Z for an aware UTC datetime, or empty string for None."""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _moon_phase_name(p: Optional[float]) -> Optional[str]:
    """Return human-friendly moon phase name for p in [0,1], or None."""
    if p is None:
        return None
    try:
        val = float(p) % 1.0
    except Exception:
        return None
    if val < 0.0625 or val >= 0.9375:
        return "New Moon"
    if val < 0.1875:
        return "Waxing Crescent"
    if val < 0.3125:
        return "First Quarter"
    if val < 0.4375:
        return "Waxing Gibbous"
    if val < 0.5625:
        return "Full Moon"
    if val < 0.6875:
        return "Waning Gibbous"
    if val < 0.8125:
        return "Last Quarter"
    return "Waning Crescent"


def _round_opt(v: Any, ndigits: int = 3) -> Any:
    """Round numeric v to ndigits or return None if v is None."""
    try:
        if v is None:
            return None
        return round(float(v), ndigits)
    except Exception:
        return v


def _find_height_for_timestamp(target_iso: str, tide_ts: List[Any], tide_heights: List[Any]) -> Optional[float]:
    """Find tide height aligned with a target ISO timestamp. Returns float or None."""
    if not target_iso or not tide_ts or not tide_heights:
        return None
    try:
        target_dt = _parse_dt(target_iso)
        if target_dt is None:
            return None
    except Exception:
        return None
    # Try exact match first
    for i, t in enumerate(tide_ts):
        dt = _parse_dt(t)
        if dt is None:
            continue
        if dt == target_dt:
            try:
                return float(tide_heights[i]) if tide_heights[i] is not None else None
            except Exception:
                return None
    # Fallback to nearest by time
    best_idx = None
    best_dt_diff = None
    for i, t in enumerate(tide_ts):
        dt = _parse_dt(t)
        if dt is None:
            continue
        diff = abs((dt - target_dt).total_seconds())
        if best_dt_diff is None or diff < best_dt_diff:
            best_dt_diff = diff
            best_idx = i
    if best_idx is not None:
        try:
            return float(tide_heights[best_idx]) if tide_heights[best_idx] is not None else None
        except Exception:
            return None
    return None


def _sanitize_components_remove_score10(comps: Any) -> Any:
    """Return components dict with per-component 'score_10' removed (non-dicts preserved)."""
    if not isinstance(comps, dict):
        return comps
    new_comps: Dict[str, Any] = {}
    for cname, cval in comps.items():
        if isinstance(cval, dict):
            cc = dict(cval)
            cc.pop("score_10", None)
            new_comps[cname] = cc
        else:
            new_comps[cname] = cval
    return new_comps


def _add_component_values_from_raw(
    comp_name: str, cc: Dict[str, Any], raw: Dict[str, Any], entry_units: str
) -> Dict[str, Any]:
    """
    Given a component dict cc (already a shallow copy) and score_calc.raw dict,
    add component-specific numeric value converted to entry_units.
    """
    try:
        if not isinstance(raw, dict):
            return cc
        if comp_name == "wind":
            v = raw.get("wind")
            if v is not None:
                if entry_units == "metric":
                    conv = unit_helpers.m_s_to_kmh(v)
                elif entry_units == "imperial":
                    conv = unit_helpers.m_s_to_mph(v)
                else:
                    conv = v
                cc["wind_speed"] = _round_opt(conv, 2)
        elif comp_name == "time":
            ts = raw.get("timestamp")
            dt = _parse_dt(ts)
            if dt is not None:
                # show local hour for readability
                try:
                    dt_local = dt_util.as_local(dt)
                    cc["hour"] = dt_local.hour
                except Exception:
                    cc["timestamp"] = _iso_z(dt)
        elif comp_name == "tide":
            v = raw.get("tide")
            if v is not None:
                if entry_units == "imperial":
                    ft = unit_helpers.m_to_ft(v)
                    cc["tide_height_ft"] = _round_opt(ft, 2)
                else:
                    cc["tide_height_m"] = _round_opt(v, 3)
        elif comp_name == "waves":
            v = raw.get("wave")
            if v is not None:
                if entry_units == "imperial":
                    cc["wave_height_ft"] = _round_opt(unit_helpers.m_to_ft(v), 2)
                else:
                    cc["wave_height_m"] = _round_opt(v, 3)
        elif comp_name == "pressure":
            v = raw.get("pressure_delta")
            if v is not None:
                cc["pressure_delta_hpa"] = _round_opt(v, 2)
        elif comp_name == "season":
            # season is categorical; nothing numeric to add reliably
            pass
        elif comp_name == "moon":
            v = raw.get("moon_phase")
            if v is not None:
                cc["moon_phase"] = _round_opt(v, 6)
        elif comp_name == "temperature":
            v = raw.get("temperature")
            if v is not None:
                if entry_units == "imperial":
                    cc["temperature_f"] = _round_opt(unit_helpers.c_to_f(v), 1)
                else:
                    cc["temperature_c"] = _round_opt(v, 1)
        # NOTE: Previously we added precipitation_probability into every component here.
        # That behavior was removed in favor of grouping safety-related values under a dedicated "safety_values" dict.
    except Exception:
        pass
    return cc


def _compute_period_values_from_indices(
    comp_name: str,
    indices: List[int],
    per_ts_forecasts: Optional[List[Dict[str, Any]]],
    entry_units: str,
) -> Optional[Any]:
    """
    Compute a representative value for a component across provided per-timestamp forecast indices.
    Returns a converted numeric value matching entry_units or None.
    Rules:
      - wind: mean wind (converted), 2 dp
      - time: not applicable
      - tide: mean tide height (m or ft)
      - waves: mean wave height (m or ft)
      - pressure: mean pressure_delta_hpa (2 dp)
      - moon: mean moon_phase (6 dp)
      - temperature: mean temperature (°C or °F)
    """
    if not indices or not isinstance(indices, (list, tuple)) or not per_ts_forecasts:
        return None
    vals = []
    for idx in indices:
        try:
            if idx < 0 or idx >= len(per_ts_forecasts):
                continue
            fe = per_ts_forecasts[idx] or {}
            score_calc = (fe.get("forecast_raw") or {}).get("score_calc") or {}
            raw = score_calc.get("raw") or {}
            if not raw:
                continue
            if comp_name == "wind":
                if raw.get("wind") is not None:
                    vals.append(float(raw.get("wind")))
            elif comp_name == "tide":
                if raw.get("tide") is not None:
                    vals.append(float(raw.get("tide")))
            elif comp_name == "waves":
                if raw.get("wave") is not None:
                    vals.append(float(raw.get("wave")))
            elif comp_name == "pressure":
                if raw.get("pressure_delta") is not None:
                    vals.append(float(raw.get("pressure_delta")))
            elif comp_name == "moon":
                if raw.get("moon_phase") is not None:
                    vals.append(float(raw.get("moon_phase")))
            elif comp_name == "temperature":
                if raw.get("temperature") is not None:
                    vals.append(float(raw.get("temperature")))
            elif comp_name == "time":
                # not meaningful to average time
                pass
        except Exception:
            continue

    if not vals:
        return None

    try:
        avg = float(sum(vals) / len(vals))
    except Exception:
        return None

    # convert to display unit
    if comp_name == "wind":
        if entry_units == "metric":
            return _round_opt(unit_helpers.m_s_to_kmh(avg), 2)
        elif entry_units == "imperial":
            return _round_opt(unit_helpers.m_s_to_mph(avg), 2)
        else:
            return _round_opt(avg, 2)
    if comp_name in ("tide", "waves"):
        if entry_units == "imperial":
            return _round_opt(unit_helpers.m_to_ft(avg), 2)
        else:
            return _round_opt(avg, 3)
    if comp_name == "pressure":
        return _round_opt(avg, 2)
    if comp_name == "moon":
        return _round_opt(avg, 6)
    if comp_name == "temperature":
        if entry_units == "imperial":
            return _round_opt(unit_helpers.c_to_f(avg), 1)
        else:
            return _round_opt(avg, 1)
    return None


def _augment_components_with_values(
    components: Any,
    score_calc_raw: Optional[Dict[str, Any]],
    period_entry: Optional[Dict[str, Any]],
    indices: Optional[List[int]],
    per_ts_forecasts: Optional[List[Dict[str, Any]]],
    entry_units: str,
) -> Any:
    """
    Given a components dict, attempt to add per-component numeric values that were used
    to compute the score. Returns a new components dict (score_10 removed).
    Prefer per-timestamp score_calc_raw when present; otherwise use period_entry top-level
    representative values or compute from per_timestamp_forecasts (indices).
    NOTE: Safety-only inputs (gusts, visibility, precip, swell period) are intentionally NOT
    injected into per-component objects; they are grouped under a dedicated 'safety_values' dict.
    """
    if not isinstance(components, dict):
        return components

    new_comps: Dict[str, Any] = {}
    for cname, cval in components.items():
        if not isinstance(cval, dict):
            new_comps[cname] = cval
            continue
        cc = dict(cval)
        # remove per-component score_10 (we already removed top-level score_10 elsewhere)
        cc.pop("score_10", None)

        # 1) try per-timestamp raw (best, canonical units)
        if score_calc_raw and isinstance(score_calc_raw, dict):
            try:
                cc = _add_component_values_from_raw(cname, cc, score_calc_raw, entry_units)
                new_comps[cname] = cc
                continue
            except Exception:
                pass

        # 2) try period_entry top-level keys (already in display units for DataFormatter fallback)
        if period_entry and isinstance(period_entry, dict):
            try:
                if cname == "wind" and "wind_speed" in period_entry:
                    # already in display units from DataFormatter
                    cc["wind_speed"] = _round_opt(period_entry.get("wind_speed"), 2)
                    new_comps[cname] = cc
                    continue
                if cname == "temperature" and "temperature" in period_entry:
                    if entry_units == "imperial":
                        cc["temperature_f"] = _round_opt(unit_helpers.c_to_f(period_entry.get("temperature")), 1)
                    else:
                        cc["temperature_c"] = _round_opt(period_entry.get("temperature"), 1)
                    new_comps[cname] = cc
                    continue
                if cname in ("waves", "tide") and "wave_height_m" in period_entry:
                    # Some period payloads may include wave_height_m explicitly
                    if entry_units == "imperial":
                        cc["wave_height_ft"] = _round_opt(unit_helpers.m_to_ft(period_entry.get("wave_height_m")), 2)
                    else:
                        cc["wave_height_m"] = _round_opt(period_entry.get("wave_height_m"), 3)
                    new_comps[cname] = cc
                    continue
                # Fallback mapping for aggregate aggregator output keys:
                if cname == "waves" and "wind_speed" in period_entry and "wind_unit" in period_entry:
                    # not ideal; skip
                    pass
                # Many DataFormatter fallback aggregated period entries include 'pressure', 'wind_gust', 'wind_speed'
                if cname == "pressure" and "pressure" in period_entry:
                    cc["pressure_hpa"] = _round_opt(period_entry.get("pressure"), 2)
                    new_comps[cname] = cc
                    continue
                if cname == "tide" and "tide_height_m" in period_entry:
                    if entry_units == "imperial":
                        cc["tide_height_ft"] = _round_opt(unit_helpers.m_to_ft(period_entry.get("tide_height_m")), 2)
                    else:
                        cc["tide_height_m"] = _round_opt(period_entry.get("tide_height_m"), 3)
                    new_comps[cname] = cc
                    continue
                if cname == "moon" and "moon_phase" in period_entry:
                    cc["moon_phase"] = _round_opt(period_entry.get("moon_phase"), 6)
                    new_comps[cname] = cc
                    continue
            except Exception:
                pass

        # 3) fallback: compute from per_timestamp_forecasts and indices if available
        try:
            val = _compute_period_values_from_indices(cname, indices or [], per_ts_forecasts, entry_units)
            if val is not None:
                # choose key names consistent with above per-component keys
                if cname == "wind":
                    cc["wind_speed"] = val
                elif cname in ("waves",):
                    # if val is in ft (imperial) or m (metric): name accordingly
                    if entry_units == "imperial":
                        cc["wave_height_ft"] = val
                    else:
                        cc["wave_height_m"] = val
                elif cname == "tide":
                    if entry_units == "imperial":
                        cc["tide_height_ft"] = val
                    else:
                        cc["tide_height_m"] = val
                elif cname == "pressure":
                    cc["pressure_delta_hpa"] = val
                elif cname == "moon":
                    cc["moon_phase"] = val
                elif cname == "temperature":
                    if entry_units == "imperial":
                        cc["temperature_f"] = val
                    else:
                        cc["temperature_c"] = val
                else:
                    # generic numeric fallback
                    cc["value"] = val
        except Exception:
            pass

        new_comps[cname] = cc
    return new_comps


def _gather_safety_values_from_sources(
    score_calc_raw: Optional[Dict[str, Any]],
    period_entry: Optional[Dict[str, Any]],
    raw_current: Optional[Dict[str, Any]],
    entry_units: str,
) -> Dict[str, Any]:
    """
    Collect safety-related raw values from available sources (prefer score_calc_raw,
    fallback to period_entry, then raw_current). Convert units for display where appropriate.
    Returns a dict containing only present values:
      - wind_gust (display units: km/h or mph, 2 dp)
      - visibility_km (km, 2 dp)
      - swell_period_s (s, 1 dp)
      - precipitation_probability (int percent)
    """
    out: Dict[str, Any] = {}
    try:
        def _pick(*keys):
            for k in keys:
                try:
                    if score_calc_raw and isinstance(score_calc_raw, dict) and score_calc_raw.get(k) is not None:
                        return score_calc_raw.get(k)
                except Exception:
                    pass
                try:
                    if period_entry and isinstance(period_entry, dict) and period_entry.get(k) is not None:
                        return period_entry.get(k)
                except Exception:
                    pass
                try:
                    if raw_current and isinstance(raw_current, dict) and raw_current.get(k) is not None:
                        return raw_current.get(k)
                except Exception:
                    pass
            return None

        # wind_gust: possible canonical keys
        gust_val = _pick("wind_gust", "wind_max_m_s", "wind_max", "windgusts_10m")
        if gust_val is not None:
            try:
                gv = float(gust_val)
                # If gust came from score_calc_raw it is in m/s; otherwise could be in other units.
                # We assume canonical m/s if score_calc_raw provided it; otherwise leave as-is but convert if entry_units suggests.
                if score_calc_raw and (("wind_gust" in score_calc_raw) or ("wind_max_m_s" in score_calc_raw)):
                    # canonical m/s -> convert to display units
                    if entry_units == "metric":
                        out["wind_gust"] = _round_opt(unit_helpers.m_s_to_kmh(gv), 2)
                        out["wind_gust_unit"] = "km/h"
                    elif entry_units == "imperial":
                        out["wind_gust"] = _round_opt(unit_helpers.m_s_to_mph(gv), 2)
                        out["wind_gust_unit"] = "mph"
                    else:
                        out["wind_gust"] = _round_opt(gv, 2)
                        out["wind_gust_unit"] = "m/s"
                else:
                    # best-effort: if key name suggests kmh/mph then convert to display unit, else assume m/s
                    out["wind_gust"] = _round_opt(gv, 2)
                    # unit label not guaranteed here
            except Exception:
                pass

        # visibility (prefer canonical visibility_km)
        vis = _pick("visibility_km", "visibility", "vis")
        if vis is not None:
            try:
                out["visibility_km"] = _round_opt(float(vis), 2)
            except Exception:
                pass

        # swell period in seconds
        sp = _pick("swell_period_s", "swell_period", "swell_period_seconds")
        if sp is not None:
            try:
                out["swell_period_s"] = _round_opt(float(sp), 1)
            except Exception:
                pass

        # precipitation_probability / pop
        pp = _pick("precipitation_probability", "pop", "precip")
        if pp is not None:
            try:
                out["precipitation_probability"] = int(round(float(pp)))
            except Exception:
                pass
    except Exception:
        pass
    return out


class OFASensor(CoordinatorEntity):
    def __init__(self, coordinator, name: str, expose_raw: bool = False):
        """
        Strict CoordinatorEntity wrapper.

        expose_raw: when True, include 'raw_payload' and per_timestamp_forecasts in attributes so developers can inspect
                    full upstream payload. Default False (hide raw output).
        """
        if not name:
            raise RuntimeError("Sensor name must be provided (strict)")

        super().__init__(coordinator)

        prefix = "ocean_fishing_assistant"
        if not name.startswith(prefix):
            name = f"{prefix}_{name}"

        self._attr_name = name
        self._expose_raw = bool(expose_raw)

        try:
            safe_name = name.replace(" ", "_")
            self._attr_unique_id = f"{prefix}_{getattr(coordinator, 'entry_id', 'noentry')}_{safe_name}"
        except Exception:
            self._attr_unique_id = None

    @property
    def available(self) -> bool:
        return bool(self.coordinator.last_update_success and self.coordinator.data is not None)

    def _get_current_forecast(self) -> Optional[Dict[str, Any]]:
        """
        Return the per-timestamp forecast that corresponds to the current hour (floored).
        Strategy:
         - Floor current UTC time to hour, look for exact timestamp match.
         - If none, prefer the next future forecast (>= floored hour).
         - If still none, use the nearest past forecast (<= floored hour).
         - Final fallback: forecasts[0].
        """
        data = self.coordinator.data or {}
        forecasts = data.get("per_timestamp_forecasts") or []
        if not isinstance(forecasts, (list, tuple)) or len(forecasts) == 0:
            return None

        now_utc = dt_util.utcnow()
        if now_utc.tzinfo is None:
            # make aware UTC if needed
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        floored = now_utc.replace(minute=0, second=0, microsecond=0)

        exact_match = None
        last_past = None
        first_future = None
        best_past_delta = None
        best_future_delta = None

        for entry in forecasts:
            try:
                ts_raw = entry.get("timestamp") if isinstance(entry, dict) else None
                if ts_raw is None:
                    # try alternate keys
                    ts_raw = entry.get("time") if isinstance(entry, dict) else None
                dt = _parse_dt(ts_raw)
                if dt is None:
                    continue
                # normalize to hour-precision comparison (we compare exact datetimes)
                if dt == floored:
                    exact_match = entry
                    break
                if dt < floored:
                    delta = (floored - dt).total_seconds()
                    if best_past_delta is None or delta < best_past_delta:
                        best_past_delta = delta
                        last_past = entry
                else:
                    delta = (dt - floored).total_seconds()
                    if best_future_delta is None or delta < best_future_delta:
                        best_future_delta = delta
                        first_future = entry
            except Exception:
                continue

        if exact_match is not None:
            _ATTR_LOGGER.debug("Selected exact-match forecast for %s", floored.isoformat())
            return exact_match
        # Prioritize upcoming (future) forecast over previous when no exact match.
        if first_future is not None:
            _ATTR_LOGGER.debug(
                "No exact-match forecast for %s; selecting next future forecast timestamp", floored.isoformat()
            )
            return first_future
        if last_past is not None:
            _ATTR_LOGGER.debug(
                "No future forecast for %s; selecting nearest past forecast timestamp", floored.isoformat()
            )
            return last_past

        # final fallback
        try:
            _ATTR_LOGGER.debug("Falling back to first forecast entry (index 0)")
            return forecasts[0]
        except Exception:
            return None

    @property
    def state(self) -> Optional[int]:
        """Return state as integer 0..100 or raise on strict failures."""
        if not self.coordinator.data:
            raise RuntimeError("Coordinator data missing for OFA sensor (strict)")

        forecast = self._get_current_forecast()
        if forecast:
            sc = forecast.get("score_100")
            if sc is None:
                raise RuntimeError("Precomputed forecast present but score_100 is missing (strict)")
            return int(sc)

        raise RuntimeError("No precomputed per_timestamp_forecasts found in coordinator.data (strict)")

    def _extract_wind_ms_and_hint(self, formatted: Optional[Dict[str, Any]], raw_current: Optional[Dict[str, Any]]) -> (Optional[float], Optional[str]):
        """
        Return a tuple (wind_m_s_or_None, src_unit_hint_or_None).
        - If formatted provides 'wind', assume canonical m/s (DataFormatter canonicalization).
        - Else look at raw_current and attempt to interpret wind/wind_speed and wind_unit.
        """
        # 1) formatted (canonical m/s)
        try:
            if formatted and isinstance(formatted.get("wind"), (int, float)):
                return float(formatted.get("wind")), "m/s"
            if formatted and formatted.get("wind") is not None:
                # if it's a string numeric
                try:
                    return float(formatted.get("wind")), "m/s"
                except Exception:
                    pass
        except Exception:
            pass

        # 2) raw_current: try common keys and unit hints
        try:
            if raw_current and isinstance(raw_current, dict):
                for k in ("wind", "wind_speed", "wind_speed_10m", "wind_m_s", "wind_kmh", "wind_mph"):
                    if k in raw_current and raw_current.get(k) is not None:
                        val = raw_current.get(k)
                        # unit hint detection
                        unit_hint = None
                        try:
                            # prefer explicit unit label key
                            if "wind_unit" in raw_current and raw_current.get("wind_unit"):
                                unit_hint = str(raw_current.get("wind_unit")).strip().lower()
                            elif k.endswith("kmh") or k.endswith("km/h") or k.endswith("kmh"):
                                unit_hint = "km/h"
                            elif k.endswith("mph"):
                                unit_hint = "mph"
                            elif k.endswith("m_s") or k.endswith("m/s"):
                                unit_hint = "m/s"
                        except Exception:
                            unit_hint = None
                        try:
                            return float(val), unit_hint
                        except Exception:
                            return None, unit_hint
        except Exception:
            pass
        return None, None

    def _to_display_wind(self, raw_val: Optional[float], src_unit_hint: Optional[str], entry_units: str) -> (Optional[float], Optional[str]):
        """
        Convert a wind value (raw_val with optional src_unit_hint) to display units.
        entry_units: "metric" => km/h, "imperial" => mph, else leave as m/s.
        Returns (value_converted_or_None, display_unit_label_or_None).
        """
        if raw_val is None:
            return None, None

        # First, coerce to m/s (canonical) from source hint if needed.
        ms_val = None
        try:
            if src_unit_hint:
                uh = str(src_unit_hint).strip().lower()
                if uh in ("m/s", "m_s", "mps"):
                    ms_val = float(raw_val)
                elif uh in ("km/h", "kmh", "kph", "km per h"):
                    ms_val = unit_helpers.kmh_to_m_s(raw_val)
                elif uh in ("mph", "mi/h", "miles/h"):
                    ms_val = unit_helpers.mph_to_m_s(raw_val)
                else:
                    # unknown hint: try numeric interpretation and assume m/s
                    ms_val = float(raw_val)
            else:
                # No hint: treat incoming as m/s (DataFormatter typically gives canonical m/s)
                ms_val = float(raw_val)
        except Exception:
            return None, None

        # Convert to requested display units
        if entry_units == "metric":
            # display km/h
            out_val = unit_helpers.m_s_to_kmh(ms_val)
            return _round_opt(out_val, 2), "km/h"
        if entry_units == "imperial":
            out_val = unit_helpers.m_s_to_mph(ms_val)
            return _round_opt(out_val, 2), "mph"

        # fallback: show m/s
        return _round_opt(ms_val, 2), "m/s"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """
        Strict attributes with display conversions applied for final outputs (wind units converted
        per user selection). Per-hour scoring (per_timestamp_forecasts) is included only when expose_raw=True.
        Raw payload is preserved under 'raw_payload' only if expose_raw is True.
        Also exposes:
         - period_forecasts: full mapping from DataFormatter (sanitized to remove per-period/profile redundancies)
         - remainder_of_today_periods: periods for the local "today" that still include future hours
         - next_5_day_periods: periods grouped by local calendar date for the next 5 days (excluding today)
         - raw_output_enabled: boolean indicating whether raw payload was exposed
         - safety_values: numeric values used for safety checks grouped separately from components
        """
        if not self.coordinator.data:
            raise RuntimeError("Coordinator data missing when building attributes (strict)")

        attrs: Dict[str, Any] = {}
        forecasts = self.coordinator.data.get("per_timestamp_forecasts")
        period_forecasts = self.coordinator.data.get("period_forecasts")
        timestamps = self.coordinator.data.get("timestamps")

        # --- CURRENT forecast first (user-friendly) ---
        current = self._get_current_forecast()

        # Always remove nested profile_used; keep only top-level profile_used below.
        if current is not None:
            try:
                # Make a shallow copy and remove profile_used and sensitive low-level keys
                current_copy = dict(current)
                current_copy.pop("profile_used", None)
                if not self._expose_raw:
                    if "forecast_raw" in current_copy:
                        current_copy.pop("forecast_raw", None)
                    # Remove any top-level score_10
                    current_copy.pop("score_10", None)
                    # Also defensively redact nested score_calc.raw if present on calculation object
                    try:
                        sc = current_copy.get("score_calc") or (current_copy.get("forecast_raw") or {}).get("score_calc")
                        if isinstance(sc, dict) and "raw" in sc:
                            sc = dict(sc)
                            sc.pop("raw", None)
                            sc.pop("score_10", None)
                            current_copy["score_calc"] = sc
                    except Exception:
                        pass
                    # Remove per-component score_10 if components present and augment components with values where possible
                    try:
                        comps = current_copy.get("components")
                        if comps is not None:
                            # augment with values where possible (prefer score_calc.raw)
                            score_calc_raw = None
                            try:
                                score_calc_raw = (current.get("forecast_raw") or {}).get("score_calc", {}).get("raw")
                            except Exception:
                                score_calc_raw = None
                            current_copy["components"] = _augment_components_with_values(
                                comps, score_calc_raw, None, None, self.coordinator.data.get("per_timestamp_forecasts"), getattr(self.coordinator, "units", "metric") or "metric"
                            )
                            # Attach grouped safety values (do not include safety-only fields under components)
                            safety_vals = _gather_safety_values_from_sources(
                                score_calc_raw,
                                None,
                                (self.coordinator.data.get("raw_payload") or {}).get("current") if isinstance(self.coordinator.data, dict) else None,
                                getattr(self.coordinator, "units", "metric") or "metric",
                            )
                            if safety_vals:
                                current_copy["safety_values"] = safety_vals
                    except Exception:
                        pass
                else:
                    # Expose_raw True: still remove per-component score_10 so nested outputs are tidy, but add values too
                    try:
                        comps = current_copy.get("components")
                        if comps is not None:
                            score_calc_raw = None
                            try:
                                score_calc_raw = (current.get("forecast_raw") or {}).get("score_calc", {}).get("raw")
                            except Exception:
                                score_calc_raw = None
                            current_copy["components"] = _augment_components_with_values(
                                comps, score_calc_raw, None, None, self.coordinator.data.get("per_timestamp_forecasts"), getattr(self.coordinator, "units", "metric") or "metric"
                            )
                            # Attach grouped safety values for visibility when raw exposed
                            safety_vals = _gather_safety_values_from_sources(
                                score_calc_raw,
                                None,
                                (self.coordinator.data.get("raw_payload") or {}).get("current") if isinstance(self.coordinator.data, dict) else None,
                                getattr(self.coordinator, "units", "metric") or "metric",
                            )
                            if safety_vals:
                                current_copy["safety_values"] = safety_vals
                    except Exception:
                        pass
                attrs["current_forecast"] = current_copy
                _ATTR_LOGGER.debug("current_forecast prepared for sensor %s", self._attr_name)
            except Exception:
                attrs["current_forecast"] = current

        # --- remainder_of_today and next_5_day grouped forecasts next ---
        remainder_of_today: Dict[str, Any] = {}
        next_5_days: Dict[str, Any] = {}
        try:
            # Use Home Assistant local "now"
            now_local = dt_util.now()
            if now_local.tzinfo is None:
                # ensure aware
                now_local = now_local.replace(tzinfo=dt_util.get_time_zone())

            # prepare set of next 5 local dates (excluding today)
            today_local = now_local.date()
            next_local_dates = {(today_local + timedelta(days=d)) for d in range(1, 6)}

            # Guard shapes
            if isinstance(period_forecasts, dict) and isinstance(timestamps, (list, tuple)):
                # iterate all period entries; classify by localized datetime computed per-index
                for date_key, pmap in period_forecasts.items():
                    if not isinstance(pmap, dict):
                        continue
                    for pname, pdata in pmap.items():
                        if not isinstance(pdata, dict):
                            continue
                        indices = pdata.get("indices") or []
                        if not isinstance(indices, (list, tuple)):
                            continue
                        included_in_today = False
                        included_in_next_days = False
                        # Check each index until we know whether it belongs to today or next days
                        for idx in indices:
                            try:
                                if not isinstance(idx, int):
                                    continue
                                if idx < 0 or idx >= len(timestamps):
                                    continue
                                ts_raw = timestamps[idx]
                                dt_utc = _parse_dt(ts_raw)
                                if dt_utc is None:
                                    continue
                                # convert to HA local timezone for calendar logic / comparisons
                                dt_local = dt_util.as_local(dt_utc)
                                # remainder of today: same local date as today and timestamp >= now_local
                                if dt_local.date() == today_local and dt_local >= now_local:
                                    included_in_today = True
                                # next 5 days: local date in next_local_dates
                                if dt_local.date() in next_local_dates:
                                    included_in_next_days = True
                                # short-circuit if both true
                                if included_in_today and included_in_next_days:
                                    break
                            except Exception:
                                continue

                        # Add to remainder_of_today if it has any future local-hour in today's date
                        if included_in_today:
                            # Always remove nested profile_used; also hide indices/score_10 when expose_raw=False
                            try:
                                if self._expose_raw:
                                    pcopy = dict(pdata)
                                    pcopy.pop("profile_used", None)
                                    # augment components with values (prefer per-timestamp when available)
                                    try:
                                        comps = pcopy.get("components")
                                        score_calc_raw = None
                                        # no single score_calc_raw for a period; pass None so augment will fallback to period_entry values or indices
                                        if comps is not None:
                                            pcopy["components"] = _augment_components_with_values(
                                                comps,
                                                score_calc_raw,
                                                pcopy,
                                                pcopy.get("indices"),
                                                self.coordinator.data.get("per_timestamp_forecasts"),
                                                getattr(self.coordinator, "units", "metric") or "metric",
                                            )
                                    except Exception:
                                        pass
                                    # Attach grouped safety values for the period (from period_entry or per-timestamp fallbacks)
                                    try:
                                        safety_vals = _gather_safety_values_from_sources(
                                            None,
                                            pcopy,
                                            (self.coordinator.data.get("raw_payload") or {}).get("current") if isinstance(self.coordinator.data, dict) else None,
                                            getattr(self.coordinator, "units", "metric") or "metric",
                                        )
                                        if safety_vals:
                                            pcopy["safety_values"] = safety_vals
                                    except Exception:
                                        pass
                                    remainder_of_today[pname] = pcopy
                                else:
                                    sanitized_p = dict(pdata)
                                    sanitized_p.pop("indices", None)
                                    sanitized_p.pop("score_10", None)
                                    sanitized_p.pop("profile_used", None)
                                    # Remove per-component score_10 in components if present and add values too
                                    try:
                                        comps = sanitized_p.get("components")
                                        if comps is not None:
                                            sanitized_p["components"] = _augment_components_with_values(
                                                comps,
                                                None,
                                                sanitized_p,
                                                pdata.get("indices"),
                                                self.coordinator.data.get("per_timestamp_forecasts"),
                                                getattr(self.coordinator, "units", "metric") or "metric",
                                            )
                                    except Exception:
                                        pass
                                    # Attach grouped safety values for the sanitized period
                                    try:
                                        safety_vals = _gather_safety_values_from_sources(
                                            None,
                                            sanitized_p,
                                            None,
                                            getattr(self.coordinator, "units", "metric") or "metric",
                                        )
                                        if safety_vals:
                                            sanitized_p["safety_values"] = safety_vals
                                    except Exception:
                                        pass
                                    remainder_of_today[pname] = sanitized_p
                            except Exception:
                                # last-resort: preserve but attempt to remove profile_used
                                try:
                                    fallback = dict(pdata)
                                    fallback.pop("profile_used", None)
                                    remainder_of_today[pname] = fallback
                                except Exception:
                                    remainder_of_today[pname] = pdata

                        # Add to next_5_days grouped by the local date(s) that apply.
                        # A single period could span multiple local dates; include it under all applicable local date keys.
                        if included_in_next_days:
                            # Determine which local dates this period touches
                            touched_local_dates = set()
                            for idx in indices:
                                try:
                                    if not isinstance(idx, int):
                                        continue
                                    if idx < 0 or idx >= len(timestamps):
                                        continue
                                    ts_raw = timestamps[idx]
                                    dt_utc = _parse_dt(ts_raw)
                                    if dt_utc is None:
                                        continue
                                    dt_local = dt_util.as_local(dt_utc)
                                    if dt_local.date() in next_local_dates:
                                        touched_local_dates.add(dt_local.date())
                                except Exception:
                                    continue
                            for d in sorted(touched_local_dates):
                                key = d.isoformat()
                                try:
                                    if self._expose_raw:
                                        pcopy = dict(pdata)
                                        pcopy.pop("profile_used", None)
                                        try:
                                            comps = pcopy.get("components")
                                            if comps is not None:
                                                pcopy["components"] = _augment_components_with_values(
                                                    comps,
                                                    None,
                                                    pcopy,
                                                    pdata.get("indices"),
                                                    self.coordinator.data.get("per_timestamp_forecasts"),
                                                    getattr(self.coordinator, "units", "metric") or "metric",
                                                )
                                        except Exception:
                                            pass
                                        # Attach grouped safety values for the period
                                        try:
                                            safety_vals = _gather_safety_values_from_sources(
                                                None,
                                                pcopy,
                                                None,
                                                getattr(self.coordinator, "units", "metric") or "metric",
                                            )
                                            if safety_vals:
                                                pcopy["safety_values"] = safety_vals
                                        except Exception:
                                            pass
                                        next_5_days.setdefault(key, {})[pname] = pcopy
                                    else:
                                        sanitized_p = dict(pdata)
                                        sanitized_p.pop("indices", None)
                                        sanitized_p.pop("score_10", None)
                                        sanitized_p.pop("profile_used", None)
                                        try:
                                            comps = sanitized_p.get("components")
                                            if comps is not None:
                                                sanitized_p["components"] = _augment_components_with_values(
                                                    comps,
                                                    None,
                                                    sanitized_p,
                                                    pdata.get("indices"),
                                                    self.coordinator.data.get("per_timestamp_forecasts"),
                                                    getattr(self.coordinator, "units", "metric") or "metric",
                                                )
                                        except Exception:
                                            pass
                                        # Attach grouped safety values for the sanitized period
                                        try:
                                            safety_vals = _gather_safety_values_from_sources(
                                                None,
                                                sanitized_p,
                                                None,
                                                getattr(self.coordinator, "units", "metric") or "metric",
                                            )
                                            if safety_vals:
                                                sanitized_p["safety_values"] = safety_vals
                                        except Exception:
                                            pass
                                        next_5_days.setdefault(key, {})[pname] = sanitized_p
                                except Exception:
                                    try:
                                        fallback = dict(pdata)
                                        fallback.pop("profile_used", None)
                                        next_5_days.setdefault(key, {})[pname] = fallback
                                    except Exception:
                                        next_5_days.setdefault(key, {})[pname] = pdata
        except Exception:
            # Defensive: do not fail the sensor attribute creation; leave trimmed views empty on any error.
            remainder_of_today = {}
            next_5_days = {}

        attrs["remainder_of_today_periods"] = remainder_of_today
        attrs["next_5_day_periods"] = next_5_days

        # --- Now include per_timestamp_periods / period_forecasts (sanitized) ---
        if forecasts is not None:
            if self._expose_raw:
                attrs["per_timestamp_forecasts"] = forecasts
            else:
                _ATTR_LOGGER.debug("per_timestamp_forecasts suppressed (expose_raw=False) for sensor %s", self._attr_name)

        if period_forecasts is not None:
            # Always present a sanitized period_forecasts mapping for the user-facing attributes
            try:
                sanitized_pf: Dict[str, Any] = {}
                if isinstance(period_forecasts, dict):
                    for date_key, pmap in period_forecasts.items():
                        if not isinstance(pmap, dict):
                            continue
                        for pname, pdata in pmap.items():
                            if not isinstance(pdata, dict):
                                # preserve non-dict payloads as-is
                                sanitized_pf.setdefault(date_key, {})[pname] = pdata
                                continue
                            sp = dict(pdata)
                            sp.pop("indices", None)
                            sp.pop("score_10", None)
                            sp.pop("profile_used", None)
                            # Clean components: remove per-component score_10 if present and augment with values
                            comps = sp.get("components")
                            if comps is not None:
                                try:
                                    sp["components"] = _augment_components_with_values(
                                        comps,
                                        None,
                                        sp,
                                        pdata.get("indices"),
                                        self.coordinator.data.get("per_timestamp_forecasts"),
                                        getattr(self.coordinator, "units", "metric") or "metric",
                                    )
                                except Exception:
                                    sp.pop("components", None)
                            # Attach grouped safety values for the sanitized period entry
                            try:
                                safety_vals = _gather_safety_values_from_sources(
                                    None,
                                    sp,
                                    None,
                                    getattr(self.coordinator, "units", "metric") or "metric",
                                )
                                if safety_vals:
                                    sp["safety_values"] = safety_vals
                            except Exception:
                                pass
                            sanitized_pf.setdefault(date_key, {})[pname] = sp
                if sanitized_pf:
                    attrs["period_forecasts"] = sanitized_pf
                else:
                    _ATTR_LOGGER.debug("period_forecasts suppressed after sanitization (no usable entries) for sensor %s", self._attr_name)
            except Exception:
                _ATTR_LOGGER.debug("Failed to sanitize period_forecasts; suppressing for sensor %s", self._attr_name)

        # Raw payload and raw_output_enabled
        raw = self.coordinator.data.get("raw_payload") or self.coordinator.data

        # Expose raw payload only if explicitly enabled via configuration (expose_raw)
        if self._expose_raw:
            attrs["raw_payload"] = raw
        else:
            _ATTR_LOGGER.debug("raw_payload attribute suppressed (expose_raw=False) for sensor %s", self._attr_name)

        # Always indicate whether raw output is enabled so callers can confirm config state
        attrs["raw_output_enabled"] = bool(self._expose_raw)

        # Profile and safety detail pointers (raw canonical values)
        # Keep a single top-level profile_used only
        attrs["profile_used"] = (current.get("profile_used") if current else None)

        entry_units = getattr(self.coordinator, "units", "metric") or "metric"
        attrs["units"] = entry_units

        # Report canonical safety_limits on coordinator but present a display-friendly copy
        raw_limits = getattr(self.coordinator, "safety_limits", {}) or {}
        # Build a display-friendly safety_limits dict with wind/gust converted to display units
        display_limits: Dict[str, Any] = {}
        for k, v in raw_limits.items():
            if v is None:
                display_limits[k] = None
                continue
            try:
                if k in ("max_wind_m_s", "max_gust_m_s"):
                    # convert m/s -> km/h or mph
                    if entry_units == "metric":
                        display_limits[k] = _round_opt(unit_helpers.m_s_to_kmh(v), 2)
                    elif entry_units == "imperial":
                        display_limits[k] = _round_opt(unit_helpers.m_s_to_mph(v), 2)
                    else:
                        display_limits[k] = _round_opt(v, 2)
                else:
                    # Keep canonical metric for other values, but round
                    display_limits[k] = _round_opt(v, 3)
            except Exception:
                display_limits[k] = v
        attrs["safety_limits"] = display_limits

        # Canonicalize safety object: keep machine-readable codes under "reason_codes" and human strings under "reason_strings"
        if current and isinstance(current.get("safety"), dict):
            safety_obj = dict(current.get("safety") or {})
            # Normalize legacy "reasons" -> "reason_codes"
            if "reasons" in safety_obj and "reason_codes" not in safety_obj:
                safety_obj["reason_codes"] = list(safety_obj.get("reasons") or [])
                safety_obj.pop("reasons", None)
            else:
                safety_obj["reason_codes"] = list(safety_obj.get("reason_codes") or [])
            # Ensure human-readable strings are available
            safety_obj["reason_strings"] = list(safety_obj.get("reason_strings") or [])
            attrs["safety"] = safety_obj
        else:
            attrs["safety"] = None

        # Extract formatted-weather (preferred) and raw current snapshot (fallback)
        formatted = None
        try:
            if current and isinstance(current.get("forecast_raw"), dict):
                formatted = current.get("forecast_raw", {}).get("formatted_weather") or {}
        except Exception:
            formatted = {}

        raw_current = None
        try:
            if isinstance(raw, dict):
                raw_current = raw.get("current")
        except Exception:
            raw_current = None

        # Helper to read a key from formatted then raw_current (return None if missing)
        def _pick_raw(*keys):
            for k in keys:
                try:
                    if formatted and k in formatted:
                        return formatted.get(k)
                except Exception:
                    pass
                try:
                    if raw_current and k in raw_current:
                        return raw_current.get(k)
                except Exception:
                    pass
            return None

        # Provide top-level attributes with conversions/rounding applied for final outputs
        # Temperature (keep °C; rounding to 1 dp)
        temp = _pick_raw("temperature", "temp", "temperature_c")
        try:
            attrs["current_temperature"] = _round_opt(float(temp), 1) if temp is not None else None
        except Exception:
            attrs["current_temperature"] = temp

        # Wind speed & gust: convert to display units (km/h or mph) using unit_helpers
        raw_wind_val, raw_wind_hint = self._extract_wind_ms_and_hint(formatted, raw_current)
        wind_disp_val, wind_unit_label = self._to_display_wind(raw_wind_val, raw_wind_hint, entry_units)
        attrs["current_wind_speed"] = wind_disp_val
        attrs["current_wind_unit"] = wind_unit_label

        # Gust
        # Try formatted wind_gust first (canonical m/s), else raw_current keys
        try:
            raw_gust_val = None
            raw_gust_hint = None
            if formatted and formatted.get("wind_gust") is not None:
                raw_gust_val = formatted.get("wind_gust")
                raw_gust_hint = "m/s"
            elif raw_current:
                for k in ("wind_gust", "wind_max_m_s", "windgusts_10m", "wind_max"):
                    if k in raw_current and raw_current.get(k) is not None:
                        raw_gust_val = raw_current.get(k)
                        # unit hint if available
                        raw_gust_hint = raw_current.get("wind_unit") or None
                        break
            gust_disp_val, gust_unit_label = self._to_display_wind(raw_gust_val, raw_gust_hint, entry_units)
            attrs["current_wind_gust"] = gust_disp_val
            attrs["current_wind_gust_unit"] = gust_unit_label
        except Exception:
            attrs["current_wind_gust"] = None
            attrs["current_wind_gust_unit"] = None

        # Pressure (hPa) round 1 decimal
        try:
            p = _pick_raw("pressure_hpa", "pressure")
            attrs["current_pressure_hpa"] = _round_opt(float(p), 1) if p is not None else None
        except Exception:
            attrs["current_pressure_hpa"] = p

        # Cloud cover (int)
        try:
            cc = _pick_raw("cloud_cover", "cloud", "cloudcover")
            attrs["current_cloud_cover"] = int(round(float(cc))) if cc is not None else None
        except Exception:
            attrs["current_cloud_cover"] = None

        # Precip probability (int)
        try:
            pop = _pick_raw("precipitation_probability", "pop")
            attrs["current_precipitation_probability"] = int(round(float(pop))) if pop is not None else None
        except Exception:
            attrs["current_precipitation_probability"] = None

        # Visibility (km) round to 2 dp
        try:
            vis = _pick_raw("visibility", "visibility_km")
            attrs["current_visibility_km"] = _round_opt(vis, 2) if vis is not None else None
        except Exception:
            attrs["current_visibility_km"] = None

        # Wave / swell metrics (keep meters/seconds canonical; round)
        try:
            wh = _pick_raw("wave_height_m", "wave_height", "wave_height")
            attrs["current_wave_height_m"] = _round_opt(wh, 3) if wh is not None else None
        except Exception:
            attrs["current_wave_height_m"] = None
        try:
            wp = _pick_raw("wave_period_s", "wave_period", "wave_period_s")
            attrs["current_wave_period_s"] = _round_opt(wp, 2) if wp is not None else None
        except Exception:
            attrs["current_wave_period_s"] = None
        try:
            sh = _pick_raw("swell_height_m", "swell_wave_height", "swell_height")
            attrs["current_swell_height_m"] = _round_opt(sh, 3) if sh is not None else None
        except Exception:
            attrs["current_swell_height_m"] = None
        try:
            sp = _pick_raw("swell_period_s", "swell_period_s")
            attrs["current_swell_period_s"] = _round_opt(sp, 1) if sp is not None else None
        except Exception:
            attrs["current_swell_period_s"] = None

        # --- Tide & Moon: prefer canonical tide dict produced by TideProxy ---
        next_high_obj = None
        next_low_obj = None
        moon_numeric = None
        try:
            tide_obj = None
            if isinstance(self.coordinator.data, dict) and "tide" in self.coordinator.data:
                tide_obj = self.coordinator.data.get("tide")
            if (not tide_obj or not isinstance(tide_obj, dict)) and isinstance(raw, dict):
                tide_obj = raw.get("tide") or tide_obj

            if isinstance(tide_obj, dict):
                tide_ts = tide_obj.get("timestamps") or tide_obj.get("time")
                tide_heights = tide_obj.get("tide_height_m") or tide_obj.get("height_m") or tide_obj.get("height")
                raw_next_high = tide_obj.get("next_high")
                raw_next_low = tide_obj.get("next_low")

                if raw_next_high:
                    parsed = _parse_dt(raw_next_high)
                    iso = _iso_z(parsed) if parsed else str(raw_next_high)
                    height = None
                    if parsed and tide_ts and tide_heights and isinstance(tide_ts, (list, tuple)) and isinstance(tide_heights, (list, tuple)):
                        height = _find_height_for_timestamp(iso, list(tide_ts), list(tide_heights))
                    if height is not None:
                        try:
                            height = round(float(height), 3)
                        except Exception:
                            pass
                    next_high_obj = {"timestamp": iso, "height_m": height}

                if raw_next_low:
                    parsed = _parse_dt(raw_next_low)
                    iso = _iso_z(parsed) if parsed else str(raw_next_low)
                    height = None
                    if parsed and tide_ts and tide_heights and isinstance(tide_ts, (list, tuple)) and isinstance(tide_heights, (list, tuple)):
                        height = _find_height_for_timestamp(iso, list(tide_ts), list(tide_heights))
                    if height is not None:
                        try:
                            height = round(float(height), 3)
                        except Exception:
                            pass
                    next_low_obj = {"timestamp": iso, "height_m": height}

                if "tide_phase" in tide_obj:
                    try:
                        moon_numeric = float(tide_obj.get("tide_phase")) if tide_obj.get("tide_phase") is not None else None
                    except Exception:
                        moon_numeric = None
        except Exception:
            next_high_obj = None
            next_low_obj = None
            moon_numeric = None

        # Fallback moon_phase from other locations if not provided by tide dict
        if moon_numeric is None:
            try:
                mp = self.coordinator.data.get("moon_phase") if isinstance(self.coordinator.data, dict) else None
                if isinstance(mp, (list, tuple)):
                    moon_numeric = float(mp[0]) if len(mp) > 0 else None
                else:
                    moon_numeric = float(mp) if mp is not None else None
            except Exception:
                moon_numeric = None
        if moon_numeric is None:
            try:
                rp_mp = raw.get("moon_phase") if isinstance(raw, dict) else None
                if isinstance(rp_mp, (list, tuple)):
                    moon_numeric = float(rp_mp[0]) if len(rp_mp) > 0 else None
                else:
                    moon_numeric = float(rp_mp) if rp_mp is not None else None
            except Exception:
                moon_numeric = None

        attrs["moon_phase"] = _round_opt(moon_numeric, 6) if moon_numeric is not None else None
        attrs["moon_phase_name"] = _moon_phase_name(moon_numeric)

        attrs["next_high_tide"] = next_high_obj
        attrs["next_low_tide"] = next_low_obj

        # Attribution to appear both as explicit key and HA-standard ATTR_ATTRIBUTION for older integrations
        attrs["attribution"] = ATTRIBUTION
        attrs[ATTR_ATTRIBUTION] = ATTRIBUTION
        return attrs


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities):
    """
    Entry setup for the single OFA sensor.
    """
    coordinator = hass.data[DOMAIN].get(entry.entry_id) if hass.data.get(DOMAIN) else None
    if coordinator is None:
        raise RuntimeError("Coordinator not found in hass.data for this config entry (strict)")

    try:
        sensor_name = entry.data.get(CONF_NAME)
    except Exception:
        raise RuntimeError("Failed to read config entry data; 'name' (CONF_NAME) is required (strict)")

    if not sensor_name:
        raise RuntimeError(
            "Missing required name in config entry data; the Ocean Fishing Assistant integration requires the user to set a name during configuration (strict)."
        )

    # Read expose_raw from entry.options first (preferred), then fall back to entry.data for backward compatibility.
    try:
        expose_raw = bool(entry.options.get("expose_raw")) if isinstance(entry.options, dict) else False
    except Exception:
        expose_raw = False
    if not expose_raw:
        try:
            expose_raw = bool(entry.data.get("expose_raw")) if isinstance(entry.data, dict) else False
        except Exception:
            expose_raw = False

    prefix = "ocean_fishing_assistant"
    final_name = sensor_name if sensor_name.startswith(prefix) else f"{prefix}_{sensor_name}"

    async_add_entities([OFASensor(coordinator, name=final_name, expose_raw=expose_raw)], update_before_add=True)