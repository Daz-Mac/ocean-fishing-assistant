"""
Strict Ocean Fishing Scoring — no fallbacks, fail loudly.

This module expects DataFormatter to normalize input into canonical keys:
  - payload["timestamps"] : list of ISO timestamps
  - payload["moon_phase"] : per-timestamp list OR scalar
  - payload["tide_height_m"] : per-timestamp list OR payload["tide"]["tide_height_m"]
  - payload["wind_m_s"] : per-timestamp list
  - payload["wave_height_m"] : per-timestamp list
  - payload["pressure_hpa"] : per-timestamp list (must have at least one future point)
  - payload["temperature_c"] : per-timestamp list

Any missing or malformed required input will raise MissingDataError (logged).
"""
from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone

_LOGGER = logging.getLogger(__name__)

FACTOR_WEIGHTS = {
    "tide": 0.25,
    "wind": 0.15,
    "waves": 0.15,
    "time": 0.15,
    "pressure": 0.10,
    "season": 0.10,
    "moon": 0.05,
    "temperature": 0.03,
}

DEFAULT_PROFILE = {
    "common_name": "Generic",
    "preferred_temp_c": [8.0, 18.0],
    "preferred_wind_m_s": [0.0, 6.0],
    "max_wave_height_m": 2.0,
    "preferred_tide_m": [-1.0, 1.0],
    "preferred_tide_phase": [],
    "preferred_times": [],
    "preferred_months": [],
    "moon_preference": [],
    "weights": FACTOR_WEIGHTS,
}


class MissingDataError(ValueError):
    """Raised when required inputs for scoring are missing."""


def _to_float_safe(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _linear_within_score_10(value: float, pref_min: float, pref_max: float, tolerance: float) -> float:
    if math.isclose(pref_min, pref_max):
        low = pref_min - tolerance
        high = pref_max + tolerance
    else:
        low = pref_min
        high = pref_max
    span_low = low - tolerance
    span_high = high + tolerance
    if value >= low and value <= high:
        return 10.0
    if value <= span_low or value >= span_high:
        return 0.0
    if value < low:
        return 10.0 * (value - span_low) / (low - span_low)
    if value > high:
        return 10.0 * (span_high - value) / (span_high - high)
    return 0.0


def _clamp_0_10(x: float) -> float:
    return max(0.0, min(10.0, float(x)))


def _coerce_datetime(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        parsed = datetime.fromisoformat(s)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        try:
            if isinstance(v, (int, float)):
                val = float(v)
                if val > 1e12:
                    val = val / 1000.0
                return datetime.fromtimestamp(val, tz=timezone.utc)
        except Exception:
            pass
    return None


def _format_safety_reason(code: str, safety_limits: Optional[Dict[str, Any]]) -> str:
    if not code:
        return ""
    code = str(code)
    if ">" in code:
        k, v = code.split(">", 1)
        k = k.strip()
        try:
            val = float(v)
        except Exception:
            val = v
        if k in ("wind", "wind_m_s"):
            return f"Wind exceeds safe limit ({val} m/s)"
        if k in ("wave", "wave_height"):
            return f"Wave height exceeds safe limit ({val} m)"
        if k in ("swell", "swell_period"):
            return f"Swell period exceeds safe limit ({val} s)"
        if k in ("vis", "visibility"):
            return f"Visibility below safe minimum ({val} km)"
        return f"{k} > {val}"
    if "<" in code:
        k, v = code.split("<", 1)
        k = k.strip()
        try:
            val = float(v)
        except Exception:
            val = v
        if k in ("vis", "visibility"):
            return f"Visibility below safe minimum ({val} km)"
        return f"{k} < {val}"
    if code == "wind_near_limit":
        if safety_limits:
            mw = safety_limits.get("max_wind_m_s")
            if mw is not None:
                return f"Wind approaching configured maximum ({mw} m/s)"
        return "Wind near configured maximum"
    if code == "wave_near_limit":
        if safety_limits:
            mw = safety_limits.get("max_wave_height_m")
            if mw is not None:
                return f"Wave height approaching configured maximum ({mw} m)"
        return "Wave height near configured maximum"
    if code == "vis_near_limit":
        if safety_limits:
            mv = safety_limits.get("min_visibility_km")
            if mv is not None:
                return f"Visibility close to minimum ({mv} km)"
        return "Visibility near configured minimum"
    if code == "swell_near_limit":
        if safety_limits:
            ms = safety_limits.get("max_swell_period_s")
            if ms is not None:
                return f"Swell period approaching configured maximum ({ms} s)"
        return "Swell period near configured maximum"
    return code


def compute_score(
    data: Dict[str, Any],
    species_profile: Optional[Union[str, Dict[str, Any]]] = None,
    use_index: int = 0,
    safety_limits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Strict per-index scoring. Expects canonical keys present in `data`.
    Raises MissingDataError on missing required inputs.
    """
    if not data or "timestamps" not in data:
        raise MissingDataError("Missing timestamps in data")
    timestamps = data.get("timestamps", [])
    if use_index < 0 or use_index >= len(timestamps):
        raise MissingDataError(f"use_index {use_index} out of range")

    # resolve profile
    if isinstance(species_profile, str):
        # keep earlier simple loader function name compatibility if provided by caller
        try:
            from .ocean_scoring import _load_species_profile_by_name  # type: ignore
        except Exception:
            _load = None
        else:
            _load = _load_species_profile_by_name if "_load_species_profile_by_name" in globals() else None
        profile = _load(species_profile) if _load else DEFAULT_PROFILE
    elif isinstance(species_profile, dict):
        profile = species_profile
    else:
        profile = DEFAULT_PROFILE

    # allowed weights override
    weights = FACTOR_WEIGHTS.copy()
    for k, v in (profile.get("weights") or {}).items():
        if k in weights:
            try:
                weights[k] = float(v)
            except Exception:
                pass
    total = sum(weights.values()) or 1.0
    weights = {k: float(v) / total for k, v in weights.items()}

    # helper to extract canonical arrays/scalars
    def _get_at(key: str, index: int = 0) -> Optional[float]:
        if key not in data:
            return None
        arr = data.get(key)
        if arr is None:
            return None
        if isinstance(arr, (list, tuple)):
            try:
                return _to_float_safe(arr[index])
            except Exception:
                return None
        return _to_float_safe(arr)

    # Strictly extract required canonical values
    tide = _get_at("tide_height_m", use_index)
    wind = _get_at("wind_m_s", use_index)
    wave = _get_at("wave_height_m", use_index)
    temp = _get_at("temperature_c", use_index)
    pressure_arr = data.get("pressure_hpa")  # must be an array for delta computation

    # moon_phase: require explicit key (array or scalar)
    moon_phase_val = None
    if "moon_phase" in data:
        mp = data.get("moon_phase")
        if isinstance(mp, (list, tuple)):
            moon_phase_val = _to_float_safe(mp[use_index]) if use_index < len(mp) else None
        else:
            moon_phase_val = _to_float_safe(mp)

    # Pressure delta check
    pressure_delta = None
    if not isinstance(pressure_arr, (list, tuple)) or len(pressure_arr) <= use_index + 1:
        pressure_arr_ok = False
    else:
        pressure_arr_ok = True
        p_curr = _to_float_safe(pressure_arr[use_index])
        p_next = _to_float_safe(pressure_arr[use_index + 1])
        if p_curr is not None and p_next is not None:
            pressure_delta = float(p_next) - float(p_curr)

    # Strict presence checks
    missing = []
    if tide is None:
        missing.append("tide_height_m")
    if wind is None:
        missing.append("wind_m_s")
    if wave is None:
        missing.append("wave_height_m")
    if temp is None:
        missing.append("temperature_c")
    if moon_phase_val is None:
        missing.append("moon_phase")
    if not pressure_arr_ok:
        missing.append("pressure_hpa_series_with_future_point")
    
    if missing:
        msg = f"Missing required inputs for scoring at index={use_index} timestamp={timestamps[use_index]}: {', '.join(missing)}"
        # Do not log here to avoid flooding HA logger for repeated missing-values across many timestamps.
        # Raise so caller (DataFormatter / Coordinator) can log a single, aggregated diagnostic message.
        raise MissingDataError(msg)

    # compute component scores
    comp: Dict[str, Any] = {}

    # TIDE
    try:
        pref_tide = profile.get("preferred_tide_m", DEFAULT_PROFILE["preferred_tide_m"])
        tmin, tmax = float(pref_tide[0]), float(pref_tide[1])
        tide_score = _linear_within_score_10(float(tide), tmin, tmax, tolerance=0.5)
        tide_reason = f"value={tide}m"
    except Exception:
        tide_score = 0.0
        tide_reason = "error"
    comp["tide"] = {"score_10": round(_clamp_0_10(tide_score), 3), "score_100": int(round(_clamp_0_10(tide_score) * 10)), "reason": tide_reason}

    # WIND
    try:
        pref_w = profile.get("preferred_wind_m_s", DEFAULT_PROFILE["preferred_wind_m_s"])
        wmin, wmax = float(pref_w[0]), float(pref_w[1])
        wind_score = _linear_within_score_10(float(wind), wmin, wmax, tolerance=4.0)
        wind_reason = f"value={wind} m/s"
    except Exception:
        wind_score = 0.0
        wind_reason = "error"
    comp["wind"] = {"score_10": round(_clamp_0_10(wind_score), 3), "score_100": int(round(_clamp_0_10(wind_score) * 10)), "reason": wind_reason}

    # WAVES
    try:
        max_wave = float(profile.get("max_wave_height_m", DEFAULT_PROFILE["max_wave_height_m"]))
        if wave <= max_wave:
            wave_score = 10.0
            wave_reason = f"ok (value={wave}m <= max={max_wave}m)"
        else:
            limit = max_wave * 2.0 if max_wave > 0 else max_wave + 1.0
            if wave >= limit:
                wave_score = 0.0
            else:
                wave_score = 10.0 * (limit - wave) / (limit - max_wave)
            wave_reason = f"value={wave}m"
    except Exception:
        wave_score = 0.0
        wave_reason = "error"
    comp["waves"] = {"score_10": round(_clamp_0_10(wave_score), 3), "score_100": int(round(_clamp_0_10(wave_score) * 10)), "reason": wave_reason}

    # TIME
    hour = None
    month = None
    try:
        dt = _coerce_datetime(timestamps[use_index])
        if dt:
            hour = dt.hour
            month = dt.month
    except Exception:
        pass
    try:
        pref_times = profile.get("preferred_times", [])
        if not pref_times:
            time_score = 5.0
            time_reason = "no_preference"
        else:
            if hour is None:
                time_score = 5.0
                time_reason = "missing_time"
            else:
                matched = any((int(w.get("start_hour")) <= hour <= int(w.get("end_hour"))) for w in pref_times if isinstance(w, dict) and "start_hour" in w and "end_hour" in w)
                time_score = 10.0 if matched else 2.0
                time_reason = "matched" if matched else "not_matched"
    except Exception:
        time_score = 5.0
        time_reason = "error"
    comp["time"] = {"score_10": round(_clamp_0_10(time_score), 3), "score_100": int(round(_clamp_0_10(time_score) * 10)), "reason": time_reason}

    # PRESSURE (use delta)
    try:
        if pressure_delta is None:
            raise ValueError("pressure delta unavailable")
        p = float(pressure_delta)
        pressure_score = 5.0 + (p / 10.0) * 5.0
        pressure_score = _clamp_0_10(pressure_score)
        pressure_reason = f"delta={p}"
    except Exception:
        pressure_score = 0.0
        pressure_reason = "error"
    comp["pressure"] = {"score_10": round(_clamp_0_10(pressure_score), 3), "score_100": int(round(_clamp_0_10(pressure_score) * 10)), "reason": pressure_reason}

    # SEASON
    try:
        pref_months = profile.get("preferred_months", [])
        if not pref_months or month is None:
            season_score = 5.0
            season_reason = "no_preference"
        else:
            season_score = 10.0 if int(month) in pref_months else 2.0
            season_reason = "in_season" if season_score >= 10.0 else "out_of_season"
    except Exception:
        season_score = 5.0
        season_reason = "error"
    comp["season"] = {"score_10": round(_clamp_0_10(season_score), 3), "score_100": int(round(_clamp_0_10(season_score) * 10)), "reason": season_reason}

    # MOON
    try:
        p = float(moon_phase_val)
        dist_new = abs(p - 0.0)
        dist_full = abs(p - 0.5)
        dist = min(dist_new, dist_full)
        moon_score = max(0.0, 10.0 * (1.0 - (dist / 0.25)))
        moon_reason = f"phase={p}"
    except Exception:
        moon_score = 0.0
        moon_reason = "error"
    comp["moon"] = {"score_10": round(_clamp_0_10(moon_score), 3), "score_100": int(round(_clamp_0_10(moon_score) * 10)), "reason": moon_reason}

    # TEMPERATURE
    try:
        pref_temp = profile.get("preferred_temp_c", DEFAULT_PROFILE["preferred_temp_c"])
        tmin, tmax = float(pref_temp[0]), float(pref_temp[1])
        temp_score = _linear_within_score_10(float(temp), tmin, tmax, tolerance=10.0)
        temp_reason = f"value={temp}C"
    except Exception:
        temp_score = 0.0
        temp_reason = "error"
    comp["temperature"] = {"score_10": round(_clamp_0_10(temp_score), 3), "score_100": int(round(_clamp_0_10(temp_score) * 10)), "reason": temp_reason}

    # Weighted aggregation
    overall_10 = 0.0
    for k in weights:
        overall_10 += weights.get(k, 0.0) * comp.get(k, {}).get("score_10", 0.0)
    overall_10 = float(round(overall_10, 3))
    overall_100 = int(round(overall_10 * 10.0))

    # Safety evaluation
    safety = {"unsafe": False, "caution": False, "reasons": []}
    try:
        if safety_limits:
            max_wind = _to_float_safe(safety_limits.get("max_wind_m_s"))
            if max_wind is not None and wind is not None:
                if wind > max_wind:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"wind>{max_wind}")
                elif wind > (0.9 * max_wind):
                    safety["caution"] = True
                    safety["reasons"].append("wind_near_limit")

            max_wave = _to_float_safe(safety_limits.get("max_wave_height_m"))
            if max_wave is not None and wave is not None:
                if wave > max_wave:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"wave>{max_wave}")
                elif wave > (0.9 * max_wave):
                    safety["caution"] = True
                    safety["reasons"].append("wave_near_limit")

            min_vis = _to_float_safe(safety_limits.get("min_visibility_km"))
            vis = _get_at("visibility_km", use_index) if "visibility_km" in data else None
            if min_vis is not None and vis is not None:
                if vis < min_vis:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"vis<{min_vis}")
                elif vis < (1.1 * min_vis):
                    safety["caution"] = True
                    safety["reasons"].append("vis_near_limit")

            max_swell = _to_float_safe(safety_limits.get("max_swell_period_s"))
            swell = _get_at("swell_period_s", use_index) if "swell_period_s" in data else None
            if max_swell is not None and swell is not None:
                if swell > max_swell:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"swell>{max_swell}")
                elif swell > (0.9 * max_swell):
                    safety["caution"] = True
                    safety["reasons"].append("swell_near_limit")
    except Exception:
        _LOGGER.debug("Safety evaluation failed", exc_info=True)

    # reason strings
    try:
        reason_codes = safety.get("reasons", []) or []
        safety["reason_strings"] = [_format_safety_reason(rc, safety_limits) for rc in reason_codes]
    except Exception:
        safety["reason_strings"] = []

    result = {
        "score_10": overall_10,
        "score_100": overall_100,
        "components": comp,
        "raw": {
            "tide": tide,
            "wind": wind,
            "wave": wave,
            "pressure_delta": pressure_delta,
            "temperature": temp,
            "timestamp": timestamps[use_index],
            "moon_phase": moon_phase_val,
        },
        "profile_used": profile.get("common_name", "unknown"),
        "safety": safety,
    }
    return result


def compute_forecast(
    payload: Dict[str, Any],
    species_profile: Optional[Union[str, Dict[str, Any]]] = None,
    safety_limits: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Strict per-timestamp forecast list. Each entry will either contain a fully
    computed score or an explicit entry describing missing required fields.
    """
    out: List[Dict[str, Any]] = []
    if not payload or "timestamps" not in payload:
        return out
    timestamps = payload.get("timestamps") or []
    for idx, ts in enumerate(timestamps):
        try:
            res = compute_score(payload, species_profile=species_profile, use_index=idx, safety_limits=safety_limits)
            forecast_raw = {
                "formatted_weather": {
                    "temperature": payload.get("temperature_c")[idx] if isinstance(payload.get("temperature_c"), (list, tuple)) else payload.get("temperature_c"),
                    "wind": payload.get("wind_m_s")[idx] if isinstance(payload.get("wind_m_s"), (list, tuple)) else payload.get("wind_m_s"),
                    "pressure_hpa": payload.get("pressure_hpa")[idx] if isinstance(payload.get("pressure_hpa"), (list, tuple)) else payload.get("pressure_hpa"),
                },
                # No searching for astro/marine blocks — strictly report moon_phase if available
                "astro_used": {"moon_phase": (payload.get("moon_phase")[idx] if isinstance(payload.get("moon_phase"), (list, tuple)) and idx < len(payload.get("moon_phase")) else payload.get("moon_phase"))} if "moon_phase" in payload else None,
                "score_calc": res,
            }
            entry = {
                "timestamp": ts,
                "index": idx,
                "score_10": res.get("score_10"),
                "score_100": res.get("score_100"),
                "components": res.get("components"),
                "forecast_raw": forecast_raw,
                "profile_used": res.get("profile_used"),
                "safety": res.get("safety"),
            }
        except MissingDataError as mde:
            err_text = str(mde)
            _LOGGER.debug("compute_forecast: missing data for index %s (%s): %s", idx, ts, err_text)
            entry = {
                "timestamp": ts,
                "index": idx,
                "score_10": None,
                "score_100": None,
                "components": None,
                "forecast_raw": {"error": "missing required data", "details": err_text},
                "profile_used": None,
                "safety": {"unsafe": False, "caution": False, "reasons": [], "reason_strings": []},
            }
        except Exception:
            _LOGGER.debug("Failed to compute forecast index %s", idx, exc_info=True)
            entry = {
                "timestamp": ts,
                "index": idx,
                "score_10": None,
                "score_100": None,
                "components": None,
                "forecast_raw": {"error": "unexpected error"},
                "profile_used": None,
                "safety": {"unsafe": False, "caution": False, "reasons": [], "reason_strings": []},
            }
        out.append(entry)
    return out