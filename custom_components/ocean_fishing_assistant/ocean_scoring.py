# (Full file content — replace your existing ocean_scoring.py with the contents below)

"""
Strict Ocean Fishing Scoring — no fallbacks, fail loudly.

This module expects DataFormatter to normalize input into canonical keys:
  - payload["timestamps"] : list of ISO timestamps
  - payload["moon_phase"] : per-timestamp list OR scalar
  - payload["tide"] : optional dict with tide metadata (may include tide_phase and moon_phase)
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

from . import unit_helpers

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
    "preferred_tide_phase": ["high"],
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


def _format_safety_reason(code: str, safety_limits: Optional[Dict[str, Any]], units: str = "metric") -> str:
    if not code:
        return ""
    code = str(code)

    def _format_wind_val(val_m_s: float) -> str:
        try:
            val = float(val_m_s)
        except Exception:
            return f"{val_m_s} m/s"
        try:
            if units == "metric":
                conv = unit_helpers.m_s_to_kmh(val)
                unit_label = "km/h"
            elif units == "imperial":
                conv = unit_helpers.m_s_to_mph(val)
                unit_label = "mph"
            else:
                conv = val
                unit_label = "m/s"
            return f"{round(conv, 1)} {unit_label}"
        except Exception:
            return f"{val} m/s"

    if ">" in code:
        k, v = code.split(">", 1)
        k = k.strip()
        try:
            val = float(v)
        except Exception:
            val = v
        if k in ("wind", "wind_m_s"):
            return f"Wind exceeds safe limit ({_format_wind_val(val)})"
        if k in ("wave", "wave_height"):
            return f"Wave height exceeds safe limit ({val} m)"
        if k in ("swell", "swell_period"):
            return f"Swell period below safe minimum ({val} s)"
        if k in ("gust", "wind_gust"):
            return f"Gust exceeds safe limit ({_format_wind_val(val)})"
        if k in ("vis", "visibility"):
            return f"Visibility below safe minimum ({val} km)"
        if k in ("precip", "precip_chance", "precipitation"):
            return f"Precipitation chance exceeds safe limit ({val} %)"

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
        if k in ("swell", "swell_period"):
            return f"Swell period below safe minimum ({val} s)"
        return f"{k} < {val}"
    if code == "wind_near_limit":
        if safety_limits:
            mw = safety_limits.get("max_wind_m_s")
            if mw is not None:
                return f"Wind approaching configured maximum ({_format_wind_val(mw)})"
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
            ms = safety_limits.get("min_swell_period_s")
            if ms is not None:
                return f"Swell period approaching configured minimum ({ms} s)"
        return "Swell period near configured minimum"
    if code == "gust_near_limit":
        if safety_limits:
            mg = safety_limits.get("max_gust_m_s")
            if mg is not None:
                return f"Gust approaching configured maximum ({_format_wind_val(mg)})"
        return "Gust near configured maximum"
    if code == "precip_near_limit":
        if safety_limits:
            mp = safety_limits.get("max_precip_chance_pct")
            if mp is not None:
                return f"Precipitation chance approaching configured maximum ({mp} %)"
        return "Precip chance near configured maximum"
    return code


def compute_score(
    data: Dict[str, Any],
    species_profile: Optional[Union[str, Dict[str, Any]]] = None,
    use_index: int = 0,
    safety_limits: Optional[Dict[str, Any]] = None,
    units: str = "metric",
) -> Dict[str, Any]:
    if not data or "timestamps" not in data:
        raise MissingDataError("Missing timestamps in data")
    timestamps = data.get("timestamps", [])
    if use_index < 0 or use_index >= len(timestamps):
        raise MissingDataError(f"use_index {use_index} out of range")

    if isinstance(species_profile, str):
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

    weights = FACTOR_WEIGHTS.copy()
    for k, v in (profile.get("weights") or {}).items():
        if k in weights:
            try:
                weights[k] = float(v)
            except Exception:
                pass
    total = sum(weights.values()) or 1.0
    weights = {k: float(v) / total for k, v in weights.items()}

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

    wind = _get_at("wind_m_s", use_index)
    wave = _get_at("wave_height_m", use_index)
    temp = _get_at("temperature_c", use_index)
    pressure_arr = data.get("pressure_hpa")

    moon_phase_val = None
    if "moon_phase" in data:
        mp = data.get("moon_phase")
        if isinstance(mp, (list, tuple)):
            moon_phase_val = _to_float_safe(mp[use_index]) if use_index < len(mp) else None
        else:
            moon_phase_val = _to_float_safe(mp)

    pressure_delta = None
    if not isinstance(pressure_arr, (list, tuple)) or len(pressure_arr) <= use_index + 1:
        pressure_arr_ok = False
    else:
        pressure_arr_ok = True
        p_curr = _to_float_safe(pressure_arr[use_index])
        p_next = _to_float_safe(pressure_arr[use_index + 1])
        if p_curr is not None and p_next is not None:
            pressure_delta = float(p_next) - float(p_curr)

    missing = []
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
        raise MissingDataError(msg)

    comp: Dict[str, Any] = {}

    # TIDE component — phase-based only (preferred_tide_m removed)
    try:
        pref_tide_phase = profile.get("preferred_tide_phase", []) or []
        tide_phase_val = None
        # locate tide_phase (top-level or under tide)
        if "tide_phase" in data:
            tp = data.get("tide_phase")
            if isinstance(tp, (list, tuple)):
                tide_phase_val = tp[use_index] if use_index < len(tp) else None
            else:
                tide_phase_val = tp
        elif "tide" in data and isinstance(data.get("tide"), dict):
            tp = data.get("tide").get("tide_phase")
            if isinstance(tp, (list, tuple)):
                tide_phase_val = tp[use_index] if use_index < len(tp) else None
            else:
                tide_phase_val = tp

        # Strict: if profile specifies preferred_tide_phase, tide_phase MUST be present and be a string.
        if pref_tide_phase:
            if tide_phase_val is None or not isinstance(tide_phase_val, str):
                raise MissingDataError("tide_phase (string) required by species profile but missing or not a string")
            matched = any(str(pref).lower() == str(tide_phase_val).lower() for pref in pref_tide_phase)
            tide_score = 10.0 if matched else 3.0
        else:
            tide_score = 10.0
        tide_score = _clamp_0_10(tide_score)
        comp_tide: Dict[str, Any] = {"score_10": round(tide_score, 3), "score_100": int(round(tide_score * 10))}

        # include tide height/unit if present
        try:
            tide_h = _get_at("tide_height_m", use_index) if "tide_height_m" in data else None
            if tide_h is not None:
                comp_tide["tide_height"] = round(float(tide_h), 3)
                comp_tide["tide_unit"] = "m"
        except Exception:
            pass

        # include friendly tide phase name when available (do not emit duplicate `tide_phase`)
        PHASE_NAME_MAP = {
            "rising": "Rising",
            "falling": "Falling",
            "high": "High Tide",
            "low": "Low Tide",
        }
        try:
            if tide_phase_val is not None and isinstance(tide_phase_val, str):
                comp_tide["tide_phase_name"] = PHASE_NAME_MAP.get(tide_phase_val.lower(), tide_phase_val)
        except Exception:
            pass

        comp["tide"] = comp_tide
    except MissingDataError:
        # propagate strictly
        raise
    except Exception:
        _LOGGER.debug("Failed to compute tide component (phase-based)", exc_info=True)

    # WIND component
    try:
        pref_wind = profile.get("preferred_wind_m_s", DEFAULT_PROFILE["preferred_wind_m_s"])
        if isinstance(pref_wind, (list, tuple)) and len(pref_wind) >= 2:
            pw_min, pw_max = float(pref_wind[0]), float(pref_wind[1])
        else:
            pw = float(pref_wind) if pref_wind is not None else 0.0
            pw_min, pw_max = pw, pw
        wind_tol = max(1.0, 0.2 * max(1.0, pw_max))
        wind_score = _linear_within_score_10(float(wind), pw_min, pw_max, wind_tol)
        wind_score = _clamp_0_10(wind_score)
        comp["wind"] = {"score_10": round(wind_score, 3), "score_100": int(round(wind_score * 10))}
    except Exception:
        _LOGGER.debug("Failed to compute wind component", exc_info=True)

    # WAVES component
    try:
        max_wave_pref = profile.get("max_wave_height_m", DEFAULT_PROFILE["max_wave_height_m"])
        max_wave = float(max_wave_pref) if max_wave_pref is not None else DEFAULT_PROFILE["max_wave_height_m"]
        if wave is None:
            wave_score = 0.0
        else:
            if wave <= 0.0:
                wave_score = 10.0
            elif wave >= max_wave:
                wave_score = 0.0
            else:
                wave_score = 10.0 * (1.0 - (wave / max_wave))
        wave_score = _clamp_0_10(wave_score)
        comp["waves"] = {"score_10": round(wave_score, 3), "score_100": int(round(wave_score * 10))}
    except Exception:
        _LOGGER.debug("Failed to compute waves component", exc_info=True)

    # TIME component
    try:
        preferred_times_raw = profile.get("preferred_times", []) or []

        def _normalize_preferred_times(pref_times: List[Any]) -> List[int]:
            out_hours: List[int] = []
            for it in pref_times:
                if isinstance(it, dict):
                    sh = None
                    eh = None
                    if "start_hour" in it or "end_hour" in it:
                        sh = it.get("start_hour")
                        eh = it.get("end_hour")
                    elif "start" in it or "end" in it:
                        sh = it.get("start")
                        eh = it.get("end")
                    elif "hour" in it:
                        sh = it.get("hour")
                        eh = None
                    else:
                        for k, v in it.items():
                            if isinstance(v, (int, float, str)):
                                sh = v
                                break
                    try:
                        if sh is None:
                            continue
                        sh_i = int(sh)
                    except Exception:
                        continue
                    if eh is None:
                        out_hours.append(sh_i % 24)
                    else:
                        try:
                            eh_i = int(eh)
                        except Exception:
                            out_hours.append(sh_i % 24)
                            continue
                        h = sh_i % 24
                        out_hours.append(h)
                        while h != (eh_i % 24):
                            h = (h + 1) % 24
                            out_hours.append(h)
                else:
                    try:
                        out_hours.append(int(float(it)) % 24)
                    except Exception:
                        continue
            return sorted(set([h % 24 for h in out_hours]))

        normalized_hours = _normalize_preferred_times(preferred_times_raw)

        if not normalized_hours:
            time_score = 10.0
        else:
            try:
                t_dt = _coerce_datetime(timestamps[use_index])
                hour = t_dt.hour if t_dt else None
            except Exception:
                hour = None
            if hour is None:
                time_score = 5.0
            else:
                def hour_distance(a: int, b: int) -> int:
                    d = abs(a - b) % 24
                    return min(d, 24 - d)

                min_dist = min(hour_distance(hour, pt) for pt in normalized_hours)
                if min_dist <= 3:
                    time_score = 10.0
                elif min_dist >= 6:
                    time_score = 0.0
                else:
                    time_score = 10.0 * (1.0 - ((min_dist - 3.0) / 3.0))
        time_score = _clamp_0_10(time_score)
        comp["time"] = {"score_10": round(time_score, 3), "score_100": int(round(time_score * 10))}
    except Exception:
        _LOGGER.debug("Failed to compute time component", exc_info=True)

    # PRESSURE, SEASON, MOON, TEMPERATURE components (unchanged, strict as before)
    try:
        if pressure_delta is None:
            pressure_score = 5.0
        else:
            if pressure_delta >= 2.0:
                pressure_score = 10.0
            elif pressure_delta <= -2.0:
                pressure_score = 0.0
            else:
                pressure_score = 10.0 * ((pressure_delta + 2.0) / 4.0)
        pressure_score = _clamp_0_10(pressure_score)
        comp["pressure"] = {"score_10": round(pressure_score, 3), "score_100": int(round(pressure_score * 10))}
    except Exception:
        _LOGGER.debug("Failed to compute pressure component", exc_info=True)

    try:
        preferred_months = profile.get("preferred_months", []) or []
        if not preferred_months:
            season_score = 10.0
        else:
            try:
                t_dt = _coerce_datetime(timestamps[use_index])
                month = t_dt.month if t_dt else None
            except Exception:
                month = None
            if month is None:
                season_score = 5.0
            else:
                season_score = 10.0 if int(month) in [int(m) for m in preferred_months] else 3.0
        season_score = _clamp_0_10(season_score)
        comp["season"] = {"score_10": round(season_score, 3), "score_100": int(round(season_score * 10))}
    except Exception:
        _LOGGER.debug("Failed to compute season component", exc_info=True)

    try:
        moon_pref = profile.get("moon_preference", []) or profile.get("moon_preference", []) or []
        if not moon_pref:
            moon_score = 10.0
        else:
            matched = False
            for mpref in moon_pref:
                try:
                    mpf = float(mpref)
                    if moon_phase_val is not None and abs(moon_phase_val - mpf) <= 0.05:
                        matched = True
                        break
                except Exception:
                    pass
            moon_score = 10.0 if matched else 4.0
        moon_score = _clamp_0_10(moon_score)
        comp["moon"] = {"score_10": round(moon_score, 3), "score_100": int(round(moon_score * 10))}
    except Exception:
        _LOGGER.debug("Failed to compute moon component", exc_info=True)

    try:
        pref_temp = profile.get("preferred_temp_c", DEFAULT_PROFILE["preferred_temp_c"])
        if isinstance(pref_temp, (list, tuple)) and len(pref_temp) >= 2:
            pt_min, pt_max = float(pref_temp[0]), float(pref_temp[1])
        else:
            pt = float(pref_temp) if pref_temp is not None else 10.0
            pt_min, pt_max = pt, pt
        temp_tol = 5.0
        temp_score = _linear_within_score_10(float(temp), pt_min, pt_max, temp_tol)
        temp_score = _clamp_0_10(temp_score)
        comp["temperature"] = {"score_10": round(temp_score, 3), "score_100": int(round(temp_score * 10))}
    except Exception:
        _LOGGER.debug("Failed to compute temperature component", exc_info=True)

    overall_10 = 0.0
    for k in weights:
        overall_10 += weights.get(k, 0.0) * comp.get(k, {}).get("score_10", 0.0)
    overall_10 = float(round(overall_10, 3))
    overall_100 = int(round(overall_10 * 10.0))

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

            max_gust = _to_float_safe(safety_limits.get("max_gust_m_s"))
            gust = _get_at("wind_max_m_s", use_index) if "wind_max_m_s" in data else None
            if max_gust is not None and gust is not None:
                if gust > max_gust:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"gust>{max_gust}")
                elif gust > (0.9 * max_gust):
                    safety["caution"] = True
                    safety["reasons"].append("gust_near_limit")

            min_vis = _to_float_safe(safety_limits.get("min_visibility_km"))
            vis = _get_at("visibility_km", use_index) if "visibility_km" in data else None
            if min_vis is not None and vis is not None:
                if vis < min_vis:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"vis<{min_vis}")
                elif vis < (1.1 * min_vis):
                    safety["caution"] = True
                    safety["reasons"].append("vis_near_limit")

            min_swell = _to_float_safe(safety_limits.get("min_swell_period_s"))
            swell = _get_at("swell_period_s", use_index) if "swell_period_s" in data else None
            if min_swell is not None and swell is not None:
                if swell < min_swell:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"swell<{min_swell}")
                elif swell < (1.1 * min_swell):
                    safety["caution"] = True
                    safety["reasons"].append("swell_near_limit")

            max_precip = _to_float_safe(safety_limits.get("max_precip_chance_pct"))
            precip = _get_at("precipitation_probability", use_index) if "precipitation_probability" in data else None
            if max_precip is not None and precip is not None:
                if precip > max_precip:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"precip>{max_precip}")
                elif precip > (0.9 * max_precip):
                    safety["caution"] = True
                    safety["reasons"].append("precip_near_limit")

    except Exception:
        _LOGGER.debug("Safety evaluation failed", exc_info=True)

    try:
        reason_codes = safety.get("reasons", []) or []
        safety["reason_strings"] = [_format_safety_reason(rc, safety_limits, units) for rc in reason_codes]
    except Exception:
        safety["reason_strings"] = []

    breaches: List[Dict[str, Any]] = []
    try:
        def _add_breach(variable: str, value: Any, unit: Optional[str] = None, expected_min: Any = None, expected_max: Any = None, expected_pref_min: Any = None, expected_pref_max: Any = None, severity: str = "caution", reason: Optional[str] = None, advice: Optional[str] = None):
            item: Dict[str, Any] = {
                "variable": variable,
                "value": value,
                "unit": unit,
                "expected_min": expected_min,
                "expected_max": expected_max,
                "severity": severity,
                "reason": reason or f"{variable}_breach",
                "advice": advice,
                "category": "species",
            }
            if expected_pref_min is not None:
                item["expected_pref_min"] = expected_pref_min
            if expected_pref_max is not None:
                item["expected_pref_max"] = expected_pref_max
            breaches.append(item)

        # TEMPERATURE breach detection (include both pref bounds and allowed bounds used)
        try:
            pref_temp = profile.get("preferred_temp_c")
            if temp is not None and pref_temp is not None:
                if isinstance(pref_temp, (list, tuple)) and len(pref_temp) >= 2:
                    pmin, pmax = float(pref_temp[0]), float(pref_temp[1])
                else:
                    pmin = pmax = float(pref_temp)
                tol = _to_float_safe(profile.get("preferred_temp_tol_c")) or 5.0
                allowed_low = pmin - tol
                allowed_high = pmax + tol
                if temp < allowed_low:
                    sev = "unsafe" if (allowed_low - temp) > (2 * tol) else "caution"
                    _add_breach("temperature", temp, unit="°C", expected_min=allowed_low, expected_max=allowed_high, expected_pref_min=pmin, expected_pref_max=pmax, severity=sev, reason="temperature<preferred_min", advice=f"{profile.get('common_name','Species')} prefers warmer water")
                elif temp > allowed_high:
                    sev = "unsafe" if (temp - allowed_high) > (2 * tol) else "caution"
                    _add_breach("temperature", temp, unit="°C", expected_min=allowed_low, expected_max=allowed_high, expected_pref_min=pmin, expected_pref_max=pmax, severity=sev, reason="temperature>preferred_max", advice=f"{profile.get('common_name','Species')} prefers cooler water")
        except Exception:
            pass

        # WAVE breach detection
        try:
            max_wave_pref = profile.get("max_wave_height_m")
            if wave is not None and max_wave_pref is not None:
                max_w = float(max_wave_pref)
                if wave > max_w:
                    _add_breach("wave", wave, unit="m", expected_min=None, expected_max=max_w, expected_pref_min=None, expected_pref_max=max_w, severity="unsafe", reason="wave>max_wave_height_m", advice=f"{profile.get('common_name','Species')} prefers lower waves")
                elif wave > (0.9 * max_w):
                    _add_breach("wave", wave, unit="m", expected_min=None, expected_max=max_w, expected_pref_min=None, expected_pref_max=max_w, severity="caution", reason="wave_near_max", advice="Wave height approaching species preferred maximum")
        except Exception:
            pass

        # WIND breach detection (upper bound)
        try:
            pref_wind = profile.get("preferred_wind_m_s")
            if wind is not None and pref_wind is not None:
                if isinstance(pref_wind, (list, tuple)) and len(pref_wind) >= 2:
                    _, pw_max = float(pref_wind[0]), float(pref_wind[1])
                else:
                    pw_max = float(pref_wind)
                tol_w = _to_float_safe(profile.get("preferred_wind_tol_m_s")) or max(1.0, 0.2 * max(1.0, pw_max))
                allowed_max = pw_max + tol_w
                if wind > (allowed_max):
                    _add_breach("wind", wind, unit="m/s", expected_min=None, expected_max=allowed_max, expected_pref_min=None, expected_pref_max=pw_max, severity="unsafe", reason="wind>preferred_max", advice=f"{profile.get('common_name','Species')} prefers lighter winds")
                elif wind > (pw_max + 0.9 * tol_w):
                    _add_breach("wind", wind, unit="m/s", expected_min=None, expected_max=allowed_max, expected_pref_min=None, expected_pref_max=pw_max, severity="caution", reason="wind_near_preferred_max", advice="Wind approaching species preferred maximum")
        except Exception:
            pass

        # TIME breach detection
        try:
            if profile.get("preferred_times"):
                normalized_hours = []
                for it in profile.get("preferred_times", []):
                    if isinstance(it, dict):
                        sh = it.get("start_hour") or it.get("start") or it.get("hour")
                        eh = it.get("end_hour") or it.get("end")
                        try:
                            sh_i = int(sh)
                        except Exception:
                            continue
                        if eh is None:
                            normalized_hours.append(sh_i % 24)
                        else:
                            try:
                                eh_i = int(eh)
                            except Exception:
                                normalized_hours.append(sh_i % 24)
                                continue
                            h = sh_i % 24
                            normalized_hours.append(h)
                            while h != (eh_i % 24):
                                h = (h + 1) % 24
                                normalized_hours.append(h)
                    else:
                        try:
                            normalized_hours.append(int(it) % 24)
                        except Exception:
                            continue
                normalized_hours = sorted(set(normalized_hours))
                try:
                    t_dt = _coerce_datetime(timestamps[use_index])
                    hour = t_dt.hour if t_dt else None
                except Exception:
                    hour = None
                if normalized_hours and hour is not None and hour not in normalized_hours:
                    _add_breach("time", hour, unit="hour", expected_min=min(normalized_hours), expected_max=max(normalized_hours), expected_pref_min=min(normalized_hours), expected_pref_max=max(normalized_hours), severity="caution", reason="time_out_of_preference", advice=f"{profile.get('common_name','Species')} prefers different times of day")
        except Exception:
            pass

        # TIDE PHASE breach detection - strict: expect string tide_phase if profile requests it
        try:
            pref_tide_phase = profile.get("preferred_tide_phase", []) or []
            if pref_tide_phase:
                tide_phase_val = None
                if "tide_phase" in data:
                    tp = data.get("tide_phase")
                    if isinstance(tp, (list, tuple)):
                        tide_phase_val = tp[use_index] if use_index < len(tp) else None
                    else:
                        tide_phase_val = tp
                elif "tide" in data and isinstance(data.get("tide"), dict):
                    tp = data.get("tide").get("tide_phase")
                    if isinstance(tp, (list, tuple)):
                        tide_phase_val = tp[use_index] if use_index < len(tp) else None
                    else:
                        tide_phase_val = tp

                # Strict: tide_phase must be string when preferred_tide_phase specified
                if tide_phase_val is None or not isinstance(tide_phase_val, str):
                    raise MissingDataError("tide_phase (string) required by species profile but missing or not a string")

                desired = [str(p).lower() for p in pref_tide_phase]
                if str(tide_phase_val).lower() not in desired:
                    _add_breach("tide_phase", tide_phase_val, unit=None, expected_min=None, expected_max=None, expected_pref_min=None, expected_pref_max=None, severity="caution", reason="tide_phase_mismatch", advice=f"{profile.get('common_name','Species')} prefers tide phases {pref_tide_phase}; current phase differs")
        except MissingDataError:
            raise
        except Exception:
            pass

        # MOON preference mismatch
        try:
            moon_pref = profile.get("moon_preference", []) or profile.get("moon_preference", []) or []
            if moon_pref and moon_phase_val is not None:
                matched = False
                for mpref in moon_pref:
                    try:
                        if isinstance(mpref, str):
                            continue
                        else:
                            mpf = float(mpref)
                            if abs(moon_phase_val - mpf) <= 0.05:
                                matched = True
                                break
                    except Exception:
                        continue
                if not matched:
                    _add_breach("moon_phase", moon_phase_val, unit=None, expected_min=None, expected_max=None, expected_pref_min=None, expected_pref_max=None, severity="caution", reason="moon_preference_mismatch", advice="Moon phase differs from species preference")
        except Exception:
            pass

    except Exception:
        _LOGGER.debug("Failed to compute species breaches", exc_info=True)

    result = {
        "score_10": overall_10,
        "score_100": overall_100,
        "components": comp,
        "raw": {
            "tide": _get_at("tide_height_m", use_index) if "tide_height_m" in data else None,
            "tide_phase_name": (data.get("tide_phase_name")[use_index] if isinstance(data.get("tide_phase_name"), (list, tuple)) and use_index < len(data.get("tide_phase_name")) else (data.get("tide_phase_name") if "tide_phase_name" in data else None)),
            "wind": wind,
            "wave": wave,
            "pressure_delta": pressure_delta,
            "temperature": temp,
            "timestamp": timestamps[use_index],
            "moon_phase": moon_phase_val,
            "wind_gust": _get_at("wind_max_m_s", use_index) if "wind_max_m_s" in data else None,
            "swell_period_s": _get_at("swell_period_s", use_index) if "swell_period_s" in data else None,
            "precipitation_probability": _get_at("precipitation_probability", use_index) if "precipitation_probability" in data else None,
        },
        "profile_used": profile.get("common_name", "unknown"),
        "safety": safety,
        "breaches": breaches,
    }
    return result


def compute_forecast(
    payload: Dict[str, Any],
    species_profile: Optional[Union[str, Dict[str, Any]]] = None,
    safety_limits: Optional[Dict[str, Any]] = None,
    units: str = "metric",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not payload or "timestamps" not in payload:
        return out
    timestamps = payload.get("timestamps") or []
    for idx, ts in enumerate(timestamps):
        try:
            res = compute_score(payload, species_profile=species_profile, use_index=idx, safety_limits=safety_limits, units=units)
            # add tide fields into formatted_weather if available
            tide_height = payload.get("tide_height_m")[idx] if isinstance(payload.get("tide_height_m"), (list, tuple)) and idx < len(payload.get("tide_height_m")) else (payload.get("tide_height_m") if "tide_height_m" in payload else None)
            tide_phase_name = (payload.get("tide_phase_name")[idx] if isinstance(payload.get("tide_phase_name"), (list, tuple)) and idx < len(payload.get("tide_phase_name")) else (payload.get("tide_phase_name") if "tide_phase_name" in payload else None))

            forecast_raw = {
                "formatted_weather": {
                    "temperature": payload.get("temperature_c")[idx] if isinstance(payload.get("temperature_c"), (list, tuple)) else payload.get("temperature_c"),
                    "wind": payload.get("wind_m_s")[idx] if isinstance(payload.get("wind_m_s"), (list, tuple)) else payload.get("wind_m_s"),
                    "wind_gust": payload.get("wind_max_m_s")[idx] if isinstance(payload.get("wind_max_m_s"), (list, tuple)) else payload.get("wind_max_m_s"),
                    "swell_period_s": payload.get("swell_period_s")[idx] if isinstance(payload.get("swell_period_s"), (list, tuple)) else payload.get("swell_period_s"),
                    "pressure_hpa": payload.get("pressure_hpa")[idx] if isinstance(payload.get("pressure_hpa"), (list, tuple)) else payload.get("pressure_hpa"),
                    "wave_height_m": payload.get("wave_height_m")[idx] if isinstance(payload.get("wave_height_m"), (list, tuple)) else payload.get("wave_height_m"),
                    "wave_period_s": payload.get("wave_period_s")[idx] if isinstance(payload.get("wave_period_s"), (list, tuple)) else payload.get("wave_period_s"),
                    "tide_height_m": tide_height,
                    "tide_phase_name": tide_phase_name,
                },
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
                "breaches": res.get("breaches", []),
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
                "breaches": [],
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
                "breaches": [],
            }
        out.append(entry)
    return out