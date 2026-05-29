# custom_components/ocean_fishing_assistant/ocean_scoring.py
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
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
from datetime import datetime, timezone

_LOGGER = logging.getLogger(__name__)

from zoneinfo import ZoneInfo

from . import unit_helpers
from .moon_utils import coerce_phase, matches_moon_preference
from .safety import SafetyValidator

# Default global factor weights (used when no per-entry weights supplied)
FACTOR_WEIGHTS = {
    "tide": 0.25,
    "wind": 0.10,
    "wind_direction": 0.05,
    "waves": 0.15,
    "time": 0.15,
    "pressure": 0.10,
    "season": 0.10,
    "moon": 0.05,
    "temperature": 0.05,
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



def _resolve_tide_phase_at(data: Dict[str, Any], index: int) -> Optional[Any]:
    """Resolve tide_phase from top-level or tide block for a given index.
    Returns the raw value or None if not present / out of range.
    """
    if not isinstance(data, dict):
        return None
    if "tide_phase" in data:
        tp = data.get("tide_phase")
        if isinstance(tp, (list, tuple)):
            return tp[index] if index < len(tp) else None
        return tp
    tide = data.get("tide")
    if isinstance(tide, dict):
        tp = tide.get("tide_phase")
        if isinstance(tp, (list, tuple)):
            return tp[index] if index < len(tp) else None
        return tp
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
        v = v.strip()
        is_str_threshold = False
        try:
            val = float(v)
        except Exception:
            val = v
            is_str_threshold = True
        if k in ("wind", "wind_m_s"):
            if is_str_threshold:
                return f"Wind exceeds safe limit ({val})"
            return f"Wind exceeds safe limit ({_format_wind_val(val)})"
        if k in ("wave", "wave_height"):
            return f"Wave height exceeds safe limit ({val} m)"
        if k in ("swell", "swell_period"):
            return f"Swell period below safe minimum ({val} s)"
        if k in ("gust", "wind_gust"):
            if is_str_threshold:
                return f"Gust exceeds safe limit ({val})"
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
        return "Wave height near configured minimum"
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


def _validate_and_normalize_factor_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Validate that weights include exactly the keys in FACTOR_WEIGHTS,
    all values are numeric >= 0 and sum > 0. Normalize them to sum to 1.0.

    If weights is None, returns normalized default FACTOR_WEIGHTS.
    Raises ValueError on invalid inputs.
    """
    if weights is None:
        # normalize defaults
        total = sum(FACTOR_WEIGHTS.values()) or 1.0
        return {k: float(v) / total for k, v in FACTOR_WEIGHTS.items()}

    if not isinstance(weights, dict):
        raise ValueError("factor_weights must be a dict mapping factor name -> numeric weight")

    expected_keys = set(FACTOR_WEIGHTS.keys())
    provided_keys = set(weights.keys())
    if provided_keys != expected_keys:
        raise ValueError(f"factor_weights must contain exactly keys: {sorted(expected_keys)}; provided: {sorted(provided_keys)}")

    # numeric and non-negative
    norm: Dict[str, float] = {}
    for k, v in weights.items():
        try:
            fv = float(v)
        except Exception:
            raise ValueError(f"factor_weights value for '{k}' is not numeric: {v!r}")
        if fv < 0.0:
            raise ValueError(f"factor_weights value for '{k}' must be >= 0")
        norm[k] = fv

    total = sum(norm.values())
    if total <= 0.0:
        raise ValueError("factor_weights sum must be > 0")

    # Normalize to sum to 1.0
    return {k: float(v) / float(total) for k, v in norm.items()}


def compute_score(
    data: Dict[str, Any],
    species_profile: Optional[Union[str, Dict[str, Any]]] = None,
    use_index: int = 0,
    safety_limits: Optional[Dict[str, Any]] = None,
    units: str = "metric",
    factor_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if not data or "timestamps" not in data:
        raise MissingDataError("Missing timestamps in data")
    timestamps = data.get("timestamps", [])
    if use_index < 0 or use_index >= len(timestamps):
        raise MissingDataError(f"use_index {use_index} out of range")

    # Enforce a resolved species profile dict (no fallback)
    if not isinstance(species_profile, dict):
        raise MissingDataError(f"species_profile must be a resolved dict (species metadata). Received: {species_profile!r}")
    profile = dict(species_profile)

    # Validate factor weights (either defaults or provided)
    try:
        weights_norm = _validate_and_normalize_factor_weights(factor_weights)
    except Exception as exc:
        # Fail fast — weight config invalid
        raise ValueError(f"Invalid factor_weights provided: {exc}")

    # Location timezone is required (strict)
    tz_str = data.get("location_tz")
    if not isinstance(tz_str, str) or not tz_str:
        raise MissingDataError("location_tz (IANA timezone string) is required in data for local time scoring (strict)")
    try:
        tzinfo_local = ZoneInfo(tz_str)
    except Exception as exc:
        raise MissingDataError(f"Invalid location_tz '{tz_str}': {exc}")

    # Helper to detect whether a profile preference is actually set (not None / not empty list)
    def _pref_is_set(x: Any) -> bool:
        return x is not None and not (isinstance(x, (list, tuple)) and len(x) == 0)

    # Coerce empty-list fields to None to avoid numeric conversion errors
    for key in ("preferred_wind_m_s", "preferred_temp_c", "max_wave_height_m", "preferred_tide_phase", "preferred_times", "moon_preference", "preferred_swell_period_s"):
        val = profile.get(key)
        if isinstance(val, (list, tuple)) and len(val) == 0:
            profile[key] = None

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

    # wave/wave-period values — use swell_period_s as canonical period
    swell_period = _get_at("swell_period_s", use_index)

    moon_phase_val = None
    if "moon_phase" in data:
        mp = data.get("moon_phase")
        if isinstance(mp, (list, tuple)):
            moon_phase_val = _to_float_safe(mp[use_index]) if use_index < len(mp) else None
        else:
            moon_phase_val = _to_float_safe(mp)
    # coerce to normalized 0..1 using helper (optional; matching uses coerce internally)
    moon_phase_val = coerce_phase(moon_phase_val) if moon_phase_val is not None else None

    # --- robust pressure delta calculation with backward-diff fallback ---
    pressure_delta = None
    pressure_arr_ok = False
    if isinstance(pressure_arr, (list, tuple)):
        # ensure we have a current value at use_index
        p_curr = _to_float_safe(pressure_arr[use_index]) if use_index < len(pressure_arr) else None
        if p_curr is not None:
            # prefer forward difference when next point exists
            if use_index + 1 < len(pressure_arr):
                p_next = _to_float_safe(pressure_arr[use_index + 1])
                if p_next is not None:
                    pressure_delta = float(p_next) - float(p_curr)
                    pressure_arr_ok = True
            # fallback to backward difference when forward not available
            if not pressure_arr_ok and use_index - 1 >= 0:
                p_prev = _to_float_safe(pressure_arr[use_index - 1])
                if p_prev is not None:
                    pressure_delta = float(p_curr) - float(p_prev)
                    pressure_arr_ok = True
    else:
        pressure_arr_ok = False

    # Build missing list but only require components if profile requires them (except pressure & moon per policy)
    missing = []
    # wind required if species provided preference for wind
    if wind is None and _pref_is_set(profile.get("preferred_wind_m_s")):
        missing.append("wind_m_s")
    # wave required if species provided a max wave preference
    if wave is None and _pref_is_set(profile.get("max_wave_height_m")):
        missing.append("wave_height_m")
    # temp required if species provided a pref temp
    if temp is None and _pref_is_set(profile.get("preferred_temp_c")):
        missing.append("temperature_c")
    # moon: keep strict requirement (integration provides moon_phase via tide provider)
    if moon_phase_val is None:
        missing.append("moon_phase")
    # require at least a neighbor pressure point (forward OR backward) -- we keep this strict
    if not pressure_arr_ok:
        missing.append("pressure_hpa_series_with_neighbor_point")
    # swell period required if profile specifies preferred_swell_period_s
    if _pref_is_set(profile.get("preferred_swell_period_s")) and swell_period is None:
        missing.append("swell_period_s")

    if missing:
        msg = f"Missing required inputs for scoring at index={use_index} timestamp={timestamps[use_index]}: {', '.join(missing)}"
        raise MissingDataError(msg)

    comp: Dict[str, Any] = {}

    # TIDE component — phase-based only (and tolerant "any" token)
    pref_tide_phase_raw = profile.get("preferred_tide_phase", []) or []
    # normalize and treat "any"/"none" as no preference
    pref_tide_phase = [str(p).strip().lower() for p in (pref_tide_phase_raw or []) if str(p).strip().lower() not in ("any", "none", "")]
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

    # Strict: if pref_tide_phase specified (non-empty after filtering), tide_phase MUST be present and be a string.
    if pref_tide_phase:
        if tide_phase_val is None or not isinstance(tide_phase_val, str):
            raise MissingDataError("tide_phase (string) required by species profile but missing or not a string")
        matched = any(str(pref).lower() == str(tide_phase_val).lower() for pref in pref_tide_phase)
        tide_score = 10.0 if matched else 3.0
    else:
        # No preference -> maximum score (do not penalize missing tide phase)
        tide_score = 10.0
    tide_score = _clamp_0_10(tide_score)
    comp_tide: Dict[str, Any] = {"score_10": round(tide_score, 3), "score_100": int(round(tide_score * 10))}

    if tide_phase_val is not None and isinstance(tide_phase_val, str):
        comp_tide["tide_phase"] = tide_phase_val

    comp["tide"] = comp_tide

    # WIND component
    pref_wind = profile.get("preferred_wind_m_s")
    if pref_wind is None:
        if wind is None:
            wind_score = 10.0
        else:
            wind_score = 10.0
    else:
        if isinstance(pref_wind, (list, tuple)) and len(pref_wind) >= 2:
            pw_min, pw_max = float(pref_wind[0]), float(pref_wind[1])
        else:
            pw = float(pref_wind) if pref_wind is not None else 0.0
            pw_min, pw_max = pw, pw
        wind_tol = max(1.0, 0.2 * max(1.0, pw_max))
        if wind is None:
            raise MissingDataError("wind_m_s required by profile but missing")
        wind_score = _linear_within_score_10(float(wind), pw_min, pw_max, wind_tol)
    wind_score = _clamp_0_10(wind_score)
    comp["wind"] = {"score_10": round(wind_score, 3), "score_100": int(round(wind_score * 10))}

    # WIND_DIRECTION component
    # Get preferred wind directions from profile (e.g., ["E", "SE", "NE"]) or user config
    pref_wind_dirs_raw = profile.get("preferred_wind_directions")
    # Handle comma-separated string or list
    pref_wind_dirs = None
    if pref_wind_dirs_raw:
        if isinstance(pref_wind_dirs_raw, str):
            pref_wind_dirs = [d.strip().upper() for d in pref_wind_dirs_raw.split(",") if d.strip()]
        elif isinstance(pref_wind_dirs_raw, list):
            pref_wind_dirs = [str(d).strip().upper() for d in pref_wind_dirs_raw if str(d).strip()]
        # Filter out "ANY" or empty
        pref_wind_dirs = [d for d in pref_wind_dirs if d and d != "ANY"]
        if not pref_wind_dirs:
            pref_wind_dirs = None

    # Get actual wind direction at this index
    wind_direction = _get_at("wind_direction", use_index)

    # Score wind direction
    if pref_wind_dirs is None or wind_direction is None:
        wind_dir_score = 10.0
    else:
        # Import wind direction constants
        try:
            from .const import WIND_DIRECTIONS, WIND_DIRECTION_TOLERANCE_DEGREES
        except Exception:
            WIND_DIRECTIONS = {}
            WIND_DIRECTION_TOLERANCE_DEGREES = 45

        # Convert preferred directions to degrees
        pref_degrees = []
        for d in pref_wind_dirs:
            if d in WIND_DIRECTIONS:
                pref_degrees.append(WIND_DIRECTIONS[d])

        if not pref_degrees:
            wind_dir_score = 10.0
        else:
            # Find minimum angular distance to any preferred direction
            min_distance = min(
                min(abs(wind_direction - pd), 360 - abs(wind_direction - pd))
                for pd in pref_degrees
            )

            # Score: 10 at 0°, 0 at 2*tolerance, linear in between
            tolerance = float(WIND_DIRECTION_TOLERANCE_DEGREES)
            if min_distance <= tolerance:
                wind_dir_score = 10.0
            elif min_distance >= 2 * tolerance:
                wind_dir_score = 0.0
            else:
                wind_dir_score = 10.0 * (1.0 - (min_distance - tolerance) / tolerance)

    wind_dir_score = _clamp_0_10(wind_dir_score)
    comp["wind_direction"] = {"score_10": round(wind_dir_score, 3), "score_100": int(round(wind_dir_score * 10))}
    if wind_direction is not None:
        comp["wind_direction"]["wind_direction_deg"] = round(wind_direction, 1)
    if pref_wind_dirs:
        comp["wind_direction"]["preferred_directions"] = pref_wind_dirs

    # WAVES component
    max_wave_pref = profile.get("max_wave_height_m")
    wave_score = None
    if max_wave_pref is None:
        if wave is None:
            wave_score = 10.0
        else:
            wave_score = 10.0
    else:
        max_wave = float(max_wave_pref)
        if wave is None:
            raise MissingDataError("wave_height_m required by profile but missing")
        if wave <= 0.0:
            wave_score = 10.0
        elif wave >= max_wave:
            wave_score = 0.0
        else:
            wave_score = 10.0 * (1.0 - (wave / max_wave))

    pref_swell_period = profile.get("preferred_swell_period_s")
    period_score = None
    if pref_swell_period and swell_period is not None:
        if isinstance(pref_swell_period, (list, tuple)) and len(pref_swell_period) >= 2:
            pp_min, pp_max = float(pref_swell_period[0]), float(pref_swell_period[1])
        else:
            pp = float(pref_swell_period)
            pp_min, pp_max = pp, pp
        period_score = _linear_within_score_10(float(swell_period), pp_min, pp_max, tolerance=2.0)

    if period_score is not None:
        final_wave_score = ((wave_score or 0.0) + (period_score or 0.0)) / 2.0
    else:
        final_wave_score = wave_score if wave_score is not None else 10.0

    final_wave_score = _clamp_0_10(final_wave_score)
    comp["waves"] = {"score_10": round(final_wave_score, 3), "score_100": int(round(final_wave_score * 10))}

    # TIME component (use local time via location_tz)
    preferred_times_raw = profile.get("preferred_times", []) or []

    def _normalize_preferred_times(pref_times: List[Any]) -> List[int]:
        out_hours: List[int] = []
        token_map = {
            # fallback token map for numeric tokens; dawn/dusk handled specially via period indices
            "day": list(range(7, 17)),  # 07-16
            "night": [h for h in range(0, 24) if h not in range(7, 17)],  # night
            "all_day": list(range(0, 24)),
        }
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
                if sh is None:
                    continue
                if isinstance(sh, str) and str(sh).strip().lower() in token_map:
                    token_hours = token_map[str(sh).strip().lower()]
                    out_hours.extend(token_hours)
                    continue
                try:
                    sh_i = int(sh)
                except Exception:
                    continue
                if eh is None:
                    out_hours.append(sh_i % 24)
                else:
                    if isinstance(eh, str) and str(eh).strip().lower() in token_map:
                        token_hours = token_map[str(eh).strip().lower()]
                        out_hours.extend(token_hours)
                        continue
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
                if isinstance(it, str):
                    key = it.strip().lower()
                    if key in token_map:
                        out_hours.extend(token_map[key])
                        continue
                try:
                    out_hours.append(int(float(it)) % 24)
                except Exception:
                    continue
        return sorted(set([h % 24 for h in out_hours]))

    normalized_hours = _normalize_preferred_times(preferred_times_raw)

    requested_special_tokens = set()
    for it in preferred_times_raw:
        if isinstance(it, str):
            key = it.strip().lower()
            if key in ("dawn", "dusk"):
                requested_special_tokens.add(key)
        elif isinstance(it, dict):
            for k in ("start", "start_hour", "hour"):
                if k in it and isinstance(it.get(k), str):
                    key = str(it.get(k)).strip().lower()
                    if key in ("dawn", "dusk"):
                        requested_special_tokens.add(key)

    # precomputed_pf may be provided in data (canonical) by DataFormatter using coordinator-supplied indices
    precomputed_pf = data.get("period_forecasts") if isinstance(data.get("period_forecasts"), dict) else {}
    time_score = 10.0
    if not normalized_hours and not requested_special_tokens:
        time_score = 10.0
    else:
        # convert current timestamp to UTC then to local tz
        t_dt_utc = _coerce_datetime(timestamps[use_index])
        local_dt = t_dt_utc.astimezone(ZoneInfo(tz_str)) if t_dt_utc else None
        hour = local_dt.hour if local_dt else None
        date_key = local_dt.date().isoformat() if local_dt else None

        used_precomputed_match = False
        if requested_special_tokens and precomputed_pf and date_key:
            pmap = precomputed_pf.get(date_key) or {}
            for tok in requested_special_tokens:
                pdata = pmap.get(tok)
                if pdata and isinstance(pdata, dict):
                    indices = pdata.get("indices") or []
                    if int(use_index) in [int(x) for x in indices]:
                        time_score = 10.0
                        used_precomputed_match = True
                        break

        if not used_precomputed_match:
            if hour is None:
                time_score = 5.0
            else:
                def hour_distance(a: int, b: int) -> int:
                    d = abs(a - b) % 24
                    return min(d, 24 - d)

                if not normalized_hours:
                    # if only requested special tokens but no matching precomputed_pf, fall back to neutral
                    time_score = 10.0 if not requested_special_tokens else 5.0
                else:
                    min_dist = min(hour_distance(hour, pt) for pt in normalized_hours)
                    if min_dist == 0:
                        time_score = 10.0
                    elif min_dist == 1:
                        time_score = 8.0
                    elif min_dist == 2:
                        time_score = 5.0
                    elif min_dist == 3:
                        time_score = 2.0
                    else:
                        time_score = 0.0

    time_score = _clamp_0_10(time_score)
    comp["time"] = {"score_10": round(time_score, 3), "score_100": int(round(time_score * 10))}

    # PRESSURE, SEASON, MOON, TEMPERATURE components (unchanged logic)...
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

    preferred_months = profile.get("preferred_months", []) or []
    if not preferred_months:
        season_score = 10.0
    else:
        t_dt_utc = _coerce_datetime(timestamps[use_index])
        local_dt = t_dt_utc.astimezone(ZoneInfo(tz_str)) if t_dt_utc else None
        month = local_dt.month if local_dt else None
        if month is None:
            season_score = 5.0
        else:
            season_score = 10.0 if int(month) in [int(m) for m in preferred_months] else 3.0
    season_score = _clamp_0_10(season_score)
    comp["season"] = {"score_10": round(season_score, 3), "score_100": int(round(season_score * 10))}

    moon_pref = profile.get("moon_preference", []) or []

    if not moon_pref:
        moon_score = 10.0
    else:
        if matches_moon_preference(moon_phase_val, moon_pref, tolerance=0.05):
            moon_score = 10.0
        else:
            moon_score = 4.0
    moon_score = _clamp_0_10(moon_score)
    comp["moon"] = {"score_10": round(moon_score, 3), "score_100": int(round(moon_score * 10))}

    pref_temp = profile.get("preferred_temp_c")
    if pref_temp is None:
        if temp is None:
            temp_score = 10.0
        else:
            temp_score = 10.0
    else:
        if isinstance(pref_temp, (list, tuple)) and len(pref_temp) >= 2:
            pt_min, pt_max = float(pref_temp[0]), float(pref_temp[1])
        else:
            pt = float(pref_temp) if pref_temp is not None else 10.0
            pt_min, pt_max = pt, pt
        temp_tol = _to_float_safe(profile.get("preferred_temp_tol_c")) or 5.0
        if temp is None:
            raise MissingDataError("temperature_c required by profile but missing")
        temp_score = _linear_within_score_10(float(temp), pt_min, pt_max, temp_tol)
    temp_score = _clamp_0_10(temp_score)
    comp["temperature"] = {"score_10": round(temp_score, 3), "score_100": int(round(temp_score * 10))}

    # Compute overall score using normalized weights_norm
    overall_10 = 0.0
    for k in weights_norm:
        comp_score = comp.get(k, {}).get("score_10")
        if comp_score is None:
            overall_10 += weights_norm.get(k, 0.0) * 10.0
        else:
            overall_10 += weights_norm.get(k, 0.0) * comp_score
    overall_10 = float(round(overall_10, 3))
    overall_100 = int(round(overall_10 * 10.0))

    gust = _get_at("wind_max_m_s", use_index) if "wind_max_m_s" in data else None
    vis = _get_at("visibility_km", use_index) if "visibility_km" in data else None
    precip = _get_at("precipitation_probability", use_index) if "precipitation_probability" in data else None

    validator = SafetyValidator(safety_limits)
    safety = validator.check(
        wind=wind, wave=wave, gust=gust,
        visibility=vis, swell_period=swell_period,
        precipitation=precip,
    )

    reason_codes = safety.get("reasons", []) or []
    safety["reason_strings"] = [_format_safety_reason(rc, safety_limits, units) for rc in reason_codes]

    # Adjust component scores based on safety limit breaches.
    # When a safety limit is breached (e.g., wind > max_wind_m_s), the corresponding
    # component score should reflect the safety concern, not just the species preference.
    # Map safety reason prefixes to component keys and their cap values.
    _SAFETY_COMPONENT_MAP = {
        "wind": ("wind", 0.0, 3.0),
        "wave": ("waves", 0.0, 3.0),
        "gust": ("wind", 0.0, 3.0),
        "swell_period": ("waves", 0.0, 3.0),
    }
    for rc in reason_codes:
        for prefix, (comp_key, unsafe_cap, caution_cap) in _SAFETY_COMPONENT_MAP.items():
            if rc.startswith(f"{prefix}>") or rc.startswith(f"{prefix}<"):
                if comp_key in comp and comp[comp_key].get("score_10") is not None:
                    comp[comp_key]["score_10"] = round(unsafe_cap, 3)
                    comp[comp_key]["score_100"] = int(round(unsafe_cap * 10))
                    comp[comp_key]["safety_capped"] = True
                break
            elif rc == f"{prefix}_near_limit":
                if comp_key in comp and comp[comp_key].get("score_10") is not None:
                    current = comp[comp_key]["score_10"]
                    if current > caution_cap:
                        comp[comp_key]["score_10"] = round(caution_cap, 3)
                        comp[comp_key]["score_100"] = int(round(caution_cap * 10))
                        comp[comp_key]["safety_capped"] = True
                break

    # Recompute overall score after safety-based component adjustments
    overall_10 = 0.0
    for k in weights_norm:
        comp_score = comp.get(k, {}).get("score_10")
        if comp_score is None:
            overall_10 += weights_norm.get(k, 0.0) * 10.0
        else:
            overall_10 += weights_norm.get(k, 0.0) * comp_score
    overall_10 = float(round(overall_10, 3))
    overall_100 = int(round(overall_10 * 10.0))

    breaches: List[Dict[str, Any]] = []
    def _add_breach(variable: str, value: Any, unit: Optional[str] = None, expected_min: Any = None, expected_max: Any = None, expected_pref_min: Any = None, expected_pref_max: Any = None, severity: str = "caution", reason: Optional[str] = None, advice: Optional[str] = None):
        item: Dict[str, Any] = {"variable": variable, "value": value, "severity": severity, "reason": reason or f"{variable}_breach", "category": "species"}
        if unit is not None:
            item["unit"] = unit
        if expected_min is not None:
            item["expected_min"] = expected_min
        if expected_max is not None:
            item["expected_max"] = expected_max
        if expected_pref_min is not None:
            item["expected_pref_min"] = expected_pref_min
        if expected_pref_max is not None:
            item["expected_pref_max"] = expected_pref_max
        breaches.append(item)

    # TEMPERATURE breach detection
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

    # WAVE breach detection
    max_wave_pref = profile.get("max_wave_height_m")
    if wave is not None and max_wave_pref is not None:
        max_w = float(max_wave_pref)
        if wave > max_w:
            _add_breach("wave", wave, unit="m", expected_min=None, expected_max=max_w, expected_pref_min=None, expected_pref_max=max_w, severity="unsafe", reason="wave>max_wave_height_m", advice=f"{profile.get('common_name','Species')} prefers lower waves")
        elif wave > (0.9 * max_w):
            _add_breach("wave", wave, unit="m", expected_min=None, expected_max=max_w, expected_pref_min=None, expected_pref_max=max_w, severity="caution", reason="wave_near_max", advice="Wave height approaching species preferred maximum")

    # WIND breach detection
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

    # TIME breach detection
    if profile.get("preferred_times"):
        t_dt_utc = _coerce_datetime(timestamps[use_index])
        local_dt = t_dt_utc.astimezone(ZoneInfo(tz_str)) if t_dt_utc else None
        hour = local_dt.hour if local_dt else None
        if 'normalized_hours' in locals() and normalized_hours and hour is not None:
            def hour_distance(a: int, b: int) -> int:
                d = abs(a - b) % 24
                return min(d, 24 - d)
            min_dist = min(hour_distance(hour, pt) for pt in normalized_hours)
            if min_dist > 3:
                _add_breach(
                    "time",
                    hour,
                    unit="hour",
                    expected_min=min(normalized_hours),
                    expected_max=max(normalized_hours),
                    expected_pref_min=min(normalized_hours),
                    expected_pref_max=max(normalized_hours),
                    severity="caution",
                    reason="time_out_of_preference",
                    advice=f"{profile.get('common_name','Species')} prefers different times of day",
                )

    # TIDE PHASE breach detection
    pref_tide_phase_check = profile.get("preferred_tide_phase", []) or []
    pref_tide_phase_check = [str(p).strip().lower() for p in pref_tide_phase_check if str(p).strip().lower() not in ("any", "none", "")]
    if pref_tide_phase_check:
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

        if tide_phase_val is None or not isinstance(tide_phase_val, str):
            raise MissingDataError("tide_phase (string) required by species profile but missing or not a string")

        desired = [str(p).lower() for p in pref_tide_phase_check]
        if str(tide_phase_val).lower() not in desired:
            _add_breach("tide_phase", tide_phase_val, unit=None, expected_min=None, expected_max=None, expected_pref_min=None, expected_pref_max=None, severity="caution", reason="tide_phase_mismatch", advice=f"{profile.get('common_name','Species')} prefers tide phases {pref_tide_phase_check}; current phase differs")

    # MOON preference mismatch
    moon_pref_check = profile.get("moon_preference", []) or []
    if moon_pref_check and moon_phase_val is not None:
        if not matches_moon_preference(moon_phase_val, moon_pref_check, tolerance=0.05):
            _add_breach("moon_phase", moon_phase_val, unit=None, expected_min=None, expected_max=None, expected_pref_min=None, expected_pref_max=None, severity="caution", reason="moon_preference_mismatch", advice="Moon phase differs from species preference")

    # deduplicate breaches (preserve order)
    unique_breaches: List[Dict[str, Any]] = []
    seen_keys = set()
    for b in breaches:
        key = (b.get("variable"), b.get("reason"), str(b.get("value")), b.get("severity"))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_breaches.append(b)
    breaches = unique_breaches

    # If safety limits mark this period as unsafe, cap the score to 30 (score_100) and reflect in score_10.
    # "caution" should not modify the score.
    if safety.get("unsafe"):
        try:
            overall_100 = min(int(overall_100), 30)
        except Exception:
            overall_100 = 30 if overall_100 else 30
        overall_10 = float(overall_100) / 10.0

    result = {
        "score_10": overall_10,
        "score_100": overall_100,
        "components": comp,
        "raw": {
            # Provide canonical tide_phase (machine-friendly) in raw if present
            "tide_phase": _resolve_tide_phase_at(data, use_index),
            "wind": wind,
            "wave": wave,
            "pressure_delta": pressure_delta,
            "temperature": temp,
            "timestamp": timestamps[use_index],
            "moon_phase": moon_phase_val,
            "wind_gust": _get_at("wind_max_m_s", use_index) if "wind_max_m_s" in data else None,
            "swell_period_s": swell_period,
            "precipitation_probability": _get_at("precipitation_probability", use_index) if "precipitation_probability" in data else None,
        },
        # Change: provide the resolved species profile as a dict so downstream display code can augment it.
        "profile_used": dict(profile),
        "safety": safety,
        "breaches": breaches,
    }
    return result


def compute_forecast(
    payload: Dict[str, Any],
    species_profile: Optional[Union[str, Dict[str, Any]]] = None,
    safety_limits: Optional[Dict[str, Any]] = None,
    units: str = "metric",
    factor_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not payload or "timestamps" not in payload:
        return out
    timestamps = payload.get("timestamps") or []
    for idx, ts in enumerate(timestamps):
        # Fail-fast: do not swallow exceptions. compute_score will raise on missing/invalid data.
        res = compute_score(payload, species_profile=species_profile, use_index=idx, safety_limits=safety_limits, units=units, factor_weights=factor_weights)

        tide_phase = _resolve_tide_phase_at(payload, idx)

        formatted_swell = payload.get("swell_period_s")[idx] if isinstance(payload.get("swell_period_s"), (list, tuple)) and idx < len(payload.get("swell_period_s")) else (payload.get("swell_period_s") if "swell_period_s" in payload else None)
        formatted_wave_period = payload.get("wave_period_s")[idx] if isinstance(payload.get("wave_period_s"), (list, tuple)) and idx < len(payload.get("wave_period_s")) else (payload.get("wave_period_s") if "wave_period_s" in payload else None)

        forecast_raw = {
            "formatted_weather": {
                "temperature": payload.get("temperature_c")[idx] if isinstance(payload.get("temperature_c"), (list, tuple)) else payload.get("temperature_c"),
                "wind": payload.get("wind_m_s")[idx] if isinstance(payload.get("wind_m_s"), (list, tuple)) else payload.get("wind_m_s"),
                "wind_gust": payload.get("wind_max_m_s")[idx] if isinstance(payload.get("wind_max_m_s"), (list, tuple)) else payload.get("wind_max_m_s"),
                "swell_period_s": formatted_swell,
                "pressure_hpa": payload.get("pressure_hpa")[idx] if isinstance(payload.get("pressure_hpa"), (list, tuple)) else payload.get("pressure_hpa"),
                "wave_height_m": payload.get("wave_height_m")[idx] if isinstance(payload.get("wave_height_m"), (list, tuple)) else payload.get("wave_height_m"),
                "wave_period_s": formatted_wave_period,
                # tide_height_m removed
                "tide_phase": tide_phase,
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
        out.append(entry)
    return out