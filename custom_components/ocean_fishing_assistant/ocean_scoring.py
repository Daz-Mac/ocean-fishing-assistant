"""
Document-accurate Ocean Fishing Scoring (index-aware) — strict mode.

This version enforces presence of all required inputs for an accurate score.
Missing required inputs will be logged (ERROR) and result in a MissingDataError
for that timestamp. compute_forecast will record the error on the per-timestamp
entry (score fields become None) so callers can see which timestamps were
incomplete.
"""
from __future__ import annotations
import json
import math
from typing import Any, Dict, Optional, Iterable, List, Union, Tuple
import pkgutil
import logging
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
    pass


def _to_float_safe(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _linear_within_score_10(value: float, pref_min: float, pref_max: float, tolerance: float) -> float:
    """
    Return 0..10 score where 10 inside [pref_min, pref_max], linearly decreases to 0
    at pref_min - tolerance and pref_max + tolerance.
    """
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


def resolve_species_profile(name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Return (profile_dict, profile_key) for a given identifier.

    Accepts either the profile key (e.g. "sea_bass") or a common name (e.g. "Sea Bass").
    Matching is case-insensitive for common names. Returns (None, None) if not found.
    """
    try:
        data_bytes = pkgutil.get_data(__package__, "species_profiles.json")
        if not data_bytes:
            return None, None
        profiles = json.loads(data_bytes.decode("utf-8"))
        # direct key match
        if name in profiles:
            return profiles.get(name), name
        # case-insensitive common_name match
        lname = str(name).strip().lower()
        for k, v in profiles.items():
            try:
                cn = str(v.get("common_name", "")).strip().lower()
                if cn and cn == lname:
                    return v, k
            except Exception:
                continue
        # case-insensitive key match
        for k in profiles.keys():
            if k.lower() == lname:
                return profiles.get(k), k
        return None, None
    except Exception:
        _LOGGER.debug("Unable to load species profiles", exc_info=True)
        return None, None


# backward compatible alias
def _load_species_profile_by_name(name: str) -> Optional[Dict[str, Any]]:
    prof, _ = resolve_species_profile(name)
    return prof


# Helper: coerce ISO-like timestamp to timezone-aware UTC datetime, or None
def _coerce_datetime(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        # handle common ISO formats; fromisoformat tolerates offsets
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        parsed = datetime.fromisoformat(s)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        # fallback numeric epoch (seconds)
        try:
            if isinstance(v, (int, float)):
                val = float(v)
                if val > 1e12:
                    val = val / 1000.0
                return datetime.fromtimestamp(val, tz=timezone.utc)
        except Exception:
            pass
    return None


# Helper: find nearest index in a list of timestamps for a target datetime
def _find_nearest_timestamp_index(timestamps: List[str], target_dt: datetime) -> Optional[int]:
    if not timestamps or target_dt is None:
        return None
    best_idx = None
    best_diff = None
    for i, ts in enumerate(timestamps):
        try:
            dt = _coerce_datetime(ts)
            if not dt:
                continue
            diff = abs((dt - target_dt).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = i
        except Exception:
            continue
    return best_idx


# Helper: try to extract a moon_phase for a given index from payload astro structures
def _get_moon_phase_for_index(payload: Dict[str, Any], use_index: int) -> Optional[float]:
    # 1) direct array keyed moon_phase (per-timestamp)
    moon_arr = payload.get("moon_phase")
    if isinstance(moon_arr, (list, tuple)):
        try:
            return _to_float_safe(moon_arr[use_index])
        except Exception:
            pass

    # 2) payload may include 'astro' or 'astronomy' which can be dict keyed by date or list
    for key in ("astro", "astronomy", "astronomy_forecast", "astro_forecast"):
        astro = payload.get(key)
        if astro is None:
            continue
        # if dict keyed by ISO dates, try to match by timestamp date
        try:
            ts = payload.get("timestamps", [])[use_index]
            dt = _coerce_datetime(ts)
            if dt is None:
                continue
            target_date = dt.date().isoformat()
        except Exception:
            target_date = None

        if isinstance(astro, dict):
            # try direct keys by date or fallback keys
            if target_date and target_date in astro:
                try:
                    return _to_float_safe(astro[target_date].get("moon_phase") or astro[target_date].get("moon"))
                except Exception:
                    pass
            # also try simple moon_phase key at top
            try:
                return _to_float_safe(astro.get("moon_phase") or astro.get("moon"))
            except Exception:
                pass
        elif isinstance(astro, (list, tuple)):
            # list of dicts with 'date' or 'iso' or per-entry moon info: find nearest by date
            try:
                # compute target datetime
                ts = payload.get("timestamps", [])[use_index]
                dt = _coerce_datetime(ts)
                if dt is None:
                    continue
                # search list for matching date string or nearest
                for entry in astro:
                    if not isinstance(entry, dict):
                        continue
                    # try 'date' / 'iso_date' / 'day'
                    cand = entry.get("date") or entry.get("iso_date") or entry.get("day")
                    if cand and str(cand).startswith(dt.date().isoformat()):
                        return _to_float_safe(entry.get("moon_phase") or entry.get("moon"))
                # fallback: if there's a 'moon_phase' that’s scalar, return it
                if len(astro) == 1 and isinstance(astro[0], dict):
                    return _to_float_safe(astro[0].get("moon_phase") or astro[0].get("moon"))
            except Exception:
                pass
    # 3) fall back to tide_phase if that was supplied per-timestamp or scalar (still treat as present)
    tide_phase_arr = payload.get("tide_phase")
    if isinstance(tide_phase_arr, (list, tuple)):
        try:
            return _to_float_safe(tide_phase_arr[use_index])
        except Exception:
            pass
    if tide_phase_arr is not None and not isinstance(tide_phase_arr, (list, tuple)):
        try:
            return _to_float_safe(tide_phase_arr)
        except Exception:
            pass
    return None


def _format_safety_reason(code: str, safety_limits: Optional[Dict[str, Any]]) -> str:
    """
    Convert a machine-coded safety reason into a human-readable string.
    Uses safety_limits when available to include numeric thresholds.
    """
    if not code:
        return ""

    code = str(code)
    # Numeric comparisons encoded as "wind>5.0", "wave>2.5", "vis<1.0", "swell>12"
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

    # named tokens
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

    # fallback to returning the raw code
    return code


def compute_score(
    data: Dict[str, Any],
    species_profile: Optional[Union[str, Dict[str, Any]]] = None,
    use_index: int = 0,
    safety_limits: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute scores with exact factor weights for a given index (timestamp).
    This strict variant requires presence of all essential inputs and will raise
    MissingDataError (after logging) when required inputs are absent.

    Returns a dict with score_10, score_100, components, raw, profile_used, safety.
    """
    if not data or "timestamps" not in data:
        raise MissingDataError("Missing timestamps in data")

    timestamps = data.get("timestamps", [])
    if use_index < 0 or use_index >= len(timestamps):
        raise MissingDataError(f"use_index {use_index} out of range for timestamps length {len(timestamps)}")

    # resolve profile
    if isinstance(species_profile, str):
        loaded = _load_species_profile_by_name(species_profile)
        profile = loaded if loaded is not None else DEFAULT_PROFILE
    elif isinstance(species_profile, dict):
        profile = species_profile
    else:
        profile = DEFAULT_PROFILE

    weights = FACTOR_WEIGHTS.copy()
    # allow profile override (but default weights should match doc)
    for k, v in (profile.get("weights") or {}).items():
        if k in weights:
            try:
                weights[k] = float(v)
            except Exception:
                pass
    # normalize to sum 1.0
    total = sum(weights.values()) or 1.0
    weights = {k: float(v) / total for k, v in weights.items()}

    # helper to extract value at index (safely) from arrays or scalars
    def _get_at(key: str, index: int = 0) -> Optional[float]:
        arr = data.get(key)
        if arr is None:
            return None
        # If it's a scalar, try to coerce
        if not isinstance(arr, (list, tuple)):
            return _to_float_safe(arr)
        try:
            return _to_float_safe(arr[index])
        except Exception:
            return None

    # Attempt to locate matching tide snapshot (tide structure may be separate)
    tide_snapshot = None
    try:
        raw_tide = data.get("tide")
        if isinstance(raw_tide, dict) and raw_tide.get("timestamps"):
            ts = timestamps[use_index]
            dt_target = _coerce_datetime(ts)
            idx = _find_nearest_timestamp_index(raw_tide.get("timestamps", []), dt_target) if dt_target else None
            if idx is not None:
                tide_snapshot = {
                    "tide_height_m": raw_tide.get("tide_height_m")[idx] if raw_tide.get("tide_height_m") else None,
                    "tide_phase": (raw_tide.get("tide_phase")[idx] if isinstance(raw_tide.get("tide_phase"), (list, tuple)) else raw_tide.get("tide_phase")),
                    "tide_strength": (raw_tide.get("tide_strength")[idx] if isinstance(raw_tide.get("tide_strength"), (list, tuple)) else raw_tide.get("tide_strength")),
                }
    except Exception:
        tide_snapshot = None

    # Extract values (index-aware, falling back to matching snapshot above)
    tide = _get_at("tide_height_m", use_index)
    if tide is None and tide_snapshot is not None:
        tide = _to_float_safe(tide_snapshot.get("tide_height_m"))

    wind = _get_at("wind_m_s", use_index) or _get_at("wind_max_m_s", use_index) or _get_at("windspeed_10m", use_index)
    # also try tolerant keys (we still search several keys to find the real field)
    if wind is None:
        wind = _get_at("wind") or _get_at("wind_speed")

    # Accept multiple possible canonical keys for wave/swell height (ensure meters)
    wave = (
        _get_at("wave_height_m", use_index)
        or _get_at("swell_height_m", use_index)
        or _get_at("swell_wave_height_m", use_index)
        or _get_at("wave_height", use_index)
    )

    # Pressure: compute delta next - current if array present
    pressure_arr = data.get("pressure_hpa")
    pressure = None
    try:
        if isinstance(pressure_arr, (list, tuple)):
            if len(pressure_arr) > use_index + 1:
                p_next = _to_float_safe(pressure_arr[use_index + 1])
                p_curr = _to_float_safe(pressure_arr[use_index])
                if p_next is not None and p_curr is not None:
                    pressure = float(p_next) - float(p_curr)
                else:
                    pressure = None
            else:
                pressure = None
        else:
            pressure = None
    except Exception:
        pressure = None

    # temperature: prefer direct hourly temperature, else average daily max/min (index-aware)
    temp = _get_at("temperature_c", use_index)
    if temp is None:
        tmax = _get_at("temperature_max_c", use_index)
        tmin = _get_at("temperature_min_c", use_index)
        if tmax is not None and tmin is not None:
            temp = (tmax + tmin) / 2.0

    # derive hour/month from timestamp for time/season scoring
    ts_str = timestamps[use_index]
    hour = None
    month = None
    try:
        dt = _coerce_datetime(ts_str)
        if dt:
            hour = dt.hour
            month = dt.month
    except Exception:
        hour = None
        month = None

    # get moon phase (prefer astro-derived)
    moon_phase = _get_moon_phase_for_index(data, use_index)

    # STRICT: require presence of essential inputs
    missing = []
    if tide is None:
        missing.append("tide_height_m")
    if wind is None:
        missing.append("wind")
    if wave is None:
        missing.append("wave_height")
    if moon_phase is None:
        missing.append("moon_phase/astro")
    if temp is None:
        missing.append("temperature_c")
    # pressure series presence check (we require enough points to compute a delta)
    if not (isinstance(pressure_arr, (list, tuple)) and len(pressure_arr) > use_index + 1 and _to_float_safe(pressure_arr[use_index]) is not None and _to_float_safe(pressure_arr[use_index + 1]) is not None):
        missing.append("pressure_hpa_series_with_future_point")

    if missing:
        msg = f"Missing required inputs for scoring at index={use_index} timestamp={ts_str}: {', '.join(missing)}"
        _LOGGER.error(msg)
        raise MissingDataError(msg)

    # compute component scores (0..10) - now that required values are present
    comp: Dict[str, Any] = {}

    # TIDE (0..10)
    try:
        pref_tide = profile.get("preferred_tide_m", DEFAULT_PROFILE["preferred_tide_m"])
        tmin, tmax = float(pref_tide[0]), float(pref_tide[1])
        tide_score = _linear_within_score_10(float(tide), tmin, tmax, tolerance=0.5)
        tide_reason = f"value={tide}m"
    except Exception:
        tide_score = 0.0
        tide_reason = "error"
    comp["tide"] = {"score_10": round(_clamp_0_10(tide_score), 3), "score_100": int(round(_clamp_0_10(tide_score) * 10)), "reason": tide_reason}

    # WIND (0..10)
    try:
        pref_w = profile.get("preferred_wind_m_s", DEFAULT_PROFILE["preferred_wind_m_s"])
        wmin, wmax = float(pref_w[0]), float(pref_w[1])
        wind_score = _linear_within_score_10(float(wind), wmin, wmax, tolerance=4.0)
        wind_reason = f"value={wind} m/s"
    except Exception:
        wind_score = 0.0
        wind_reason = "error"
    comp["wind"] = {"score_10": round(_clamp_0_10(wind_score), 3), "score_100": int(round(_clamp_0_10(wind_score) * 10)), "reason": wind_reason}

    # WAVES (0..10)
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

    # TIME (0..10)
    try:
        pref_times = profile.get("preferred_times", [])
        if not pref_times:
            time_score = 5.0
            time_reason = "no_preference"
        else:
            matched = False
            if hour is None:
                time_score = 5.0
                time_reason = "missing_time"
            else:
                for w in pref_times:
                    try:
                        sh = int(w.get("start_hour"))
                        eh = int(w.get("end_hour"))
                        if sh <= hour <= eh:
                            matched = True
                            break
                    except Exception:
                        continue
                time_score = 10.0 if matched else 2.0
                time_reason = "matched" if matched else "not_matched"
    except Exception:
        time_score = 5.0
        time_reason = "error"
    comp["time"] = {"score_10": round(_clamp_0_10(time_score), 3), "score_100": int(round(_clamp_0_10(time_score) * 10)), "reason": time_reason}

    # PRESSURE (0..10) - use pressure delta (hPa)
    try:
        p = float(pressure)
        pressure_score = 5.0 + (p / 10.0) * 5.0
        pressure_score = _clamp_0_10(pressure_score)
        pressure_reason = f"delta={p}"
    except Exception:
        pressure_score = 0.0
        pressure_reason = "error"
    comp["pressure"] = {"score_10": round(_clamp_0_10(pressure_score), 3), "score_100": int(round(_clamp_0_10(pressure_score) * 10)), "reason": pressure_reason}

    # SEASON (0..10)
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

    # MOON (0..10) - requires moon_phase present (we enforced above)
    try:
        p = float(moon_phase)
        # distance to new moon (0) and full (~0.5). smaller distance => better
        dist_new = abs(p - 0.0)
        dist_full = abs(p - 0.5)
        dist = min(dist_new, dist_full)
        moon_score = max(0.0, 10.0 * (1.0 - (dist / 0.25)))  # 0..10
        moon_reason = f"phase={p}"
    except Exception:
        moon_score = 0.0
        moon_reason = "error"
    comp["moon"] = {"score_10": round(_clamp_0_10(moon_score), 3), "score_100": int(round(_clamp_0_10(moon_score) * 10)), "reason": moon_reason}

    # TEMPERATURE (0..10)
    try:
        pref_temp = profile.get("preferred_temp_c", DEFAULT_PROFILE["preferred_temp_c"])
        tmin, tmax = float(pref_temp[0]), float(pref_temp[1])
        temp_score = _linear_within_score_10(float(temp), tmin, tmax, tolerance=10.0)
        temp_reason = f"value={temp}C"
    except Exception:
        temp_score = 0.0
        temp_reason = "error"
    comp["temperature"] = {"score_10": round(_clamp_0_10(temp_score), 3), "score_100": int(round(_clamp_0_10(temp_score) * 10)), "reason": temp_reason}

    # Weighted aggregation (on 0..10)
    overall_10 = 0.0
    for k in weights:
        overall_10 += weights.get(k, 0.0) * comp.get(k, {}).get("score_10", 0.0)
    overall_10 = float(round(overall_10, 3))
    overall_100 = int(round(overall_10 * 10.0))

    # Safety evaluation (if safety_limits provided) -> produce structured info
    safety = {"unsafe": False, "caution": False, "reasons": []}
    try:
        if safety_limits:
            # Canonical keys expected:
            # max_wind_m_s, max_wave_height_m, min_visibility_km, max_swell_period_s
            # WIND (max)
            max_wind = _to_float_safe(safety_limits.get("max_wind_m_s"))
            if max_wind is not None and wind is not None:
                if wind > max_wind:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"wind>{max_wind}")
                elif wind > (0.9 * max_wind):
                    safety["caution"] = True
                    safety["reasons"].append("wind_near_limit")

            # WAVE (max)
            max_wave = _to_float_safe(safety_limits.get("max_wave_height_m"))
            if max_wave is not None and wave is not None:
                if wave > max_wave:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"wave>{max_wave}")
                elif wave > (0.9 * max_wave):
                    safety["caution"] = True
                    safety["reasons"].append("wave_near_limit")

            # VISIBILITY (min)
            min_vis = _to_float_safe(safety_limits.get("min_visibility_km"))
            vis = _get_at("visibility_km", use_index) or _get_at("visibility", use_index)
            if min_vis is not None and vis is not None:
                if vis < min_vis:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"vis<{min_vis}")
                elif vis < (1.1 * min_vis):
                    # Slightly above minimum threshold -> caution
                    safety["caution"] = True
                    safety["reasons"].append("vis_near_limit")

            # SWELL PERIOD (max)
            max_swell = _to_float_safe(safety_limits.get("max_swell_period_s"))
            swell = _get_at("swell_period_s", use_index) or _get_at("swell_period", use_index)
            if max_swell is not None and swell is not None:
                if swell > max_swell:
                    safety["unsafe"] = True
                    safety["reasons"].append(f"swell>{max_swell}")
                elif swell > (0.9 * max_swell):
                    safety["caution"] = True
                    safety["reasons"].append("swell_near_limit")
    except Exception:
        # Do not fail scoring for safety evaluation errors; log and continue
        _LOGGER.debug("Safety evaluation failed", exc_info=True)

    # Add human-readable reason strings derived from the codes using the safety_limits for context
    try:
        reason_codes = safety.get("reasons", []) or []
        reason_strings = [_format_safety_reason(rc, safety_limits) for rc in reason_codes]
        safety["reason_strings"] = reason_strings
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
            "pressure_delta": pressure,
            "temperature": temp,
            "timestamp": ts_str,
            "moon_phase": moon_phase,
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
    Produce a per-timestamp formatted forecast list.
    Each entry includes:
      - timestamp (ISO string from payload 'timestamps')
      - index
      - score_10 (0..10) or None
      - score_100 (0..100) or None
      - components (dict) or None
      - forecast_raw: raw inputs used in scoring + matched astro/tide/marine snippets for debug
    """
    out: List[Dict[str, Any]] = []
    if not payload or "timestamps" not in payload:
        return out
    timestamps = payload.get("timestamps") or []
    # If the payload contains dedicated tide/marine/astro forecasts, keep them handy
    tide_forecast = payload.get("tide")  # may be None or dict
    marine_forecast = payload.get("marine") or payload.get("marine_forecast") or payload.get("marine_current")
    astro_data = payload.get("astro") or payload.get("astronomy") or payload.get("astronomy_forecast") or payload.get("astro_forecast")

    for idx, ts in enumerate(timestamps):
        try:
            # compute main score (compute_score will enforce presence of required inputs)
            res = compute_score(payload, species_profile=species_profile, use_index=idx, safety_limits=safety_limits)
            # build forecast_raw: include raw inputs and, where possible, matched astro/tide/marine entries
            dt_obj = _coerce_datetime(ts)
            # find matched tide snapshot if tide_forecast exists and is dict-with-timestamps
            matched_tide = None
            try:
                if isinstance(tide_forecast, dict) and tide_forecast.get("timestamps"):
                    j = _find_nearest_timestamp_index(tide_forecast.get("timestamps", []), dt_obj) if dt_obj else None
                    if j is not None:
                        matched_tide = {k: (tide_forecast.get(k)[j] if isinstance(tide_forecast.get(k), (list, tuple)) else tide_forecast.get(k)) for k in ("tide_height_m", "tide_phase", "tide_strength")}
            except Exception:
                matched_tide = None
            # matched marine
            matched_marine = None
            try:
                if isinstance(marine_forecast, dict) and marine_forecast.get("timestamps"):
                    k = _find_nearest_timestamp_index(marine_forecast.get("timestamps", []), dt_obj) if dt_obj else None
                    if k is not None:
                        # pull common marine fields if present
                        keys = ("wave_height_m", "swell_period_s", "swell_height_m", "wind_speed", "visibility_km")
                        matched_marine = {kk: (marine_forecast.get(kk)[k] if isinstance(marine_forecast.get(kk), (list, tuple)) else marine_forecast.get(kk)) for kk in keys if kk in marine_forecast}
                # if marine_forecast is list of dicts, find nearest by timestamp
                if isinstance(marine_forecast, (list, tuple)):
                    # find nearest list item by matching its datetime-like field
                    best = None
                    best_diff = None
                    for item in marine_forecast:
                        if not isinstance(item, dict):
                            continue
                        cand_ts = item.get("datetime") or item.get("time") or item.get("timestamp")
                        cand_dt = _coerce_datetime(cand_ts)
                        if not cand_dt or dt_obj is None:
                            continue
                        diff = abs((cand_dt - dt_obj).total_seconds())
                        if best_diff is None or diff < best_diff:
                            best = item
                            best_diff = diff
                    if best:
                        matched_marine = best
            except Exception:
                matched_marine = None

            # matched astro (enhanced: include per-timestamp moon_phase array or scalar if present)
            matched_astro = None
            try:
                if isinstance(astro_data, dict):
                    # try keyed by date
                    if dt_obj:
                        dkey = dt_obj.date().isoformat()
                        if dkey in astro_data:
                            matched_astro = astro_data.get(dkey)
                    # fallback to top-level scalar keys
                    if matched_astro is None:
                        # include moon_phase if present
                        if "moon_phase" in astro_data:
                            matched_astro = {"moon_phase": astro_data.get("moon_phase")}
                elif isinstance(astro_data, (list, tuple)):
                    # find nearest entry by date or 'date' key
                    best = None
                    best_diff = None
                    for item in astro_data:
                        if not isinstance(item, dict):
                            continue
                        cand = item.get("date") or item.get("iso_date") or item.get("day") or item.get("datetime") or item.get("time")
                        cand_dt = _coerce_datetime(cand)
                        if not cand_dt or dt_obj is None:
                            continue
                        diff = abs((cand_dt - dt_obj).total_seconds())
                        if best_diff is None or diff < best_diff:
                            best = item
                            best_diff = diff
                    if best:
                        matched_astro = best
            except Exception:
                matched_astro = None

            # FALLBACK: if we didn't match an astro block, but payload includes a per-timestamp
            # 'moon_phase' array or scalar, expose that as astro_used so diagnostics show the
            # actual moon_phase value used by compute_score (compute_score itself still uses
            # _get_moon_phase_for_index — we only populate the debug field here).
            if matched_astro is None:
                try:
                    moon_arr = payload.get("moon_phase")
                    if isinstance(moon_arr, (list, tuple)) and idx < len(moon_arr):
                        matched_astro = {"moon_phase": _to_float_safe(moon_arr[idx])}
                    elif moon_arr is not None and not isinstance(moon_arr, (list, tuple)):
                        # scalar moon_phase
                        matched_astro = {"moon_phase": _to_float_safe(moon_arr)}
                except Exception:
                    matched_astro = None

            forecast_raw = {
                "raw_input": payload,
                "formatted_weather": {
                    # convenient per-forecast weather hints (tolerant)
                    "temperature": payload.get("temperature_c", [None] * len(timestamps))[idx] if isinstance(payload.get("temperature_c"), (list, tuple)) else payload.get("temperature_c"),
                    "wind": payload.get("wind_m_s", [None] * len(timestamps))[idx] if isinstance(payload.get("wind_m_s"), (list, tuple)) else payload.get("wind_m_s"),
                    "pressure_hpa": payload.get("pressure_hpa", [None] * len(timestamps))[idx] if isinstance(payload.get("pressure_hpa"), (list, tuple)) else payload.get("pressure_hpa"),
                },
                "astro_used": matched_astro,
                "tide_used": matched_tide,
                "marine_used": matched_marine,
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
            # Missing required inputs for this index — record the error so callers know why the score is absent.
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