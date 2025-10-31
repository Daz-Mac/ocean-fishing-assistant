"""
Document-accurate Ocean Fishing Scoring.

Implements factor names and weights per OceanFishingScorer._get_factor_weights:
- tide: 0.25
- wind: 0.15
- waves: 0.15
- time: 0.15
- pressure: 0.10
- season: 0.10
- moon: 0.05
- temperature: 0.03

All component scores are computed on 0..10 scale (score_10). The payload includes both
score_10 and score_100 (rounded int) for each component and for overall.
"""
from __future__ import annotations
import json
import math
from typing import Any, Dict, Optional, Iterable, List, Union
import pkgutil
import logging

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
    "preferred_times": [],  # list of {"start_hour":, "end_hour":} or empty
    "preferred_months": [],  # ints 1..12
    "moon_preference": [],  # e.g. ["full", "new"]
    "weights": FACTOR_WEIGHTS,
}

class MissingDataError(ValueError):
    pass

def _first_numeric(values: Optional[Iterable]) -> Optional[float]:
    if not values:
        return None
    for v in values:
        try:
            if v is None:
                continue
            return float(v)
        except Exception:
            continue
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

def _load_species_profile_by_name(name: str) -> Optional[Dict[str, Any]]:
    try:
        data_bytes = pkgutil.get_data(__name__, "species_profiles.json")
        if not data_bytes:
            return None
        profiles = json.loads(data_bytes.decode("utf-8"))
        return profiles.get(name)
    except Exception:
        _LOGGER.debug("Unable to load species profiles", exc_info=True)
        return None

def compute_score(
    data: Dict[str, Any],
    species_profile: Optional[Union[str, Dict[str, Any]]] = None,
    use_index: int = 0,
) -> Dict[str, Any]:
    """
    Compute scores with exact factor weights. Returns:
    {
      "score_10": float,
      "score_100": int,
      "components": {
         "tide": {"score_10":..., "score_100":..., ...},
         ...
      },
      "raw": {...},
      "profile_used": ...
    }
    """

    if not data or "timestamps" not in data:
        raise MissingDataError("Missing timestamps in data")

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

    # helper to extract value at index
    def _get_at(key: str) -> Optional[float]:
        arr = data.get(key)
        return _first_numeric(arr)

    # Extract values (prefer specific keys)
    tide = _get_at("tide_height_m")
    wind = _get_at("wind_m_s") or _get_at("wind_max_m_s") or _get_at("windspeed_10m")
    wave = _get_at("wave_height_m")
    pressure_arr = data.get("pressure_hpa")
    pressure = None
    if pressure_arr:
        try:
            # use difference 1->0 if available
            if len(pressure_arr) >= 2:
                pressure = float(pressure_arr[1]) - float(pressure_arr[0])
            else:
                pressure = 0.0
        except Exception:
            pressure = None
    temp = _get_at("temperature_c") or (_get_at("temperature_max_c") + _get_at("temperature_min_c") / 2.0 if _get_at("temperature_max_c") and _get_at("temperature_min_c") else None)

    # time & moon & season info
    # timestamp at index
    timestamps = data.get("timestamps", [])
    ts_str = timestamps[use_index] if len(timestamps) > use_index else None
    hour = None
    month = None
    if ts_str:
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).astimezone(timezone.utc)
            hour = dt.hour
            month = dt.month
        except Exception:
            hour = None
            month = None

    moon_phase = data.get("moon_phase") or data.get("tide_phase") or None  # tide_phase from tide proxy can be used
    # compute component scores (0..10)
    comp: Dict[str, Any] = {}

    # TIDE (0..10)
    try:
        pref_tide = profile.get("preferred_tide_m", DEFAULT_PROFILE["preferred_tide_m"])
        tmin, tmax = float(pref_tide[0]), float(pref_tide[1])
        if tide is None:
            tide_score = 5.0
            tide_reason = "missing"
        else:
            tide_score = _linear_within_score_10(float(tide), tmin, tmax, tolerance=0.5)
            tide_reason = f"value={tide}"
            # phase bonus
            pref_phases = profile.get("preferred_tide_phase", [])
            if pref_phases and moon_phase is not None:
                # simplistic: if moon_phase near spring and pref_phases contains 'spring' or 'full' give +1
                # allow textual matches too
                # here we simply not implement phase textual matching in depth; it's optional
                pass
    except Exception:
        tide_score = 5.0
        tide_reason = "error"
    comp["tide"] = {"score_10": round(_clamp_0_10(tide_score), 3), "score_100": int(round(_clamp_0_10(tide_score) * 10)), "reason": tide_reason}

    # WIND (0..10)
    try:
        pref_w = profile.get("preferred_wind_m_s", DEFAULT_PROFILE["preferred_wind_m_s"])
        wmin, wmax = float(pref_w[0]), float(pref_w[1])
        if wind is None:
            wind_score = 5.0
            wind_reason = "missing"
        else:
            wind_score = _linear_within_score_10(float(wind), wmin, wmax, tolerance=4.0)
            wind_reason = f"value={wind}"
    except Exception:
        wind_score = 5.0
        wind_reason = "error"
    comp["wind"] = {"score_10": round(_clamp_0_10(wind_score), 3), "score_100": int(round(_clamp_0_10(wind_score) * 10)), "reason": wind_reason}

    # WAVES (0..10)
    try:
        max_wave = float(profile.get("max_wave_height_m", DEFAULT_PROFILE["max_wave_height_m"]))
        if wave is None:
            wave_score = 5.0
            wave_reason = "missing"
        else:
            if wave <= max_wave:
                wave_score = 10.0
                wave_reason = f"ok (value={wave} <= max={max_wave})"
            else:
                limit = max_wave * 2.0 if max_wave > 0 else max_wave + 1.0
                if wave >= limit:
                    wave_score = 0.0
                else:
                    wave_score = 10.0 * (limit - wave) / (limit - max_wave)
                wave_reason = f"value={wave}"
    except Exception:
        wave_score = 5.0
        wave_reason = "error"
    comp["waves"] = {"score_10": round(_clamp_0_10(wave_score), 3), "score_100": int(round(_clamp_0_10(wave_score) * 10)), "reason": wave_reason}

    # TIME (0..10) - preferred_times in profile
    try:
        pref_times = profile.get("preferred_times", [])
        if not pref_times:
            # No preference -> neutral 5.0
            time_score = 5.0
            time_reason = "no_preference"
        else:
            # pref_times can be list of dict windows: {"start_hour":6,"end_hour":8}
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
        if pressure is None:
            pressure_score = 5.0
            pressure_reason = "missing"
        else:
            # map delta -10..+10 hPa to 0..10 (midpoint 0 -> 5)
            p = float(pressure)
            pressure_score = 5.0 + (p / 10.0) * 5.0
            pressure_score = _clamp_0_10(pressure_score)
            pressure_reason = f"delta={p}"
    except Exception:
        pressure_score = 5.0
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

    # MOON (0..10)
    try:
        moon_score = 5.0
        moon_reason = "default"
        if moon_phase is not None:
            # simple heuristic: near spring phases -> beneficial (higher)
            try:
                p = float(moon_phase)
                # map distance to nearest spring (0 or 0.5) -> 10..0 (0..0.25 -> 10..0)
                dist_new = abs(p - 0.0)
                dist_full = abs(p - 0.5)
                dist = min(dist_new, dist_full)
                moon_score = max(0.0, 10.0 * (1.0 - (dist / 0.25)))
                moon_reason = f"phase={p}"
            except Exception:
                moon_score = 5.0
                moon_reason = "bad_phase"
    except Exception:
        moon_score = 5.0
        moon_reason = "error"
    comp["moon"] = {"score_10": round(_clamp_0_10(moon_score), 3), "score_100": int(round(_clamp_0_10(moon_score) * 10)), "reason": moon_reason}

    # TEMPERATURE (0..10)
    try:
        pref_temp = profile.get("preferred_temp_c", DEFAULT_PROFILE["preferred_temp_c"])
        tmin, tmax = float(pref_temp[0]), float(pref_temp[1])
        if temp is None:
            temp_score = 5.0
            temp_reason = "missing"
        else:
            temp_score = _linear_within_score_10(float(temp), tmin, tmax, tolerance=10.0)
            temp_reason = f"value={temp}"
    except Exception:
        temp_score = 5.0
        temp_reason = "error"
    comp["temperature"] = {"score_10": round(_clamp_0_10(temp_score), 3), "score_100": int(round(_clamp_0_10(temp_score) * 10)), "reason": temp_reason}

    # Weighted aggregation (on 0..10)
    overall_10 = (
        weights.get("tide", 0.0) * comp["tide"]["score_10"]
        + weights.get("wind", 0.0) * comp["wind"]["score_10"]
        + weights.get("waves", 0.0) * comp["waves"]["score_10"]
        + weights.get("time", 0.0) * comp["time"]["score_10"]
        + weights.get("pressure", 0.0) * comp["pressure"]["score_10"]
        + weights.get("season", 0.0) * comp["season"]["score_10"]
        + weights.get("moon", 0.0) * comp["moon"]["score_10"]
        + weights.get("temperature", 0.0) * comp["temperature"]["score_10"]
    )
    overall_10 = float(round(overall_10, 3))
    overall_100 = int(round(overall_10 * 10.0))

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
    }
    return result