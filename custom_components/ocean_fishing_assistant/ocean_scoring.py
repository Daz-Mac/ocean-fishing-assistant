# custom_components/ocean_fishing_assistant/ocean_scoring.py
"""
Simplified, strict scoring module (fixed).

Changes since last version included:
- Pressure delta now prefers forward difference, falls back to backward difference,
  and finally defaults to a neutral 0.0 (no longer raises MissingDataError for end-of-series).
- Moon phase is only required when the species profile explicitly expresses a moon preference.
- Tide phase lookup now accepts either a top-level "tide_phase" key or a nested
  "tide" dict with "tide_phase" list/key (both canonical shapes supported).
- Keeps the rest of the strict checks for other profile-required inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

from zoneinfo import ZoneInfo

from .moon_utils import coerce_phase, matches_moon_preference
from . import unit_helpers

# Default factor weights
FACTOR_WEIGHTS = {
    "tide": 0.25,
    "wind": 0.15,
    "waves": 0.15,
    "time": 0.15,
    "pressure": 0.10,
    "season": 0.10,
    "moon": 0.05,
    "temperature": 0.05,
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


def _clamp_0_10(x: float) -> float:
    return max(0.0, min(10.0, float(x)))


def _validate_and_normalize_factor_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    if weights is None:
        total = sum(FACTOR_WEIGHTS.values()) or 1.0
        return {k: float(v) / total for k, v in FACTOR_WEIGHTS.items()}
    if set(weights.keys()) != set(FACTOR_WEIGHTS.keys()):
        raise ValueError("factor_weights must contain exact factor keys")
    norm = {k: float(weights[k]) for k in weights}
    total = sum(norm.values())
    if total <= 0:
        raise ValueError("factor_weights sum must be > 0")
    return {k: norm[k] / total for k in norm}


def _coerce_iso_dt(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    try:
        if isinstance(s, datetime):
            return s.astimezone(timezone.utc) if s.tzinfo else s.replace(tzinfo=timezone.utc)
        ss = str(s)
        if ss.endswith("Z"):
            ss = ss[:-1] + "+00:00"
        dt = datetime.fromisoformat(ss)
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        try:
            # numeric epoch
            v = float(s)
            if v > 1e12:
                v = v / 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except Exception:
            return None


def compute_score(
    data: Dict[str, Any],
    species_profile: Optional[Union[str, Dict[str, Any]]] = None,
    use_index: int = 0,
    safety_limits: Optional[Dict[str, Any]] = None,
    units: str = "metric",
    factor_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Strict compute for a single timestamp (index).
    This implementation assumes canonical inputs but avoids failing when moon or pressure-delta
    can be reasonably approximated or are not required by the species profile.
    """
    if "timestamps" not in data:
        raise MissingDataError("timestamps missing")
    timestamps = data["timestamps"]
    if use_index < 0 or use_index >= len(timestamps):
        raise MissingDataError("use_index out of range")

    # species_profile must be a dict (per strict policy)
    if not isinstance(species_profile, dict):
        raise MissingDataError("species_profile must be a dict")

    weights = _validate_and_normalize_factor_weights(factor_weights)

    tz = data.get("location_tz")
    if not tz:
        raise MissingDataError("location_tz required for time scoring")
    try:
        tzinfo = ZoneInfo(tz)
    except Exception as exc:
        raise MissingDataError(f"invalid timezone {tz}: {exc}")

    # helpers to fetch scalar at index
    def _get(key: str) -> Optional[float]:
        v = data.get(key)
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return _to_float_safe(v[use_index]) if use_index < len(v) else None
        return _to_float_safe(v)

    wind = _get("wind_m_s")
    gust = _get("wind_max_m_s")
    wave = _get("wave_height_m")
    temp = _get("temperature_c")
    pressure_arr = data.get("pressure_hpa")
    swell_period = _get("swell_period_s")

    # moon
    moon_val = None
    if "moon_phase" in data:
        mp = data.get("moon_phase")
        if isinstance(mp, (list, tuple)):
            moon_val = _to_float_safe(mp[use_index]) if use_index < len(mp) else None
        else:
            moon_val = _to_float_safe(mp)
        moon_val = coerce_phase(moon_val) if moon_val is not None else None

    # pressure delta: prefer forward difference, fall back to backward difference,
    # otherwise default to 0.0 (neutral). This avoids MissingDataError at series edges.
    pressure_delta: Optional[float] = None
    if isinstance(pressure_arr, (list, tuple)):
        p_curr = _to_float_safe(pressure_arr[use_index]) if use_index < len(pressure_arr) else None
        # try forward
        if use_index + 1 < len(pressure_arr):
            p_next = _to_float_safe(pressure_arr[use_index + 1])
            if p_curr is not None and p_next is not None:
                pressure_delta = float(p_next) - float(p_curr)
        # try backward as approximation
        if pressure_delta is None and use_index - 1 >= 0:
            p_prev = _to_float_safe(pressure_arr[use_index - 1])
            if p_curr is not None and p_prev is not None:
                # approximate forward delta as current minus previous
                pressure_delta = float(p_curr) - float(p_prev)
        # if still None -> neutral
        if pressure_delta is None:
            pressure_delta = 0.0
    else:
        # scalar pressure or missing series: neutral
        pressure_delta = 0.0

    # minimal missing requirements depending on profile preferences
    def _pref_is_set(key: str) -> bool:
        v = species_profile.get(key)
        return v is not None and not (isinstance(v, (list, tuple)) and len(v) == 0)

    missing = []
    if wind is None and _pref_is_set("preferred_wind_m_s"):
        missing.append("wind_m_s")
    if wave is None and _pref_is_set("max_wave_height_m"):
        missing.append("wave_height_m")
    if temp is None and _pref_is_set("preferred_temp_c"):
        missing.append("temperature_c")
    # moon only required if profile explicitly specifies a moon_preference (and not 'any')
    moon_pref = species_profile.get("moon_preference")
    if moon_val is None and moon_pref:
        # If moon_pref is truthy (could be list/string), require numeric moon_val
        missing.append("moon_phase")
    # pressure_delta is no longer treated as mandatory (we default it to neutral above)

    if missing:
        raise MissingDataError(f"Missing required inputs: {missing}")

    components: Dict[str, Dict[str, float]] = {}

    # TIDE (phase preference simple check)
    pref_tide = species_profile.get("preferred_tide_phase")
    if pref_tide:
        tide_phase = None

        # Accept tide_phase at top-level or under a nested 'tide' dict (canonical shapes)
        # 1) top-level "tide_phase"
        if "tide_phase" in data:
            tp = data.get("tide_phase")
            if isinstance(tp, (list, tuple)):
                tide_phase = tp[use_index] if use_index < len(tp) else None
            else:
                tide_phase = tp
        else:
            # 2) nested under data["tide"]["tide_phase"]
            tide_obj = data.get("tide")
            if isinstance(tide_obj, dict) and "tide_phase" in tide_obj:
                tp = tide_obj.get("tide_phase")
                if isinstance(tp, (list, tuple)):
                    tide_phase = tp[use_index] if use_index < len(tp) else None
                else:
                    tide_phase = tp

        # validate
        if tide_phase is None or not isinstance(tide_phase, str):
            raise MissingDataError("tide_phase required by profile but missing")
        # normalize comparison (case-insensitive)
        good_names = [str(x).lower() for x in (pref_tide if isinstance(pref_tide, (list, tuple)) else [pref_tide])]
        components["tide"] = {"score_10": 10.0 if str(tide_phase).lower() in good_names else 3.0}
    else:
        components["tide"] = {"score_10": 10.0}

    # WIND simple linear preference (single value or [min,max])
    pref_wind = species_profile.get("preferred_wind_m_s")
    if not pref_wind:
        components["wind"] = {"score_10": 10.0}
    else:
        if isinstance(pref_wind, (list, tuple)) and len(pref_wind) >= 2:
            wmin = float(pref_wind[0]); wmax = float(pref_wind[1])
        else:
            wmin = wmax = float(pref_wind)
        if wind is None:
            raise MissingDataError("wind required by profile")
        # tolerance 20% of max or 1.0 m/s minimum
        tol = max(1.0, 0.2 * max(1.0, wmax))
        if wmin <= wind <= wmax:
            ws = 10.0
        elif wind < wmin - tol or wind > wmax + tol:
            ws = 0.0
        elif wind < wmin:
            ws = 10.0 * (wind - (wmin - tol)) / (wmin - (wmin - tol))
        else:
            ws = 10.0 * ((wmax + tol) - wind) / (wmax + tol - wmax)
        components["wind"] = {"score_10": _clamp_0_10(ws)}

    # WAVES simple: prefer max_wave_height_m
    pref_wave = species_profile.get("max_wave_height_m")
    if not pref_wave:
        components["waves"] = {"score_10": 10.0}
    else:
        if wave is None:
            raise MissingDataError("wave_height_m required by profile")
        max_w = float(pref_wave)
        if wave <= 0.0:
            ws = 10.0
        elif wave >= max_w:
            ws = 0.0
        else:
            ws = 10.0 * (1.0 - (wave / max_w))
        # optionally blend swell preference if present
        pref_swell = species_profile.get("preferred_swell_period_s")
        if pref_swell and swell_period is not None:
            if isinstance(pref_swell, (list, tuple)) and len(pref_swell) >= 2:
                spmin = float(pref_swell[0]); spmax = float(pref_swell[1])
            else:
                spmin = spmax = float(pref_swell)
            # linear map for swell
            if spmin <= swell_period <= spmax:
                sp_score = 10.0
            else:
                sp_score = 0.0
            waves_score = (ws + sp_score) / 2.0
        else:
            waves_score = ws
        components["waves"] = {"score_10": _clamp_0_10(waves_score)}

    # TIME: simple hours matching (preferred_times expressed as hours list or absent -> full score)
    pref_times = species_profile.get("preferred_times") or []
    if not pref_times:
        components["time"] = {"score_10": 10.0}
    else:
        # normalize hours: accept list of ints or single int
        hours: List[int] = []
        if isinstance(pref_times, (list, tuple)):
            for it in pref_times:
                try:
                    hours.append(int(it) % 24)
                except Exception:
                    continue
        else:
            try:
                hours.append(int(pref_times) % 24)
            except Exception:
                hours = []
        # determine local hour
        t_iso = timestamps[use_index]
        dt = _coerce_iso_dt(t_iso)
        local_dt = dt.astimezone(tzinfo)
        hour = local_dt.hour
        if not hours:
            tscore = 10.0
        elif hour in hours:
            tscore = 10.0
        elif any(abs(hour - h) == 1 for h in hours):
            tscore = 8.0
        else:
            tscore = 3.0
        components["time"] = {"score_10": _clamp_0_10(tscore)}

    # PRESSURE: map pressure_delta to score (neutral if no meaningful delta)
    if pressure_delta is None:
        components["pressure"] = {"score_10": 5.0}
    else:
        if pressure_delta >= 2.0:
            ps = 10.0
        elif pressure_delta <= -2.0:
            ps = 0.0
        else:
            ps = 10.0 * ((pressure_delta + 2.0) / 4.0)
        components["pressure"] = {"score_10": _clamp_0_10(ps)}

    # SEASON
    months_pref = species_profile.get("preferred_months") or []
    if not months_pref:
        components["season"] = {"score_10": 10.0}
    else:
        dt = _coerce_iso_dt(timestamps[use_index])
        local_dt = dt.astimezone(tzinfo)
        components["season"] = {"score_10": 10.0 if local_dt.month in [int(m) for m in months_pref] else 3.0}

    # MOON
    moon_pref = species_profile.get("moon_preference") or []
    if not moon_pref:
        components["moon"] = {"score_10": 10.0}
    else:
        if moon_val is not None and matches_moon_preference(moon_val, moon_pref, tolerance=0.05):
            components["moon"] = {"score_10": 10.0}
        else:
            components["moon"] = {"score_10": 4.0}

    # TEMPERATURE
    pref_temp = species_profile.get("preferred_temp_c")
    if not pref_temp:
        components["temperature"] = {"score_10": 10.0}
    else:
        if temp is None:
            raise MissingDataError("temperature required by profile")
        if isinstance(pref_temp, (list, tuple)) and len(pref_temp) >= 2:
            pmin = float(pref_temp[0]); pmax = float(pref_temp[1])
        else:
            pmin = pmax = float(pref_temp)
        tol = float(species_profile.get("preferred_temp_tol_c") or 5.0)
        # linear within tolerance
        if pmin <= temp <= pmax:
            ts = 10.0
        elif temp < pmin - tol or temp > pmax + tol:
            ts = 0.0
        elif temp < pmin:
            ts = 10.0 * (temp - (pmin - tol)) / (pmin - (pmin - tol))
        else:
            ts = 10.0 * ((pmax + tol) - temp) / (pmax + tol - pmax)
        components["temperature"] = {"score_10": _clamp_0_10(ts)}

    # Combine to overall score using weights
    overall_10 = 0.0
    for k, w in weights.items():
        comp_score = components.get(k, {}).get("score_10")
        overall_10 += w * (comp_score if comp_score is not None else 10.0)
    overall_10 = round(overall_10, 3)
    overall_100 = int(round(overall_10 * 10.0))

    # Evaluate simple safety flags
    safety = {"unsafe": False, "caution": False, "reasons": []}
    if safety_limits:
        max_wind = _to_float_safe(safety_limits.get("max_wind_m_s"))
        if max_wind is not None and wind is not None:
            if wind > max_wind:
                safety["unsafe"] = True
                safety["reasons"].append(f"wind>{max_wind}")
            elif wind > 0.9 * max_wind:
                safety["caution"] = True
                safety["reasons"].append("wind_near_limit")
        max_wave = _to_float_safe(safety_limits.get("max_wave_height_m"))
        if max_wave is not None and wave is not None:
            if wave > max_wave:
                safety["unsafe"] = True
                safety["reasons"].append(f"wave>{max_wave}")
            elif wave > 0.9 * max_wave:
                safety["caution"] = True
                safety["reasons"].append("wave_near_limit")

    # Breaches: a simple list based on species preference mismatches (kept minimal)
    breaches: List[Dict[str, Any]] = []
    # temperature breach
    try:
        if pref_temp and temp is not None:
            if temp < (pmin - tol):
                breaches.append({"variable": "temperature", "value": temp, "severity": "unsafe"})
            elif temp < pmin:
                breaches.append({"variable": "temperature", "value": temp, "severity": "caution"})
    except Exception:
        pass

    result = {
        "score_10": overall_10,
        "score_100": overall_100,
        "components": {k: {"score_10": round(v["score_10"], 3)} for k, v in components.items()},
        "raw": {
            "wind": wind,
            "wave": wave,
            "pressure_delta": pressure_delta,
            "temperature": temp,
            "timestamp": timestamps[use_index],
            "moon_phase": moon_val,
            "wind_gust": gust,
            "swell_period_s": swell_period,
        },
        "profile_used": dict(species_profile),
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
    timestamps = payload["timestamps"]
    for idx, ts in enumerate(timestamps):
        try:
            res = compute_score(
                payload,
                species_profile=species_profile or {},
                use_index=idx,
                safety_limits=safety_limits,
                units=units,
                factor_weights=factor_weights,
            )
            forecast_raw = {
                "formatted_weather": {
                    "temperature": payload.get("temperature_c")[idx] if isinstance(payload.get("temperature_c"), (list, tuple)) else payload.get("temperature_c"),
                    "wind": payload.get("wind_m_s")[idx] if isinstance(payload.get("wind_m_s"), (list, tuple)) else payload.get("wind_m_s"),
                    "wind_gust": payload.get("wind_max_m_s")[idx] if isinstance(payload.get("wind_max_m_s"), (list, tuple)) else payload.get("wind_max_m_s"),
                    "wave_height_m": payload.get("wave_height_m")[idx] if isinstance(payload.get("wave_height_m"), (list, tuple)) else payload.get("wave_height_m"),
                    "wave_period_s": payload.get("wave_period_s")[idx] if isinstance(payload.get("wave_period_s"), (list, tuple)) else payload.get("wave_period_s"),
                    "swell_period_s": payload.get("swell_period_s")[idx] if isinstance(payload.get("swell_period_s"), (list, tuple)) else payload.get("swell_period_s"),
                    "pressure_hpa": payload.get("pressure_hpa")[idx] if isinstance(payload.get("pressure_hpa"), (list, tuple)) else payload.get("pressure_hpa"),
                },
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
            entry = {
                "timestamp": ts,
                "index": idx,
                "score_10": None,
                "score_100": None,
                "components": None,
                "forecast_raw": {"error": "missing required data", "details": str(mde)},
                "profile_used": None,
                "safety": {"unsafe": False, "caution": False, "reasons": []},
                "breaches": [],
            }
        except Exception:
            entry = {
                "timestamp": ts,
                "index": idx,
                "score_10": None,
                "score_100": None,
                "components": None,
                "forecast_raw": {"error": "unexpected error"},
                "profile_used": None,
                "safety": {"unsafe": False, "caution": False, "reasons": []},
                "breaches": [],
            }
        out.append(entry)
    return out