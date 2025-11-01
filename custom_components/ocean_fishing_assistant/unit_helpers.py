"""Unit conversion helper utilities shared across the integration.

All functions attempt to coerce to float and return None on failure.
Canonical units used by the integration:
- distance: meters (m) and kilometers (km)
- wind: meters/second (m/s)
- temperature: Celsius (°C)
- pressure: hectopascals (hPa)
- time periods: seconds (s)
"""
from typing import Any, Optional


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def kmh_to_m_s(v: Any) -> Optional[float]:
    """Convert km/h to m/s."""
    f = _to_float(v)
    if f is None:
        return None
    return f * 0.277777778


def mph_to_m_s(v: Any) -> Optional[float]:
    """Convert mph to m/s."""
    f = _to_float(v)
    if f is None:
        return None
    return f * 0.44704


def ft_to_m(v: Any) -> Optional[float]:
    """Convert feet to meters."""
    f = _to_float(v)
    if f is None:
        return None
    return f * 0.3048


def miles_to_km(v: Any) -> Optional[float]:
    """Convert miles to kilometers."""
    f = _to_float(v)
    if f is None:
        return None
    return f * 1.609344


def f_to_c(v: Any) -> Optional[float]:
    """Convert Fahrenheit to Celsius."""
    f = _to_float(v)
    if f is None:
        return None
    return (f - 32.0) * (5.0 / 9.0)


def inhg_to_hpa(v: Any) -> Optional[float]:
    """Convert inches of mercury to hectopascals (hPa)."""
    f = _to_float(v)
    if f is None:
        return None
    return f * 33.8638866667


import logging
from typing import Dict, Any, Tuple, List
from .const import DEFAULT_SAFETY_LIMITS

_LOGGER = logging.getLogger(__name__)

# Display <-> metric helpers

def m_s_to_kmh(v: Any) -> Optional[float]:
    f = _to_float(v)
    if f is None:
        return None
    return f * 3.6


def m_s_to_mph(v: Any) -> Optional[float]:
    f = _to_float(v)
    if f is None:
        return None
    return f * 2.2369362920544


def kmh_to_m_s_safe(v: Any) -> Optional[float]:
    return kmh_to_m_s(v)


def mph_to_m_s_safe(v: Any) -> Optional[float]:
    return mph_to_m_s(v)


def m_to_ft(v: Any) -> Optional[float]:
    f = _to_float(v)
    if f is None:
        return None
    return f / 0.3048


def ft_to_m_safe(v: Any) -> Optional[float]:
    return ft_to_m(v)


def c_to_f(v: Any) -> Optional[float]:
    f = _to_float(v)
    if f is None:
        return None
    return (f * 9.0 / 5.0) + 32.0


# High-level helpers for safety limits conversion

# Map UI/display keys to internal canonical keys
_DISPLAY_TO_CANONICAL = {
    "safety_max_wind": "max_wind_m_s",
    "safety_max_wave_height": "max_wave_height_m",
    "safety_min_visibility": "min_visibility_km",
    "safety_max_swell_period": "max_swell_period_s",
}


def convert_safety_display_to_metric(safety: Dict[str, Any], entry_units: str = "metric") -> Dict[str, Any]:
    """Convert safety values provided in UI/display units into canonical metric keys/values.
    - safety: keys as used in the config flow (safety_max_wind, safety_max_wave_height, ...)
    - entry_units: the units the UI was showing ("metric" or "imperial").

    Returns a dict with canonical metric keys (e.g. max_wind_m_s) and numeric metric values (floats) or None.
    """
    out: Dict[str, Any] = {}
    # Wind: UI shows km/h for metric, mph for imperial -> convert to m/s
    raw_wind = safety.get("safety_max_wind")
    if raw_wind is None:
        out["max_wind_m_s"] = None
    else:
        if entry_units == "metric":
            out["max_wind_m_s"] = kmh_to_m_s_safe(raw_wind)
        else:
            out["max_wind_m_s"] = mph_to_m_s_safe(raw_wind)

    # Wave height: UI shows m for metric, ft for imperial -> convert to m
    raw_wave = safety.get("safety_max_wave_height")
    if raw_wave is None:
        out["max_wave_height_m"] = None
    else:
        if entry_units == "metric":
            out["max_wave_height_m"] = _to_float(raw_wave)
        else:
            out["max_wave_height_m"] = ft_to_m_safe(raw_wave)

    # Visibility: UI shows km for metric, miles for imperial -> convert to km
    raw_vis = safety.get("safety_min_visibility")
    if raw_vis is None:
        out["min_visibility_km"] = None
    else:
        if entry_units == "metric":
            out["min_visibility_km"] = _to_float(raw_vis)
        else:
            # assume miles -> km
            miles = _to_float(raw_vis)
            out["min_visibility_km"] = miles_to_km(miles) if miles is not None else None

    # Swell period: UI shows seconds in both systems; keep as seconds
    raw_swell = safety.get("safety_max_swell_period")
    out["max_swell_period_s"] = _to_float(raw_swell) if raw_swell is not None else None

    return out


# Validation schema for canonical metric keys
_SAFETY_SCHEMA = {
    "max_wave_height_m": ("m", 0.0, 30.0, DEFAULT_SAFETY_LIMITS.get("max_wave_height_m", 2.5), False),
    "max_wind_m_s": ("m/s", 0.0, 60.0, DEFAULT_SAFETY_LIMITS.get("max_wind_m_s", 15.0), False),
    "min_visibility_km": ("km", 0.0, 200.0, None, True),
    "max_swell_period_s": ("s", 0.0, 120.0, None, True),
}


def validate_and_normalize_safety_limits(safety_limits: Dict[str, Any], strict: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    """Validate and normalize safety limits given in canonical metric keys.

    Returns (normalized_limits, warnings). On invalid inputs (parse/convert issues), the function will warn and use defaults or None (warn-and-clamp mode).
    If strict is True, ValueError will be raised on parse/convert failures.
    """
    normalized: Dict[str, Any] = {}
    warnings: List[str] = []

    for key, (unit, vmin, vmax, default, optional) in _SAFETY_SCHEMA.items():
        raw = safety_limits.get(key)
        if raw is None:
            normalized[key] = default
            continue
        # raw expected numeric already (float) because conversion done earlier
        try:
            val = float(raw)
        except Exception:
            msg = f"safety_limits[{key}] could not be interpreted as number: {raw!r}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg + "; using default/None")
            normalized[key] = default
            continue
        # clamp
        if val < vmin:
            warnings.append(f"safety_limits[{key}]={val} below min {vmin}; clamping to {vmin}")
            val = float(vmin)
        if val > vmax:
            warnings.append(f"safety_limits[{key}]={val} above max {vmax}; clamping to {vmax}")
            val = float(vmax)
        normalized[key] = float(val)

    return normalized, warnings