"""Unit conversion helper utilities shared across the integration.

All functions attempt to coerce to float and return None on failure.
Canonical units used by the integration:
- distance: meters (m) and kilometers (km)
- wind: meters/second (m/s)
- temperature: Celsius (°C)
- pressure: hectopascals (hPa)
- time periods: seconds (s)
- precipitation probability: percent (0..100) -> stored as _pct
"""
from typing import Any, Optional, Dict, Tuple, List
import logging
from .const import DEFAULT_SAFETY_LIMITS

_LOGGER = logging.getLogger(__name__)


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


# ---- Converters from display units -> canonical metric (m, m/s, °C, hPa, s) ----

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


# ---- Display helpers (canonical -> display) ----

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


def m_to_ft(v: Any) -> Optional[float]:
    f = _to_float(v)
    if f is None:
        return None
    return f / 0.3048


def c_to_f(v: Any) -> Optional[float]:
    f = _to_float(v)
    if f is None:
        return None
    return (f * 9.0 / 5.0) + 32.0

# ---- Convenience display helpers (canonical -> display value + unit) ----

def wind_m_s_to_display(v_m_s: Optional[float], units: str) -> Tuple[Optional[float], Optional[str]]:
    """Convert canonical wind m/s to display value + unit.

    Returns (value, unit) where value is numeric or None and unit is a string like "km/h" or "mph".
    """
    if v_m_s is None:
        return None, None
    if units == "metric":
        return m_s_to_kmh(v_m_s), "km/h"
    if units == "imperial":
        return m_s_to_mph(v_m_s), "mph"
    return v_m_s, "m/s"


def length_m_to_display(v_m: Optional[float], units: str) -> Tuple[Optional[float], Optional[str]]:
    """Convert meters to display length (m or ft)."""
    if v_m is None:
        return None, None
    if units == "imperial":
        return m_to_ft(v_m), "ft"
    return v_m, "m"


def temp_c_to_display(temp_c: Optional[float], units: str) -> Tuple[Optional[float], Optional[str]]:
    """Convert Celsius to display temperature and unit."""
    if temp_c is None:
        return None, None
    if units == "imperial":
        return c_to_f(temp_c), "\u00b0F"
    return temp_c, "\u00b0C"


def pressure_hpa_to_display(hpa: Optional[float], units: str) -> Tuple[Optional[float], Optional[str]]:
    """Convert hPa to display pressure and unit."""
    if hpa is None:
        return None, None
    if units == "imperial":
        return (hpa / 33.8638866667), "inHg"
    return hpa, "hPa"


def visibility_km_to_display(km: Optional[float], units: str) -> Tuple[Optional[float], Optional[str]]:
    """Convert kilometers to display visibility (km or miles)."""
    if km is None:
        return None, None
    if units == "imperial":
        return (km / 1.609344), "miles"
    return km, "km"



# ---- High-level helpers for safety limits conversion ----

# Map UI/display keys to internal canonical keys (kept for reference)
_DISPLAY_TO_CANONICAL = {
    "safety_max_wind": "max_wind_m_s",
    "safety_max_wave_height": "max_wave_height_m",
    "safety_min_visibility": "min_visibility_km",
    "safety_min_swell_period": "min_swell_period_s",
    "safety_max_gust": "max_gust_m_s",
    "safety_max_precip_chance": "max_precip_chance_pct",
}


def convert_safety_display_to_metric(safety: Dict[str, Any], entry_units: str = "metric") -> Dict[str, Any]:
    """Convert safety values provided in UI/display units into canonical metric keys/values.

    - safety: keys as used in the config flow (safety_max_wind, safety_max_gust, safety_max_wave_height, safety_min_swell_period, safety_max_precip_chance, ...)
    - entry_units: the units the UI was showing ("metric" or "imperial").

    Returns a dict with canonical metric keys (e.g. max_wind_m_s, max_gust_m_s) and numeric metric values (floats) or None.
    """
    out: Dict[str, Any] = {}
    # Wind: UI shows km/h for metric, mph for imperial -> convert to m/s
    raw_wind = safety.get("safety_max_wind")
    if raw_wind is None:
        out["max_wind_m_s"] = None
    else:
        if entry_units == "metric":
            out["max_wind_m_s"] = kmh_to_m_s(raw_wind)
        else:
            out["max_wind_m_s"] = mph_to_m_s(raw_wind)

    # Gust: same units as wind -> convert to m/s
    raw_gust = safety.get("safety_max_gust")
    if raw_gust is None:
        out["max_gust_m_s"] = None
    else:
        if entry_units == "metric":
            out["max_gust_m_s"] = kmh_to_m_s(raw_gust)
        else:
            out["max_gust_m_s"] = mph_to_m_s(raw_gust)

    # Wave height: UI shows m for metric, ft for imperial -> convert to m
    raw_wave = safety.get("safety_max_wave_height")
    if raw_wave is None:
        out["max_wave_height_m"] = None
    else:
        if entry_units == "metric":
            out["max_wave_height_m"] = _to_float(raw_wave)
        else:
            out["max_wave_height_m"] = ft_to_m(raw_wave)

    # Visibility: UI shows km for metric, miles for imperial -> convert to km
    raw_vis = safety.get("safety_min_visibility")
    if raw_vis is None:
        out["min_visibility_km"] = None
    else:
        if entry_units == "metric":
            out["min_visibility_km"] = _to_float(raw_vis)
        else:
            miles = _to_float(raw_vis)
            out["min_visibility_km"] = miles_to_km(miles) if miles is not None else None

    # Swell period: STRICT mapping — only accept 'safety_min_swell_period'
    raw_swell = safety.get("safety_min_swell_period")
    out["min_swell_period_s"] = _to_float(raw_swell) if raw_swell is not None else None

    # Precipitation chance: UI value is percent 0..100 -> store canonical pct
    raw_precip = safety.get("safety_max_precip_chance")
    out["max_precip_chance_pct"] = _to_float(raw_precip) if raw_precip is not None else None

    return out


# Validation schema for canonical metric keys
_SAFETY_SCHEMA = {
    "max_wave_height_m": ("m", 0.0, 30.0, DEFAULT_SAFETY_LIMITS.get("max_wave_height_m", 2.5), False),
    "max_wind_m_s": ("m/s", 0.0, 60.0, DEFAULT_SAFETY_LIMITS.get("max_wind_m_s", 15.0), False),
    "max_gust_m_s": ("m/s", 0.0, 80.0, DEFAULT_SAFETY_LIMITS.get("max_gust_m_s", None), True),
    "min_visibility_km": ("km", 0.0, 50.0, DEFAULT_SAFETY_LIMITS.get("min_visibility_km", None), True),
    "min_swell_period_s": ("s", 0.0, 30.0, DEFAULT_SAFETY_LIMITS.get("min_swell_period_s", None), True),
    "max_precip_chance_pct": ("%", 0.0, 100.0, DEFAULT_SAFETY_LIMITS.get("max_precip_chance_pct", None), True),
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
        try:
            val = float(raw)
        except Exception:
            msg = f"safety_limits[{key}] could not be interpreted as number: {raw!r}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg + "; using default/None")
            normalized[key] = default
            continue
        if val < vmin:
            warnings.append(f"safety_limits[{key}]={val} below min {vmin}; clamping to {vmin}")
            val = float(vmin)
        if val > vmax:
            warnings.append(f"safety_limits[{key}]={val} above max {vmax}; clamping to {vmax}")
            val = float(vmax)
        normalized[key] = float(val)

    return normalized, warnings