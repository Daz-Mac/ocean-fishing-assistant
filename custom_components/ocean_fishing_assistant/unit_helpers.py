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