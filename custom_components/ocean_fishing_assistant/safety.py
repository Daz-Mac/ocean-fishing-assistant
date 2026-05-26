"""Safety limit validation for Ocean Fishing Assistant.

Provides table-driven bounds checking against configurable safety limits.
Replaces inline if-else chains across coordinator.py and ocean_scoring.py.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Table-driven check definitions:
# (value_key, limit_key, upper_bound, near_ratio)
#   upper_bound=True:  value > limit → unsafe,  value > limit*near_ratio → caution
#   upper_bound=False: value < limit → unsafe,  value < limit*near_ratio → caution
_CHECK_DEFINITIONS = [
    ("wind", "max_wind_m_s", True, 0.9),
    ("wave", "max_wave_height_m", True, 0.9),
    ("gust", "max_gust_m_s", True, 0.9),
    ("visibility", "min_visibility_km", False, 1.1),
    ("swell_period", "min_swell_period_s", False, 1.1),
    ("precipitation", "max_precip_chance_pct", True, 0.9),
]


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


class SafetyValidator:
    """Table-driven safety limit checker.

    Evaluates current conditions against configured safety limits and returns
    structured safety results (unsafe/caution/reasons).

    Typical usage:
        validator = SafetyValidator(safety_limits)
        result = validator.check(wind=wind_m_s, wave=wave_height_m, ...)
    """

    def __init__(self, safety_limits: Optional[Dict[str, Any]] = None):
        self._limits = {k: _to_float(v) for k, v in (safety_limits or {}).items() if v is not None}

    def check(self, **values: Any) -> Dict[str, Any]:
        """Evaluate values against configured limits.

        Returns:
            {"unsafe": bool, "caution": bool, "reasons": [str]}
        """
        result: Dict[str, Any] = {"unsafe": False, "caution": False, "reasons": []}

        if not self._limits:
            return result

        for value_key, limit_key, upper_bound, near_ratio in _CHECK_DEFINITIONS:
            limit = self._limits.get(limit_key)
            val = _to_float(values.get(value_key))

            if limit is None or val is None:
                continue

            if upper_bound:
                if val > limit:
                    result["unsafe"] = True
                    result["reasons"].append(f"{value_key}>{limit}")
                elif val > limit * near_ratio:
                    result["caution"] = True
                    result["reasons"].append(f"{value_key}_near_limit")
            else:
                if val < limit:
                    result["unsafe"] = True
                    result["reasons"].append(f"{value_key}<{limit}")
                elif val < limit * near_ratio:
                    result["caution"] = True
                    result["reasons"].append(f"{value_key}_near_limit")

        return result
