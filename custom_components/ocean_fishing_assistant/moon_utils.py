# custom_components/ocean_fishing_assistant/moon_utils.py
from typing import Any, Optional, Iterable, Union

# canonical mapping: common textual tokens -> fractional phase (0..1)
_NAME_TO_FRAC = {
    "new": 0.0,
    "new_moon": 0.0,
    "newmoon": 0.0,
    "first_quarter": 0.25,
    "first": 0.25,
    "waxing": 0.25,
    "full": 0.5,
    "full_moon": 0.5,
    "fullmoon": 0.5,
    "last_quarter": 0.75,
    "last": 0.75,
    "waning": 0.75,
}


def coerce_phase(phase: Any) -> Optional[float]:
    """Coerce numeric or numeric-like phase input to a float in [0.0, 1.0], or None."""
    if phase is None:
        return None
    try:
        p = float(phase)
    except Exception:
        return None
    p = p % 1.0
    if p < 0.0:
        p += 1.0
    return float(p)


def name_to_fraction(name: Optional[str]) -> Optional[float]:
    """Map a textual moon token (case-insensitive) to canonical fraction, or None."""
    if not name:
        return None
    key = str(name).strip().lower().replace(" ", "_")
    return _NAME_TO_FRAC.get(key)


def fraction_to_name(frac: Optional[float], tolerance: float = 0.035) -> Optional[str]:
    """Map numeric fractional moon phase to friendly name (e.g. 'Full Moon').
    Returns None if frac is None or cannot be mapped.
    Tolerance controls how close to exact phase (0.0, 0.25, 0.5, 0.75, 1.0)
    the phase must be to show the named phase rather than the transitional name.
    Default 0.035 = ±3.5% of lunar cycle, matching astronomical convention for
    "near enough to call it" and within the spring tide bonus check (±5%).
    """
    if frac is None:
        return None
    try:
        p = float(frac) % 1.0
    except Exception:
        return None
    eps = max(tolerance, 1e-6)
    if p <= eps or abs(p - 1.0) <= eps:
        return "New Moon"
    if abs(p - 0.25) <= eps:
        return "First Quarter"
    if abs(p - 0.5) <= eps:
        return "Full Moon"
    if abs(p - 0.75) <= eps:
        return "Last Quarter"
    if 0.0 < p < 0.25:
        return "Waxing Crescent"
    if 0.25 < p < 0.5:
        return "Waxing Gibbous"
    if 0.5 < p < 0.75:
        return "Waning Gibbous"
    if 0.75 < p < 1.0:
        return "Waning Crescent"
    return None


def matches_moon_preference(phase: Any, pref: Union[str, float, Iterable[Union[str, float]], None], tolerance: float = 0.05) -> bool:
    """
    Given a numeric phase (or phase-like), and a preference token/list from species profile,
    determine whether the phase matches.

    - pref may be:
      * None -> interpreted as 'no preference' => returns True
      * "any" -> always True
      * single token (string or numeric) -> match if within tolerance or name matches
      * iterable of tokens -> True if any element matches
    """
    if pref is None:
        return True
    if isinstance(pref, (str, bytes)) and str(pref).strip().lower() == "any":
        return True

    pval = coerce_phase(phase)

    def _single_match(single):
        if single is None:
            return False
        # numeric input
        if isinstance(single, (int, float)):
            try:
                sf = float(single) % 1.0
                if pval is None:
                    return False
                return abs(sf - pval) <= float(tolerance)
            except Exception:
                return False
        s = str(single).strip().lower()
        if s == "any":
            return True
        frac = name_to_fraction(s)
        if frac is not None and pval is not None:
            return abs(frac - pval) <= float(tolerance)
        # numeric string fallback
        try:
            sf = float(s)
            if pval is None:
                return False
            return abs((sf % 1.0) - pval) <= float(tolerance)
        except Exception:
            pass
        return False

    if isinstance(pref, (list, tuple, set)):
        for it in pref:
            if _single_match(it):
                return True
        return False

    return _single_match(pref)