"""Tests for SafetyValidator class.

Tests the core logic by building a minimal SafetyValidator inline,
avoiding the HA-dependent package imports.
"""
from typing import Any, Dict, Optional

# Same check definitions as safety.py
_CHECK_DEFINITIONS = [
    ("wind", "max_wind_m_s", True, 0.9),
    ("wave", "max_wave_height_m", True, 0.9),
    ("gust", "max_gust_m_s", True, 0.9),
    ("visibility", "min_visibility_km", False, 1.1),
    ("swell_period", "min_swell_period_s", False, 1.1),
    ("precipitation", "max_precip_chance_pct", True, 0.9),
]


def _to_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


class SafetyValidator:
    """Minimal test version matching the production SafetyValidator logic."""
    def __init__(self, safety_limits=None):
        self._limits = {k: _to_float(v) for k, v in (safety_limits or {}).items() if v is not None}

    def check(self, **values):
        result = {"unsafe": False, "caution": False, "reasons": []}
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


# ---- Tests ----

def test_no_limits():
    v = SafetyValidator()
    r = v.check(wind=5.0)
    assert r == {"unsafe": False, "caution": False, "reasons": []}


def test_wind_unsafe():
    v = SafetyValidator({"max_wind_m_s": 10.0})
    r = v.check(wind=15.0)
    assert r["unsafe"] is True
    assert any("wind" in s for s in r["reasons"])


def test_wind_caution():
    v = SafetyValidator({"max_wind_m_s": 10.0})
    r = v.check(wind=9.5)
    assert r["unsafe"] is False
    assert r["caution"] is True


def test_wind_safe():
    v = SafetyValidator({"max_wind_m_s": 10.0})
    r = v.check(wind=5.0)
    assert r["unsafe"] is False and r["caution"] is False


def test_wave_unsafe():
    v = SafetyValidator({"max_wave_height_m": 2.0})
    assert v.check(wave=3.0)["unsafe"] is True


def test_wave_caution():
    v = SafetyValidator({"max_wave_height_m": 2.0})
    r = v.check(wave=1.9)
    assert r["unsafe"] is False and r["caution"] is True


def test_visibility_below():
    v = SafetyValidator({"min_visibility_km": 5.0})
    r = v.check(visibility=2.0)
    assert r["unsafe"] is True
    assert any("visibility" in s for s in r["reasons"])


def test_visibility_caution():
    v = SafetyValidator({"min_visibility_km": 5.0})
    r = v.check(visibility=5.3)
    assert r["unsafe"] is False and r["caution"] is True


def test_swell_below():
    v = SafetyValidator({"min_swell_period_s": 8.0})
    assert v.check(swell_period=5.0)["unsafe"] is True


def test_precip_over():
    v = SafetyValidator({"max_precip_chance_pct": 70.0})
    assert v.check(precipitation=90.0)["unsafe"] is True


def test_multiple_safe():
    v = SafetyValidator({"max_wind_m_s": 10.0, "max_wave_height_m": 2.0})
    r = v.check(wind=5.0, wave=1.0)
    assert r["unsafe"] is False and r["caution"] is False


def test_multiple_unsafe():
    v = SafetyValidator({"max_wind_m_s": 10.0, "max_wave_height_m": 2.0})
    r = v.check(wind=15.0, wave=3.0)
    assert r["unsafe"] is True
    assert len(r["reasons"]) >= 2


def test_none_values():
    v = SafetyValidator({"max_wind_m_s": 10.0})
    r = v.check(wind=None, wave=None)
    assert r["unsafe"] is False


def test_boundary_exact():
    v = SafetyValidator({"max_wind_m_s": 10.0})
    r = v.check(wind=10.0)
    assert r["unsafe"] is False
    assert r["caution"] is True


def test_empty_limits():
    v = SafetyValidator({})
    r = v.check(wind=100.0)
    assert r["unsafe"] is False


def test_gust_unsafe():
    v = SafetyValidator({"max_gust_m_s": 15.0})
    assert v.check(gust=20.0)["unsafe"] is True


def test_upper_and_lower_together():
    v = SafetyValidator({"max_wind_m_s": 10.0, "min_visibility_km": 5.0})
    assert v.check(wind=5.0, visibility=10.0)["unsafe"] is False
    r = v.check(wind=15.0, visibility=2.0)
    assert r["unsafe"] is True
    assert len(r["reasons"]) == 2


def test_swell_caution():
    v = SafetyValidator({"min_swell_period_s": 8.0})
    r = v.check(swell_period=8.5)
    assert r["unsafe"] is False and r["caution"] is True


def test_precip_caution():
    v = SafetyValidator({"max_precip_chance_pct": 70.0})
    r = v.check(precipitation=65.0)
    assert r["unsafe"] is False and r["caution"] is True


# ---- Run all ----

def _run():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {name} — {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed" + (f", {failed} FAILED" if failed else ", all passed"))
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
