"""Tests for data_formatter.py — validation, unit conversion, period building.

Runs via import bootstrap to load production code without HA dependencies.
"""
from __future__ import annotations

import math
from typing import Any, Dict

from _scoring_bootstrap import load_ocean_module

data_formatter = load_ocean_module("data_formatter")
ocean_scoring = load_ocean_module("ocean_scoring")

# Convenience references
_merge_breach_example = data_formatter._merge_breach_example

# ---------------------------------------------------------------------------
# 3a. _merge_breach_example tests
# ---------------------------------------------------------------------------

def _make_breach(**overrides):
    b: Dict[str, Any] = {
        "variable": "test",
        "value": 5.0,
        "severity": "caution",
        "reason": "test_breach",
        "unit": "m/s",
    }
    b.update(overrides)
    return b


def test_merge_wind_metric():
    result = _merge_breach_example(_make_breach(), "metric")
    # 5 m/s → 18.0 km/h
    assert "18.0 km/h" in str(result["value"])


def test_merge_wind_imperial():
    result = _merge_breach_example(_make_breach(), "imperial")
    # 5 m/s → 11.18 mph
    assert "mph" in str(result["value"])


def test_merge_temperature():
    b = _make_breach(value=18.5, unit="°C")
    result = _merge_breach_example(b, "metric")
    assert "°C" in str(result["value"])


def test_merge_hour_no_decimals():
    b = _make_breach(value=14, unit="hour")
    result = _merge_breach_example(b, "metric")
    # Production code uses round(14, 0) → 14.0 → "14.0 hour"
    assert "hour" in str(result["value"])


def test_merge_already_has_unit():
    b = _make_breach(value="5.0 m/s", unit="m/s")
    result = _merge_breach_example(b, "metric")
    # Already has unit in value string → passes through
    assert result["value"] == "5.0 m/s"


def test_merge_none_value():
    b = _make_breach(value=None)
    result = _merge_breach_example(b, "metric")
    # Missing value → passes through
    assert result["value"] is None


# ---------------------------------------------------------------------------
# 3b. validate() rejection cases
# ---------------------------------------------------------------------------

def _make_minimal_raw():
    return {
        "hourly": {
            "time": ["2026-05-29T10:00:00Z", "2026-05-29T11:00:00Z"],
            "temperature_2m": [18.0, 19.0],
            "wind_speed_10m": [5.0, 6.0],
            "wind_direction_10m": [90, 100],
            "pressure_msl": [1013.0, 1015.0],
            "precipitation_probability": [10, 20],
            "wave_height": [1.0, 1.2],
            "wave_period": [8.0, 9.0],
        },
        "hourly_units": {
            "wind_speed_10m": "km/h",
        },
        "location_tz": "Europe/Gibraltar",
        "tide": {
            "tide_phase": ["high", "falling"],
            "moon_phase": [0.5, 0.5],
        },
    }


def test_reject_non_dict():
    df = data_formatter.DataFormatter({})
    try:
        df.validate("not a dict")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_reject_missing_hourly():
    df = data_formatter.DataFormatter({})
    try:
        df.validate({})
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_reject_empty_time():
    df = data_formatter.DataFormatter({})
    try:
        df.validate({"hourly": {"time": []}})
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_reject_no_temperature():
    df = data_formatter.DataFormatter({})
    try:
        df.validate({"hourly": {"time": ["2026-05-29T10:00:00Z"], "wind_speed_10m": [5.0]}})
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_reject_no_wind():
    df = data_formatter.DataFormatter({})
    try:
        df.validate({"hourly": {"time": ["2026-05-29T10:00:00Z"], "temperature_2m": [18.0]}})
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 3c. _convert_wind_array_value tests
# ---------------------------------------------------------------------------

def test_convert_wind_kmh_to_ms():
    df = data_formatter.DataFormatter({})
    result = df._convert_wind_array_value(36, "km/h")
    assert abs(result - 10.0) < 0.01  # 36 km/h = 10 m/s


def test_convert_wind_mph_to_ms():
    df = data_formatter.DataFormatter({})
    result = df._convert_wind_array_value(10, "mph")
    assert abs(result - 4.47) < 0.01  # 10 mph ≈ 4.47 m/s


def test_convert_wind_already_ms():
    df = data_formatter.DataFormatter({})
    result = df._convert_wind_array_value(5.0, "m/s")
    assert result == 5.0


def test_convert_wind_none():
    df = data_formatter.DataFormatter({})
    try:
        df._convert_wind_array_value(None, "m/s")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 3c. validate() unit conversion
# ---------------------------------------------------------------------------

def test_validate_wind_conversion_kmh():
    """Wind in km/h gets converted to m/s in canonical output."""
    raw = _make_minimal_raw()
    df = data_formatter.DataFormatter({})
    ppi = {"2026-05-29": {"period_0": {"indices": [0]}}}
    result = df.validate(raw, precomputed_period_indices=ppi,
                         species_profile={"common_name": "Test"})
    wind = result.get("wind_m_s", [])
    # Input: 5 km/h → 1.39 m/s, 6 km/h → 1.67 m/s
    assert len(wind) == 2
    assert abs(wind[0] - 1.39) < 0.05


def test_validate_preserves_moon_phase():
    """Moon phase array is preserved in canonical output."""
    raw = _make_minimal_raw()
    df = data_formatter.DataFormatter({})
    ppi = {"2026-05-29": {"period_0": {"indices": [0]}}}
    result = df.validate(raw, precomputed_period_indices=ppi,
                         species_profile={"common_name": "Test"})
    assert "moon_phase" in result
    assert len(result["moon_phase"]) == 2


def test_validate_preserves_tide_phase():
    """Tide phase array is preserved in canonical output."""
    raw = _make_minimal_raw()
    df = data_formatter.DataFormatter({})
    ppi = {"2026-05-29": {"period_0": {"indices": [0]}}}
    result = df.validate(raw, precomputed_period_indices=ppi,
                         species_profile={"common_name": "Test"})
    assert "tide_phase" in result
    assert result["tide_phase"] == ["high", "falling"]


def test_validate_location_tz():
    raw = _make_minimal_raw()
    df = data_formatter.DataFormatter({})
    ppi = {"2026-05-29": {"period_0": {"indices": [0]}}}
    result = df.validate(raw, precomputed_period_indices=ppi,
                         species_profile={"common_name": "Test"})
    assert result.get("location_tz") == "Europe/Gibraltar"


# ---------------------------------------------------------------------------
# 3d. validate() period building
# ---------------------------------------------------------------------------

def test_period_single():
    """Single period → score summary built correctly."""
    raw = _make_minimal_raw()
    df = data_formatter.DataFormatter({})
    ppi = {"2026-05-29": {"morning": {"indices": [0]}}}
    result = df.validate(raw, precomputed_period_indices=ppi,
                         species_profile={"common_name": "Test"})
    assert "period_forecasts" in result
    pf = result["period_forecasts"]
    assert "2026-05-29" in pf
    assert "morning" in pf["2026-05-29"]
    morning = pf["2026-05-29"]["morning"]
    assert "score_10" in morning
    assert "score_100" in morning
    assert morning["score_10"] is not None


def test_period_multiple():
    """Multiple periods → each period has its own summary."""
    raw = _make_minimal_raw()
    df = data_formatter.DataFormatter({})
    ppi = {"2026-05-29": {"morning": {"indices": [0]}, "afternoon": {"indices": [1]}}}
    result = df.validate(raw, precomputed_period_indices=ppi,
                         species_profile={"common_name": "Test"})
    pf = result["period_forecasts"]["2026-05-29"]
    assert "morning" in pf
    assert "afternoon" in pf
    assert pf["morning"]["score_10"] is not None
    assert pf["afternoon"]["score_10"] is not None


def test_period_breach_included():
    """Period with breach → breach included in summary."""
    raw = _make_minimal_raw()
    # Set temperature far outside any preference to trigger breach
    raw["hourly"]["temperature_2m"] = [5.0, 19.0]  # first one is cold
    df = data_formatter.DataFormatter({})
    ppi = {"2026-05-29": {"morning": {"indices": [0]}}}
    profile = {"common_name": "Test", "preferred_temp_c": [15, 20]}
    result = df.validate(raw, precomputed_period_indices=ppi,
                         species_profile=profile)
    pf = result["period_forecasts"]["2026-05-29"]["morning"]
    assert "breaches" in pf
    bv = pf["breaches"].get("by_variable", {})
    assert "temperature" in bv


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _run():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR: {name} — {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed + failed} passed" + (f", {failed} FAILED" if failed else ", all passed"))
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
