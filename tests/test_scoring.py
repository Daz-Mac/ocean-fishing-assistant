"""Tests for ocean_scoring.py — all 8 scoring factors, safety capping, breach detection.

Runs via import bootstrap to load the real production code without HA dependencies.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from _scoring_bootstrap import load_ocean_module

scoring = load_ocean_module("ocean_scoring")
const = load_ocean_module("const")

compute_score = scoring.compute_score
compute_forecast = scoring.compute_forecast
MissingDataError = scoring.MissingDataError

# ---------------------------------------------------------------------------
# 2a. Pure helper tests (copied inline)
# ---------------------------------------------------------------------------

def _to_float_safe(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _clamp_0_10(x):
    return max(0.0, min(10.0, float(x)))


def test_to_float_safe_none():
    assert _to_float_safe(None) is None


def test_to_float_safe_int():
    assert _to_float_safe(42) == 42.0


def test_to_float_safe_float():
    assert _to_float_safe(3.14) == 3.14


def test_to_float_safe_string():
    assert _to_float_safe("5") == 5.0


def test_to_float_safe_invalid():
    assert _to_float_safe("abc") is None


def test_clamp_below():
    assert _clamp_0_10(-5) == 0.0


def test_clamp_mid():
    assert _clamp_0_10(3) == 3.0


def test_clamp_above():
    assert _clamp_0_10(12) == 10.0


def test_clamp_zero():
    assert _clamp_0_10(0) == 0.0


def test_clamp_ten():
    assert _clamp_0_10(10) == 10.0


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------

MINIMAL_PAYLOAD = {
    "timestamps": ["2026-05-29T12:00:00Z"],
    "location_tz": "Europe/Gibraltar",
    "wind_m_s": [5.0],
    "wave_height_m": [1.0],
    "temperature_c": [18.0],
    "pressure_hpa": [1013.0, 1015.0],
    "moon_phase": [0.5],
    "tide_phase": ["high"],
}

MINIMAL_PROFILE = {
    "common_name": "Test Species",
    "preferred_months": [],
    "preferred_temp_c": [],
    "preferred_wind_m_s": [],
    "preferred_swell_period_s": [],
    "max_wave_height_m": [],
    "preferred_tide_phase": [],
    "preferred_times": [],
    "moon_preference": [],
}


def _build_payload(**overrides):
    """Merge overrides into MINIMAL_PAYLOAD and return a copy."""
    p = dict(MINIMAL_PAYLOAD)
    p.update(overrides)
    return p


def _build_profile(**overrides):
    p = dict(MINIMAL_PROFILE)
    p.update(overrides)
    return p


# ---------------------------------------------------------------------------
# 2b. Scoring helper tests (via bootstrap)
# ---------------------------------------------------------------------------

# _linear_within_score_10
def test_linear_within_at_center():
    pref_min, pref_max, tol = 5.0, 15.0, 3.0
    result = scoring._linear_within_score_10(10.0, pref_min, pref_max, tol)
    assert result == 10.0


def test_linear_within_in_range():
    result = scoring._linear_within_score_10(7.0, 5.0, 15.0, 3.0)
    assert result == 10.0


def test_linear_within_at_tolerance_boundary():
    # value = low - tolerance = 5 - 3 = 2
    result = scoring._linear_within_score_10(2.0, 5.0, 15.0, 3.0)
    assert result == 0.0


def test_linear_within_outside_tolerance():
    result = scoring._linear_within_score_10(1.0, 5.0, 15.0, 3.0)
    assert result == 0.0


def test_linear_within_halfway_in():
    # value = low - tolerance/2 = 5 - 1.5 = 3.5
    result = scoring._linear_within_score_10(3.5, 5.0, 15.0, 3.0)
    assert result == 5.0


def test_linear_within_equal_min_max():
    result = scoring._linear_within_score_10(6.0, 6.0, 6.0, 2.0)
    assert result == 10.0


def test_linear_within_equal_min_max_outside():
    # Special case: when min==max, it uses value ± tolerance as effective range (4 to 8)
    # value=3 is halfway between span_low=4 and low=4-tolerance=2 → 10 * (3-2)/(4-2) = 5
    result = scoring._linear_within_score_10(3.0, 6.0, 6.0, 2.0)
    assert result == 5.0


def test_linear_within_high_side():
    # value = high + tolerance = 15 + 3 = 18
    result = scoring._linear_within_score_10(18.0, 5.0, 15.0, 3.0)
    assert result == 0.0


# _validate_and_normalize_factor_weights
def test_weights_none_returns_defaults():
    w = scoring._validate_and_normalize_factor_weights(None)
    expected_total = sum(scoring.FACTOR_WEIGHTS.values())
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_weights_valid_normalizes():
    custom = {k: 10.0 for k in scoring.FACTOR_WEIGHTS}
    w = scoring._validate_and_normalize_factor_weights(custom)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v > 0 for v in w.values())


def test_weights_missing_key():
    bad = dict(scoring.FACTOR_WEIGHTS)
    bad.pop("tide")
    try:
        scoring._validate_and_normalize_factor_weights(bad)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_weights_extra_key():
    bad = dict(scoring.FACTOR_WEIGHTS)
    bad["extra"] = 1.0
    try:
        scoring._validate_and_normalize_factor_weights(bad)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_weights_negative():
    bad = {k: 10.0 for k in scoring.FACTOR_WEIGHTS}
    bad["tide"] = -1.0
    try:
        scoring._validate_and_normalize_factor_weights(bad)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_weights_zero_sum():
    bad = {k: 0.0 for k in scoring.FACTOR_WEIGHTS}
    try:
        scoring._validate_and_normalize_factor_weights(bad)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_weights_non_dict():
    try:
        scoring._validate_and_normalize_factor_weights("invalid")
        assert False, "Expected ValueError"
    except ValueError:
        pass


# _format_safety_reason
def test_format_wind_exceeds():
    msg = scoring._format_safety_reason("wind>15", {}, "metric")
    assert "Wind exceeds" in msg
    assert "km/h" in msg


def test_format_wave_exceeds():
    msg = scoring._format_safety_reason("wave>2.5", {}, "metric")
    assert "Wave height exceeds" in msg


def test_format_visibility_below():
    msg = scoring._format_safety_reason("vis<5", {}, "metric")
    assert "Visibility below" in msg


def test_format_wind_near_limit():
    msg = scoring._format_safety_reason("wind_near_limit", {"max_wind_m_s": 15.0}, "metric")
    assert "approaching" in msg


def test_format_gust_exceeds():
    msg = scoring._format_safety_reason("gust>20", {}, "metric")
    assert "Gust exceeds" in msg


def test_format_precip_exceeds():
    msg = scoring._format_safety_reason("precipitation>70", {}, "metric")
    assert "Precipitation chance exceeds" in msg


def test_format_unknown_code():
    msg = scoring._format_safety_reason("unknown_code", {}, "metric")
    assert msg == "unknown_code"


def test_format_imperial_wind():
    msg = scoring._format_safety_reason("wind>15", {}, "imperial")
    assert "mph" in msg


# _coerce_datetime
def test_coerce_iso_string():
    dt = scoring._coerce_datetime("2026-05-29T12:00:00Z")
    assert dt is not None
    assert dt.hour == 12
    assert dt.tzinfo is not None


def test_coerce_timestamp_seconds():
    dt = scoring._coerce_datetime(1716974400.0)
    assert dt is not None
    assert dt.tzinfo is not None


def test_coerce_timestamp_ms():
    dt = scoring._coerce_datetime(1716974400000)
    assert dt is not None


def test_coerce_datetime_object():
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    dt = scoring._coerce_datetime(now)
    assert dt is not None


def test_coerce_none():
    assert scoring._coerce_datetime(None) is None


def test_coerce_invalid():
    assert scoring._coerce_datetime("not-a-date") is None


# _resolve_tide_phase_at
def test_resolve_tide_phase_list():
    result = scoring._resolve_tide_phase_at(
        {"tide_phase": ["low", "high", "rising"]}, 1
    )
    assert result == "high"


def test_resolve_tide_phase_scalar():
    result = scoring._resolve_tide_phase_at({"tide_phase": "high"}, 0)
    assert result == "high"


def test_resolve_tide_phase_nested_list():
    result = scoring._resolve_tide_phase_at({"tide": {"tide_phase": ["low", "high"]}}, 1)
    assert result == "high"


def test_resolve_tide_phase_nested_scalar():
    result = scoring._resolve_tide_phase_at({"tide": {"tide_phase": "rising"}}, 0)
    assert result == "rising"


def test_resolve_tide_phase_out_of_range():
    result = scoring._resolve_tide_phase_at(
        {"tide_phase": ["low", "high"]}, 5
    )
    assert result is None


def test_resolve_tide_phase_missing():
    result = scoring._resolve_tide_phase_at({}, 0)
    assert result is None


# ---------------------------------------------------------------------------
# 2c. _normalize_preferred_times (tested via compute_score)
# ---------------------------------------------------------------------------

# Europe/Gibraltar in May uses CEST (UTC+2). UTC time in the payload gets
# converted to local time via location_tz before time scoring. The helper
# builds a payload at a given UTC hour; compute_score converts to local.
_HOUR_OFFSET = 2  # CEST offset in May


def _score_for_hour(local_hour, preferred_times):
    """Build a payload at a given LOCAL hour and check time component score."""
    utc_hour = (local_hour - _HOUR_OFFSET) % 24
    ts = f"2026-05-29T{utc_hour:02d}:00:00Z"
    payload = _build_payload(timestamps=[ts])
    profile = _build_profile(preferred_times=preferred_times)
    result = compute_score(payload, species_profile=profile, use_index=0)
    return result["components"]["time"]["score_10"]


def test_time_within_range():
    score = _score_for_hour(13, [{"start_hour": 12, "end_hour": 14}])
    assert score == 10.0


def test_time_distance_2():
    # local 16: distance to 14 = 2 → expected 5.0
    score = _score_for_hour(16, [{"start_hour": 12, "end_hour": 14}])
    assert score == 5.0


def test_time_distance_3():
    # local 17: distance to 14 = 3 → expected 2.0
    score = _score_for_hour(17, [{"start_hour": 12, "end_hour": 14}])
    assert score == 2.0


def test_time_distance_gt_3():
    score = _score_for_hour(18, [{"start_hour": 12, "end_hour": 14}])
    assert score == 0.0


def test_time_day_token():
    # "day" token = hours 7-16 local. local hour 10 → within range → 10.0
    score = _score_for_hour(10, ["day"])
    assert score == 10.0


def test_time_night_token():
    # "night" = all hours not 7-16. local hour 22 → night → 10.0
    score = _score_for_hour(22, ["night"])
    assert score == 10.0


def test_time_all_day_token():
    score = _score_for_hour(3, ["all_day"])
    assert score == 10.0


def test_time_no_preference():
    score = _score_for_hour(12, [])
    assert score == 10.0


def test_time_dawn_precomputed():
    payload = _build_payload(
        timestamps=["2026-05-29T05:00:00Z"],
        period_forecasts={
            "2026-05-29": {
                "dawn": {"indices": [0]}
            }
        }
    )
    profile = _build_profile(preferred_times=[{"start": "dawn"}])
    result = compute_score(payload, species_profile=profile, use_index=0)
    score = result["components"]["time"]["score_10"]
    assert score == 10.0


# ---------------------------------------------------------------------------
# 2d. compute_score integration tests
# ---------------------------------------------------------------------------

# ---- Missing data / error cases ----

def test_missing_timestamps():
    try:
        compute_score({}, species_profile=MINIMAL_PROFILE)
        assert False, "Expected MissingDataError"
    except MissingDataError:
        pass


def test_index_out_of_range():
    try:
        compute_score(MINIMAL_PAYLOAD, species_profile=MINIMAL_PROFILE, use_index=99)
        assert False, "Expected MissingDataError"
    except MissingDataError:
        pass


def test_no_species_profile():
    try:
        compute_score(MINIMAL_PAYLOAD, species_profile=None)
        assert False, "Expected MissingDataError"
    except MissingDataError:
        pass


def test_missing_location_tz():
    payload = _build_payload()
    del payload["location_tz"]
    try:
        compute_score(payload, species_profile=MINIMAL_PROFILE)
        assert False, "Expected MissingDataError"
    except MissingDataError:
        pass


def test_invalid_location_tz():
    payload = _build_payload(location_tz="NotATimezone")
    try:
        compute_score(payload, species_profile=MINIMAL_PROFILE)
        assert False, "Expected MissingDataError"
    except MissingDataError:
        pass


# ---- Tide scoring ----

def test_tide_matching_phase():
    profile = _build_profile(preferred_tide_phase=["high"])
    result = compute_score(MINIMAL_PAYLOAD, species_profile=profile)
    assert result["components"]["tide"]["score_10"] == 10.0


def test_tide_non_matching_phase():
    profile = _build_profile(preferred_tide_phase=["low"])
    result = compute_score(MINIMAL_PAYLOAD, species_profile=profile)
    assert result["components"]["tide"]["score_10"] == 3.0


def test_tide_no_preference():
    profile = _build_profile(preferred_tide_phase=[])
    result = compute_score(MINIMAL_PAYLOAD, species_profile=profile)
    assert result["components"]["tide"]["score_10"] == 10.0


def test_tide_any_token():
    profile = _build_profile(preferred_tide_phase=["any"])
    result = compute_score(MINIMAL_PAYLOAD, species_profile=profile)
    assert result["components"]["tide"]["score_10"] == 10.0


def test_tide_missing_phase_with_preference():
    payload = _build_payload()
    del payload["tide_phase"]
    profile = _build_profile(preferred_tide_phase=["high"])
    try:
        compute_score(payload, species_profile=profile)
        assert False, "Expected MissingDataError"
    except MissingDataError:
        pass


# ---- Wind scoring ----

def test_wind_within_preference():
    profile = _build_profile(preferred_wind_m_s=[0, 10])
    result = compute_score(MINIMAL_PAYLOAD, species_profile=profile)
    assert result["components"]["wind"]["score_10"] == 10.0


def test_wind_outside_tolerance():
    payload = _build_payload(wind_m_s=[50.0])
    profile = _build_profile(preferred_wind_m_s=[0, 10])
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["wind"]["score_10"] == 0.0


def test_wind_no_preference():
    profile = _build_profile(preferred_wind_m_s=[])
    result = compute_score(MINIMAL_PAYLOAD, species_profile=profile)
    assert result["components"]["wind"]["score_10"] == 10.0


def test_wind_missing_with_preference():
    payload = _build_payload()
    del payload["wind_m_s"]
    profile = _build_profile(preferred_wind_m_s=[0, 10])
    try:
        compute_score(payload, species_profile=profile)
        assert False, "Expected MissingDataError"
    except MissingDataError:
        pass


# ---- Wind direction scoring ----

def test_wind_dir_matches():
    payload = _build_payload(wind_direction=[90])
    profile = _build_profile(preferred_wind_directions=["E"])
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["wind_direction"]["score_10"] == 10.0


def test_wind_dir_at_tolerance():
    # 90° (E) preferred, 135° (SE) is 45° away = tolerance
    payload = _build_payload(wind_direction=[135])
    profile = _build_profile(preferred_wind_directions=["E"])
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["wind_direction"]["score_10"] == 10.0


def test_wind_dir_at_midpoint():
    # 90° (E) preferred, 157.5° (SSE) is 67.5° away = 1.5 * tolerance
    payload = _build_payload(wind_direction=[157.5])
    profile = _build_profile(preferred_wind_directions=["E"])
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["wind_direction"]["score_10"] == 5.0


def test_wind_dir_outside_2x_tolerance():
    # 90° (E) preferred, 270° (W) is 180° away >= 2 * tolerance
    payload = _build_payload(wind_direction=[270])
    profile = _build_profile(preferred_wind_directions=["E"])
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["wind_direction"]["score_10"] == 0.0


def test_wind_dir_no_preference():
    payload = _build_payload(wind_direction=[90])
    profile = _build_profile(preferred_wind_directions=[])
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["wind_direction"]["score_10"] == 10.0


def test_wind_dir_missing_data():
    profile = _build_profile(preferred_wind_directions=["E"])
    result = compute_score(MINIMAL_PAYLOAD, species_profile=profile)
    assert result["components"]["wind_direction"]["score_10"] == 10.0


# ---- Wave scoring ----

def test_wave_below_max():
    payload = _build_payload(wave_height_m=[0.5])
    profile = _build_profile(max_wave_height_m=2.0)
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["waves"]["score_10"] > 0.0


def test_wave_at_max():
    payload = _build_payload(wave_height_m=[2.0])
    profile = _build_profile(max_wave_height_m=2.0)
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["waves"]["score_10"] == 0.0


def test_wave_above_max():
    payload = _build_payload(wave_height_m=[3.0])
    profile = _build_profile(max_wave_height_m=2.0)
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["waves"]["score_10"] == 0.0


def test_wave_no_preference():
    profile = _build_profile(max_wave_height_m=[])
    result = compute_score(MINIMAL_PAYLOAD, species_profile=profile)
    assert result["components"]["waves"]["score_10"] == 10.0


def test_wave_with_swell_period():
    payload = _build_payload(wave_height_m=[1.0], swell_period_s=[10.0])
    profile = _build_profile(max_wave_height_m=2.0, preferred_swell_period_s=[8, 12])
    result = compute_score(payload, species_profile=profile)
    score = result["components"]["waves"]["score_10"]
    assert score > 0.0  # both height and period good


def test_wave_missing_with_profile():
    payload = _build_payload()
    del payload["wave_height_m"]
    profile = _build_profile(max_wave_height_m=2.0)
    try:
        compute_score(payload, species_profile=profile)
        assert False, "Expected MissingDataError"
    except MissingDataError:
        pass


# ---- Pressure scoring ----

def test_pressure_rising():
    payload = _build_payload(pressure_hpa=[1013.0, 1017.0])  # delta = 4
    result = compute_score(payload, species_profile=MINIMAL_PROFILE)
    assert result["components"]["pressure"]["score_10"] == 10.0


def test_pressure_falling():
    payload = _build_payload(pressure_hpa=[1013.0, 1009.0])  # delta = -4
    result = compute_score(payload, species_profile=MINIMAL_PROFILE)
    assert result["components"]["pressure"]["score_10"] == 0.0


def test_pressure_stable():
    payload = _build_payload(pressure_hpa=[1013.0, 1013.0])  # delta = 0
    result = compute_score(payload, species_profile=MINIMAL_PROFILE)
    assert result["components"]["pressure"]["score_10"] == 5.0


def test_pressure_partial_rise():
    payload = _build_payload(pressure_hpa=[1013.0, 1014.0])  # delta = 1
    result = compute_score(payload, species_profile=MINIMAL_PROFILE)
    expected = 10.0 * ((1.0 + 2.0) / 4.0)  # 7.5
    assert result["components"]["pressure"]["score_10"] == expected


def test_pressure_no_neighbor():
    payload = _build_payload(pressure_hpa=[1013.0])  # only 1 element, no neighbor
    try:
        compute_score(payload, species_profile=MINIMAL_PROFILE)
        assert False, "Expected MissingDataError"
    except MissingDataError:
        pass


# ---- Season scoring ----

def test_season_in_preferred():
    # May = month 5
    payload = _build_payload(timestamps=["2026-05-15T12:00:00Z"])
    profile = _build_profile(preferred_months=[5, 6, 7])
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["season"]["score_10"] == 10.0


def test_season_outside_preferred():
    payload = _build_payload(timestamps=["2026-05-15T12:00:00Z"])
    profile = _build_profile(preferred_months=[1, 2, 3])
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["season"]["score_10"] == 3.0


def test_season_no_preference():
    payload = _build_payload(timestamps=["2026-05-15T12:00:00Z"])
    profile = _build_profile(preferred_months=[])
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["season"]["score_10"] == 10.0


# ---- Moon scoring ----

def test_moon_matches():
    payload = _build_payload(moon_phase=[0.5])  # full moon
    profile = _build_profile(moon_preference=["full"])
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["moon"]["score_10"] == 10.0


def test_moon_mismatch():
    payload = _build_payload(moon_phase=[0.5])  # full moon
    profile = _build_profile(moon_preference=["new"])
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["moon"]["score_10"] == 4.0


def test_moon_no_preference():
    profile = _build_profile(moon_preference=[])
    result = compute_score(MINIMAL_PAYLOAD, species_profile=profile)
    assert result["components"]["moon"]["score_10"] == 10.0


def test_moon_missing_phase():
    payload = _build_payload()
    del payload["moon_phase"]
    try:
        compute_score(payload, species_profile=MINIMAL_PROFILE)
        assert False, "Expected MissingDataError"
    except MissingDataError:
        pass


# ---- Temperature scoring ----

def test_temp_within_preference():
    payload = _build_payload(temperature_c=[18.0])
    profile = _build_profile(preferred_temp_c=[15, 20], preferred_temp_tol_c=5)
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["temperature"]["score_10"] == 10.0


def test_temp_above_preference():
    # pref=[18,18] (single point), tol=5, pref_min=pref_max=18
    # equal min/max → low=13, high=23, span_low=13, span_high=28
    # temp=24 > high=23 → linear: 10 * (28 - 24) / (28 - 23) = 10 * 4/5 = 8.0
    payload = _build_payload(temperature_c=[24.0])
    profile = _build_profile(preferred_temp_c=[18, 18], preferred_temp_tol_c=5)
    result = compute_score(payload, species_profile=profile)
    score = result["components"]["temperature"]["score_10"]
    assert score == 8.0


def test_temp_below_preference():
    # pref=[18,18], tol=5, low=13, high=23, span_low=8, span_high=28
    # temp=11 < low=13 → linear: 10 * (11 - 8) / (13 - 8) = 10 * 3/5 = 6.0
    payload = _build_payload(temperature_c=[11.0])
    profile = _build_profile(preferred_temp_c=[18, 18], preferred_temp_tol_c=5)
    result = compute_score(payload, species_profile=profile)
    assert result["components"]["temperature"]["score_10"] == 6.0


def test_temp_no_preference():
    profile = _build_profile(preferred_temp_c=[])
    result = compute_score(MINIMAL_PAYLOAD, species_profile=profile)
    assert result["components"]["temperature"]["score_10"] == 10.0


def test_temp_missing_with_profile():
    payload = _build_payload()
    del payload["temperature_c"]
    profile = _build_profile(preferred_temp_c=[15, 20])
    try:
        compute_score(payload, species_profile=profile)
        assert False, "Expected MissingDataError"
    except MissingDataError:
        pass


# ---------------------------------------------------------------------------
# Safety capping
# ---------------------------------------------------------------------------

def test_safety_wind_unsafe_caps_component():
    payload = _build_payload(wind_m_s=[50.0])
    profile = _build_profile(preferred_wind_m_s=[0, 10])
    result = compute_score(payload, species_profile=profile,
                           safety_limits={"max_wind_m_s": 20.0})
    assert result["components"]["wind"].get("safety_capped") is True
    assert result["components"]["wind"]["score_10"] == 0.0


def test_safety_wave_near_limit_caps_component():
    # No wave preference in profile → component score stays 10.0 before safety
    # Safety limit: max_wave_height_m=1.0, wave=1.0 → at limit, val > limit*0.9=0.9
    # → near_limit triggered, caution cap=3.0 applied
    payload = _build_payload(wave_height_m=[1.0])
    profile = _build_profile(max_wave_height_m=[])
    result = compute_score(payload, species_profile=profile,
                           safety_limits={"max_wave_height_m": 1.0})
    waves = result["components"]["waves"]
    assert waves.get("safety_capped") is True
    assert waves["score_10"] == 3.0


def test_safety_wind_unsafe_caps_overall():
    payload = _build_payload(wind_m_s=[50.0])
    profile = _build_profile(preferred_wind_m_s=[0, 10])
    result = compute_score(payload, species_profile=profile,
                           safety_limits={"max_wind_m_s": 20.0})
    assert result["score_100"] <= 30


def test_safety_no_limits_no_capping():
    result = compute_score(MINIMAL_PAYLOAD, species_profile=MINIMAL_PROFILE,
                           safety_limits=None)
    if "safety_capped" in result.get("components", {}).get("wind", {}):
        # Should not be safety capped with no limits
        assert not result["components"]["wind"].get("safety_capped", False)


def test_safety_gust_unsafe():
    payload = _build_payload(wind_m_s=[5.0], wind_max_m_s=[50.0])
    profile = _build_profile(preferred_wind_m_s=[0, 10])
    result = compute_score(payload, species_profile=profile,
                           safety_limits={"max_gust_m_s": 20.0})
    assert result["components"]["wind"].get("safety_capped") is True


def test_safety_swell_unsafe():
    payload = _build_payload(swell_period_s=[2.0])
    profile = _build_profile(max_wave_height_m=[], preferred_swell_period_s=[])
    result = compute_score(payload, species_profile=profile,
                           safety_limits={"min_swell_period_s": 8.0})
    # Safety produces "swell_period<8.0" which now matches "swell_period" prefix
    assert result["components"]["waves"].get("safety_capped") is True


# ---------------------------------------------------------------------------
# Breach detection
# ---------------------------------------------------------------------------

def test_breach_temperature():
    payload = _build_payload(temperature_c=[5.0])
    profile = _build_profile(preferred_temp_c=[15, 20])
    result = compute_score(payload, species_profile=profile)
    assert len(result.get("breaches", [])) > 0
    b = result["breaches"][0]
    assert b["variable"] == "temperature"


def test_breach_wave_exceeding():
    payload = _build_payload(wave_height_m=[3.0])
    profile = _build_profile(max_wave_height_m=2.0)
    result = compute_score(payload, species_profile=profile)
    breaches = [b for b in result.get("breaches", []) if b["variable"] == "wave"]
    assert len(breaches) > 0
    assert breaches[0]["severity"] == "unsafe"


def test_breach_time_outside():
    payload = _build_payload(timestamps=["2026-05-29T03:00:00Z"])
    profile = _build_profile(preferred_times=[{"start_hour": 12, "end_hour": 14}])
    result = compute_score(payload, species_profile=profile)
    breaches = [b for b in result.get("breaches", []) if b["variable"] == "time"]
    assert len(breaches) > 0
    assert breaches[0]["severity"] == "caution"


def test_breach_tide_phase_mismatch():
    profile = _build_profile(preferred_tide_phase=["low"])
    result = compute_score(MINIMAL_PAYLOAD, species_profile=profile)
    breaches = [b for b in result.get("breaches", []) if b["variable"] == "tide_phase"]
    assert len(breaches) > 0


def test_breach_moon_mismatch():
    payload = _build_payload(moon_phase=[0.5])
    profile = _build_profile(moon_preference=["new"])
    result = compute_score(payload, species_profile=profile)
    breaches = [b for b in result.get("breaches", []) if b["variable"] == "moon_phase"]
    assert len(breaches) > 0


def test_no_breaches_when_good():
    result = compute_score(MINIMAL_PAYLOAD, species_profile=MINIMAL_PROFILE)
    assert len(result.get("breaches", [])) == 0


# ---------------------------------------------------------------------------
# Weighted score computation
# ---------------------------------------------------------------------------

def test_default_weights_produce_score():
    result = compute_score(MINIMAL_PAYLOAD, species_profile=MINIMAL_PROFILE)
    assert result["score_10"] > 0.0
    assert result["score_100"] > 0


def test_custom_weights_change_score():
    profile_general = _build_profile(preferred_tide_phase=["low"])
    # Default weights: tide=0.25 → with non-matching tide (3.0), overall drops
    result_default = compute_score(MINIMAL_PAYLOAD, species_profile=profile_general)

    # Tide weight = 0 → tide score doesn't affect overall
    zero_tide_weights = {k: (0.0 if k == "tide" else 1.0) for k in scoring.FACTOR_WEIGHTS}
    result_custom = compute_score(MINIMAL_PAYLOAD, species_profile=profile_general,
                                  factor_weights=zero_tide_weights)
    assert result_custom["score_10"] != result_default["score_10"]


# ---------------------------------------------------------------------------
# 2e. compute_forecast tests
# ---------------------------------------------------------------------------

def test_forecast_single_timestamp():
    results = compute_forecast(MINIMAL_PAYLOAD, species_profile=MINIMAL_PROFILE)
    assert len(results) == 1


def test_forecast_multiple_timestamps():
    payload = _build_payload(
        timestamps=["2026-05-29T12:00:00Z", "2026-05-29T13:00:00Z"],
        wind_m_s=[5.0, 6.0],
        wave_height_m=[1.0, 1.2],
        temperature_c=[18.0, 19.0],
        pressure_hpa=[1013.0, 1015.0, 1017.0],
        moon_phase=[0.5, 0.5],
        tide_phase=["high", "falling"],
    )
    results = compute_forecast(payload, species_profile=MINIMAL_PROFILE)
    assert len(results) == 2


def test_forecast_result_keys():
    results = compute_forecast(MINIMAL_PAYLOAD, species_profile=MINIMAL_PROFILE)
    entry = results[0]
    for key in ("timestamp", "index", "score_10", "score_100", "components",
                "forecast_raw", "profile_used", "safety", "breaches"):
        assert key in entry, f"Missing key: {key}"


def test_forecast_fail_fast():
    payload = _build_payload()
    del payload["moon_phase"]
    try:
        compute_forecast(payload, species_profile=MINIMAL_PROFILE)
        assert False, "Expected exception"
    except Exception:
        pass


def test_forecast_raw_data_assembly():
    """Verify forecast_raw sub-dict contains expected keys."""
    payload = _build_payload(visibility_km=[20.0])
    results = compute_forecast(payload, species_profile=MINIMAL_PROFILE)
    entry = results[0]
    fr = entry.get("forecast_raw", {})
    fw = fr.get("formatted_weather", {})
    # Core weather keys present in forecast_raw.formatted_weather
    for key in ("temperature", "wind", "wave_height_m", "pressure_hpa", "tide_phase"):
        assert key in fw, f"Missing forecast_raw.formatted_weather.{key}"
    # astro_used has moon_phase
    assert "astro_used" in fr
    # score_calc has the full compute_score result
    assert "score_calc" in fr
    assert "score_10" in fr["score_calc"]


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
            print(f"  ERROR: {name} — {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} passed" + (f", {failed} FAILED" if failed else ", all passed"))
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
