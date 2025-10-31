import pytest
from custom_components.ocean_fishing_assistant.ocean_scoring import compute_score, MissingDataError

def make_base_data(temp=12.0, wind=4.0, wave=0.5, tide=0.5):
    return {
        "timestamps": ["2025-01-01T06:00:00Z"],
        "temperature_c": [temp],
        "wind_m_s": [wind],
        "wave_height_m": [wave],
        "tide_height_m": [tide],
        "pressure_hpa": [1013.0, 1014.0],
        "tide_phase": 0.0
    }

def test_score_includes_tide_component():
    data = make_base_data(tide=0.5)
    r = compute_score(data, species_profile="sea_bass")
    assert "components" in r
    assert "tide" in r["components"]
    assert 0 <= r["score_100"] <= 100

def test_missing_data_raises():
    with pytest.raises(MissingDataError):
        compute_score({}, None)