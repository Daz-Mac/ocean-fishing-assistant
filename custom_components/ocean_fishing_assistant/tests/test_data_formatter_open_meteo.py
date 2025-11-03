import pytest
from custom_components.ocean_fishing_assistant.data_formatter import DataFormatter

SAMPLE_OM_PAYLOAD = {
    "hourly": {
        "time": ["2025-11-03T00:00:00Z", "2025-11-03T01:00:00Z"],
        "temperature_2m": [12.3, 12.0],
        "wind_speed_10m": [3.5, 4.2],
        "windgusts_10m": [5.0, 6.0],
        "pressure_msl": [1012.0, 1011.5],
        "cloudcover": [20, 30],
        "precipitation_probability": [0, 5],
        # marine keys (example)
        "wave_height": [1.2, 1.4],
        "wave_period": [6.0, 5.8],
    }
}

def test_open_meteo_hourly_to_canonical_arrays():
    df = DataFormatter()
    # prepare payload shaped like fetcher.fetch output when hourly present
    payload = {"hourly": SAMPLE_OM_PAYLOAD["hourly"]}
    out = df.validate(payload, species_profile=None, units="metric", safety_limits={})

    # Check timestamps
    assert out["timestamps"] == ["2025-11-03T00:00:00Z", "2025-11-03T01:00:00Z"]

    # Check raw_payload contains canonical arrays
    raw = out["raw_payload"]
    assert "temperature_c" in raw
    assert raw["temperature_c"] == [12.3, 12.0]

    assert "wind_m_s" in raw
    assert raw["wind_m_s"] == [3.5, 4.2]

    assert "wind_gust_m_s" in raw
    assert raw["wind_gust_m_s"] == [5.0, 6.0]

    assert "pressure_hpa" in raw
    assert raw["pressure_hpa"] == [1012.0, 1011.5]

    assert "cloud_cover" in raw and raw["cloud_cover"] == [20, 30]
    assert "precipitation_probability" in raw and raw["precipitation_probability"] == [0, 5]

    # marine fields mapped
    assert "wave_height_m" in raw and raw["wave_height_m"] == [1.2, 1.4]
    assert "wave_period_s" in raw and raw["wave_period_s"] == [6.0, 5.8]