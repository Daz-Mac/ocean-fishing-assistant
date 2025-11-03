import json
import pytest

from custom_components.ocean_fishing_assistant.coordinator import OFACoordinator
from custom_components.ocean_fishing_assistant.data_formatter import DataFormatter


class MockFetcherFromFile:
    def __init__(self, payload):
        self.payload = payload

    async def fetch(self, lat, lon, mode="hourly", days=5):
        # return a deep copy-like dict to simulate a network response
        return dict(self.payload)

    async def get_weather_data(self):
        # provide a simple current snapshot (not strictly required)
        return {"temperature": 10.1, "wind_speed": 4.2, "wind_unit": "m/s"}


MOCK_TIDE = {
    "timestamps": ["2025-11-03T00:00:00Z", "2025-11-03T01:00:00Z", "2025-11-03T02:00:00Z"],
    "tide_height_m": [0.4, 0.45, 0.5],
    "tide_phase": 0.2,
    "tide_strength": 0.7,
    "next_high": "2025-11-03T06:00:00Z",
    "next_low": "2025-11-03T12:00:00Z",
    "confidence": "astronomical",
    "source": "astronomical_skyfield",
}


class MockTideProxy:
    async def get_tide_for_timestamps(self, timestamps):
        return MOCK_TIDE


class DummyHass:
    def __init__(self):
        self.data = {}
        class Config:
            units = None
        self.config = Config()


@pytest.mark.asyncio
async def test_full_pipeline_with_recorded_open_meteo(monkeypatch, tmp_path):
    # load recorded sample payload
    samples_path = tmp_path / "samples"
    samples_path.mkdir()
    # copy packaged sample file next to tests for this run
    import os
    base_dir = os.path.dirname(__file__)
    sample_file = os.path.join(base_dir, "sample_open_meteo.json")

    with open(sample_file, "r") as fh:
        payload = json.load(fh)

    # Setup fetcher and formatter
    fetcher = MockFetcherFromFile(payload)
    formatter = DataFormatter()

    # Monkeypatch TideProxy in coordinator to use our mock
    from custom_components.ocean_fishing_assistant import coordinator as coord_mod
    monkeypatch.setattr(coord_mod, "TideProxy", lambda hass, lat, lon: MockTideProxy())

    hass = DummyHass()
    coord = OFACoordinator(hass, entry_id="e2e1", fetcher=fetcher, formatter=formatter, lat=52.0, lon=4.0, update_interval=10)

    data = await coord._async_update_data()

    # Basic sanity checks
    assert isinstance(data, dict)
    assert "forecasts" in data and isinstance(data["forecasts"], list)
    assert "raw_payload" in data and isinstance(data["raw_payload"], dict)

    raw = data["raw_payload"]

    # Ensure hourly -> canonical arrays mapped
    assert raw.get("temperature_c") == [10.1, 9.8, 9.5]
    assert raw.get("wind_m_s") == [4.2, 3.8, 3.6]
    assert raw.get("wind_gust_m_s") == [6.5, 6.0, 5.8]
    assert raw.get("pressure_hpa") == [1015.2, 1014.8, 1014.5]

    # marine fields present
    assert raw.get("wave_height_m") == [1.1, 1.2, 1.3]
    assert raw.get("wave_period_s") == [5.8, 5.6, 5.4]

    # tide info merged
    assert "tide" in raw and raw["tide"]["tide_height_m"] == MOCK_TIDE["tide_height_m"]

    # timestamps propagated to top-level data
    assert data.get("timestamps") == payload.get("hourly", {}).get("time")