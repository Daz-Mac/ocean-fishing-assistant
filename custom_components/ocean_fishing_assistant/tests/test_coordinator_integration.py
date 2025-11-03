import asyncio
import pytest
import time

from custom_components.ocean_fishing_assistant.coordinator import OFACoordinator
from custom_components.ocean_fishing_assistant.data_formatter import DataFormatter

# Simple mock fetcher that returns a strict Open-Meteo-like payload
class MockFetcher:
    def __init__(self, payload):
        self.payload = payload

    async def fetch(self, lat, lon, mode="hourly", days=5):
        # return a copy to simulate network result
        return dict(self.payload)

    async def get_weather_data(self):
        return {"temperature": 12.3, "wind_speed": 3.5, "wind_unit": "m/s"}

# Mock TideProxy result (small predictable tide arrays)
MOCK_TIDE = {
    "timestamps": ["2025-11-03T00:00:00Z", "2025-11-03T01:00:00Z"],
    "tide_height_m": [0.5, 0.6],
    "tide_phase": 0.25,
    "tide_strength": 0.8,
    "next_high": "2025-11-03T06:00:00Z",
    "next_low": "2025-11-03T12:00:00Z",
    "confidence": "astronomical",
    "source": "astronomical_skyfield",
}

class MockTideProxy:
    def __init__(self):
        pass
    async def get_tide_for_timestamps(self, timestamps):
        return MOCK_TIDE

class DummyHass:
    def __init__(self):
        self.data = {}
        class Config:
            units = None
        self.config = Config()
    def async_add_job(self, *args, **kwargs):
        # not used here
        pass

@pytest.mark.asyncio
async def test_coordinator_merges_hourly_and_tide(monkeypatch):
    # sample Open-Meteo hourly used by fetcher
    sample_hourly = {
        "hourly": {
            "time": ["2025-11-03T00:00:00Z", "2025-11-03T01:00:00Z"],
            "temperature_2m": [12.3, 12.0],
            "wind_speed_10m": [3.5, 4.2],
        }
    }
    hass = DummyHass()
    fetcher = MockFetcher(sample_hourly)
    formatter = DataFormatter()

    # Monkeypatch TideProxy in coordinator to use our MockTideProxy
    from custom_components.ocean_fishing_assistant import coordinator as coord_mod
    monkeypatch.setattr(coord_mod, "TideProxy", lambda hass, lat, lon: MockTideProxy())

    # instantiate coordinator (small update_interval)
    c = OFACoordinator(hass, entry_id="t1", fetcher=fetcher, formatter=formatter, lat=1.0, lon=1.0, update_interval=10)
    # call update method directly
    data = await c._async_update_data()

    # Validate returned structure
    assert "forecasts" in data
    assert "raw_payload" in data
    raw = data["raw_payload"]

    # tide merged under raw_payload['tide'] and convenience arrays present
    assert "tide" in raw and raw["tide"]["tide_height_m"] == MOCK_TIDE["tide_height_m"]
    assert raw.get("tide_height_m") == MOCK_TIDE["tide_height_m"]
    # timestamps are propagated
    assert data["timestamps"] == ["2025-11-03T00:00:00Z", "2025-11-03T01:00:00Z"]