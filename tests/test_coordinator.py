import pytest
from custom_components.ocean_fishing_assistant.data_formatter import DataFormatter
from custom_components.ocean_fishing_assistant.coordinator import OFACoordinator
from types import SimpleNamespace

@pytest.mark.asyncio
async def test_store_load(tmp_path, hass, aiohttp_server):
    # Create a fake fetcher which returns a valid structure
    fetcher = SimpleNamespace()
    async def fake_fetch(lat, lon, mode="hourly"):
        return {"timestamps": ["t"], "temperature_c": [10.0]}
    fetcher.fetch = fake_fetch

    formatter = DataFormatter()
    coord = OFACoordinator(hass, fetcher, formatter, lat=0.0, lon=0.0, update_interval=60, store_enabled=False)
    data = await coord._async_update_data()
    assert "timestamps" in data