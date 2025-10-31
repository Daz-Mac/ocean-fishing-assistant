import asyncio
from custom_components.ocean_fishing_assistant.weather_fetcher import OpenMeteoClient
import aiohttp
import pytest

@pytest.mark.asyncio
async def test_fetcher_returns_structure(aiohttp_unused_port, aiohttp_server):
    # Spin up a fake OM server to simulate minimal response
    async def handler(request):
        return aiohttp.web.json_response({
            "hourly": {
                "time": ["2025-01-01T00:00Z"],
                "temperature_2m": [10.0],
                "windspeed_10m": [5.0],
                "pressure_msl": [1013.0],
                "wave_height": [0.5]
            }
        })

    from aiohttp import web
    app = web.Application()
    app.router.add_get("/v1/forecast", handler)
    server = await aiohttp_server(app)
    port = server.port
    session = aiohttp.ClientSession()
    client = OpenMeteoClient(session)
    data = await client.fetch(lat=0.0, lon=0.0, mode="hourly")
    await session.close()

    assert "timestamps" in data or "time" in data
    assert data.get("temperature_c") is not None or data.get("temperature_2m") is not None