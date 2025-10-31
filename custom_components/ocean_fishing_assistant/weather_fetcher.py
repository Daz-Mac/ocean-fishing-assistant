import aiohttp
import asyncio
from typing import Dict, Any

from .const import OM_BASE

class OpenMeteoClient:
    """
    Minimal internal Open-Meteo client that requests specific fields and
    returns canonical SI units:
      - temperature: degrees Celsius (°C)
      - wind speed: meters per second (m/s)
      - wave height: meters (m)
      - pressure: hectopascal (hPa)
    """

    def __init__(self, session: aiohttp.ClientSession):
        self._session = session

    async def fetch(self, lat: float, lon: float, mode: str = "hourly") -> Dict[str, Any]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": "UTC",
        }

        # fields we request — Open-Meteo names; wave data may be in a different service
        if mode == "hourly":
            params["hourly"] = ",".join(
                ["temperature_2m", "windspeed_10m", "pressure_msl", "wave_height"]
            )
        else:
            params["daily"] = ",".join(
                ["temperature_2m_max", "temperature_2m_min", "windspeed_10m_max", "pressure_msl"]
            )

        async with self._session.get(OM_BASE, params=params, timeout=30) as resp:
            resp.raise_for_status()
            data = await resp.json()

        # Normalize to a minimal canonical structure expected downstream.
        return self._normalize_to_si(data, mode)

    def _normalize_to_si(self, om_response: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """
        Open-Meteo generally returns temperature in C and windspeed in m/s when asked.
        This function extracts keys and maps them to our canonical keys.
        """
        out = {"raw": om_response, "mode": mode}
        if mode == "hourly":
            hourly = om_response.get("hourly", {})
            out["timestamps"] = hourly.get("time", [])
            out["temperature_c"] = hourly.get("temperature_2m")
            out["wind_m_s"] = hourly.get("windspeed_10m")
            out["pressure_hpa"] = hourly.get("pressure_msl")
            # wave_height may not be present; if present, expect meters
            out["wave_height_m"] = hourly.get("wave_height")
        else:
            daily = om_response.get("daily", {})
            out["timestamps"] = daily.get("time", [])
            out["temperature_max_c"] = daily.get("temperature_2m_max")
            out["temperature_min_c"] = daily.get("temperature_2m_min")
            out["wind_max_m_s"] = daily.get("windspeed_10m_max")
            out["pressure_hpa"] = daily.get("pressure_msl")
            out["wave_height_m"] = daily.get("wave_height")
        return out