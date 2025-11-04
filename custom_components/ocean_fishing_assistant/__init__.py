import asyncio
import logging

from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.storage import Store

from .const import DOMAIN, DEFAULT_UPDATE_INTERVAL, STORE_KEY, STORE_VERSION, DEFAULT_SAFETY_LIMITS
from .coordinator import OFACoordinator
from .weather_fetcher import WeatherFetcher
from .data_formatter import DataFormatter
from . import unit_helpers

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass, entry):
    """Set up integration from a config entry."""
    session = aiohttp_client.async_get_clientsession(hass)
    # ensure a shared in-memory fetch cache exists so multiple entries at the coords
    # can reuse Open-Meteo responses and reduce API calls
    hass.data.setdefault(DOMAIN, {}).setdefault("fetch_cache", {})
    
    # require explicit coordinates per the development guide (no fallback to global HA location)
    lat = entry.data.get(CONF_LATITUDE)
    lon = entry.data.get(CONF_LONGITUDE)
    if lat is None or lon is None:
        _LOGGER.error("Config entry missing latitude/longitude; aborting setup")
        return False

    # Instantiate formatter; fetcher will be constructed after options are read
    formatter = DataFormatter()

    # Read canonical safety_limits already normalized by config flow; fall back to defaults
    units = entry.options.get("units", "metric") or "metric"
    safety_limits = entry.options.get("safety_limits") or {}
    if safety_limits:
        for k, v in DEFAULT_SAFETY_LIMITS.items():
            safety_limits.setdefault(k, v)
    else:
        safety_limits = DEFAULT_SAFETY_LIMITS.copy()
        _LOGGER.debug("Using DEFAULT_SAFETY_LIMITS for entry %s", entry.entry_id)

    # Determine wind unit preference: prefer explicit per-entry option, else follow units
    speed_unit = entry.options.get("wind_unit") or ("km/h" if units == "metric" else "mph")
    fetcher = WeatherFetcher(hass, lat, lon, speed_unit=speed_unit)

    coord = OFACoordinator(
        hass,
        entry.entry_id,
        fetcher=fetcher,
        formatter=formatter,
        lat=lat,
        lon=lon,
        update_interval=entry.options.get("update_interval", DEFAULT_UPDATE_INTERVAL),
        store_enabled=entry.options.get("persist_last_fetch", False),
        ttl=entry.options.get("persist_ttl", 3600),
        species=entry.options.get("species"),
        units=units,
        safety_limits=safety_limits,
    )
    # try to load persisted last successful fetch before first refresh (fast recovery)
    if entry.options.get("persist_last_fetch", False):
        await coord.async_load_from_store()

    # request a fresh update (will run after any restored data is available)
    await coord.async_request_refresh()
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coord

    # forward entry to sensors
    hass.async_create_task(
        hass.config_entries.async_forward_entry_setup(entry, "sensor")
    )
    return True



async def async_unload_entry(hass, entry):
    """Unload a config entry."""
    # forward unload to platform
    unload_ok = await hass.config_entries.async_forward_entry_unload(entry, "sensor")
    # cleanup coordinator from hass.data
    try:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    except Exception:
        pass
    return unload_ok