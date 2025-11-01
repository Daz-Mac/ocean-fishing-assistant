import asyncio
import logging

from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.storage import Store

from .const import DOMAIN, DEFAULT_UPDATE_INTERVAL, STORE_KEY, STORE_VERSION
from .coordinator import OFACoordinator
from .weather_fetcher import OpenMeteoClient
from .data_formatter import DataFormatter

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass, entry):
    """Set up integration from a config entry."""
    session = aiohttp_client.async_get_clientsession(hass)
    client = OpenMeteoClient(session)
    formatter = DataFormatter()
    # require explicit coordinates per the development guide (no fallback to global HA location)
    lat = entry.data.get(CONF_LATITUDE)
    lon = entry.data.get(CONF_LONGITUDE)
    if lat is None or lon is None:
        _LOGGER.error("Config entry missing latitude/longitude; aborting setup")
        return False

    coord = OFACoordinator(
        hass,
        entry.entry_id,
        fetcher=client,
        formatter=formatter,
        lat=lat,
        lon=lon,
        update_interval=entry.options.get("update_interval", DEFAULT_UPDATE_INTERVAL),
        store_enabled=entry.options.get("persist_last_fetch", False),
        ttl=entry.options.get("persist_ttl", 3600),
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
    await hass.config_entries.async_forward_entry_unload(entry, "sensor")
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True