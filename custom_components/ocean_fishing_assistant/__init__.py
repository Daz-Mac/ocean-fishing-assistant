import asyncio
import logging

from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.storage import Store

from .const import DOMAIN, DEFAULT_UPDATE_INTERVAL, STORE_KEY, STORE_VERSION, DEFAULT_SAFETY_LIMITS
from .coordinator import OFACoordinator
from .weather_fetcher import OpenMeteoClient
from .data_formatter import DataFormatter
from . import unit_helpers

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass, entry):
    """Set up integration from a config entry."""
    session = aiohttp_client.async_get_clientsession(hass)
    # ensure a shared in-memory fetch cache exists so multiple entries at the coords
    # can reuse Open-Meteo responses and reduce API calls
    hass.data.setdefault(DOMAIN, {}).setdefault("fetch_cache", {})
    
    client = OpenMeteoClient(session)
    formatter = DataFormatter()
    # require explicit coordinates per the development guide (no fallback to global HA location)
    lat = entry.data.get(CONF_LATITUDE)
    lon = entry.data.get(CONF_LONGITUDE)
    if lat is None or lon is None:
        _LOGGER.error("Config entry missing latitude/longitude; aborting setup")
        return False

    # Normalize and validate per-entry safety limits (convert to canonical metric internally)
    raw_safety = entry.options.get("safety_limits") or {}
    units = entry.options.get("units", "metric") or "metric"

    safety_limits = {}
    try:
        # Wind: user input is km/h (metric) or mph (imperial) per UX decision
        raw_wind = raw_safety.get("max_wind")
        if raw_wind is not None:
            if units == "metric":
                safety_limits["max_wind_m_s"] = unit_helpers.kmh_to_m_s(raw_wind)
            else:
                safety_limits["max_wind_m_s"] = unit_helpers.mph_to_m_s(raw_wind)
        # Wave height: meters (metric) or feet (imperial)
        raw_wave = raw_safety.get("max_wave_height")
        if raw_wave is not None:
            if units == "metric":
                safety_limits["max_wave_height_m"] = float(raw_wave)
            else:
                safety_limits["max_wave_height_m"] = unit_helpers.ft_to_m(raw_wave)
        # Visibility: km (metric) or miles (imperial)
        raw_vis = raw_safety.get("min_visibility")
        if raw_vis is not None:
            if units == "metric":
                safety_limits["min_visibility_km"] = float(raw_vis)
            else:
                safety_limits["min_visibility_km"] = unit_helpers.miles_to_km(raw_vis)
        # Swell period: seconds (user-facing units are seconds)
        raw_period = raw_safety.get("max_swell_period")
        if raw_period is not None:
            safety_limits["max_swell_period_s"] = float(raw_period)
    except Exception:
        _LOGGER.debug("Failed to normalize safety limits; falling back to defaults", exc_info=True)

    # If after normalization we have no valid limits, fall back to conservative defaults
    if not safety_limits:
        safety_limits = DEFAULT_SAFETY_LIMITS.copy()
        _LOGGER.debug("Using DEFAULT_SAFETY_LIMITS for entry %s", entry.entry_id)

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
    await hass.config_entries.async_forward_entry_unload(entry, "sensor")
    hass.data[DOMAIN].pop(entry.entry_id, None)
    return True