"""
Ocean Fishing Assistant - integration entry points.

This module defers heavy/optional imports until runtime in async_setup_entry to avoid
ImportErrors during config flow / integration discovery (e.g. when optional deps like
skyfield are not available at import time).
"""
import logging

from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import aiohttp_client

from .const import DOMAIN, DEFAULT_UPDATE_INTERVAL, DEFAULT_SAFETY_LIMITS

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry):
    """Set up integration from a config entry.

    Heavy imports are deferred here so that Home Assistant can import the package
    (e.g. when running the config flow) without pulling in optional or heavy deps.
    """
    session = aiohttp_client.async_get_clientsession(hass)  # kept in case fetcher needs it

    # ensure a shared in-memory fetch cache exists so multiple entries at similar coords
    # can reuse Open-Meteo responses and reduce API calls
    hass.data.setdefault(DOMAIN, {}).setdefault("fetch_cache", {})

    # require explicit coordinates per the development guide (no fallback to global HA location)
    lat = entry.data.get(CONF_LATITUDE)
    lon = entry.data.get(CONF_LONGITUDE)
    if lat is None or lon is None:
        _LOGGER.error("Config entry missing latitude/longitude; aborting setup")
        return False

    # Defer imports that may raise (optional deps, heavy modules)
    try:
        from .coordinator import OFACoordinator
        from .weather_fetcher import WeatherFetcher
        from .data_formatter import DataFormatter
        # import unit_helpers only when needed (kept for backward compatibility)
        from . import unit_helpers  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive in case optional deps missing
        _LOGGER.exception("Failed to import integration modules: %s", exc)
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
        try:
            await coord.async_load_from_store()
        except Exception:
            _LOGGER.debug("Failed to load persisted fetch for entry %s", entry.entry_id, exc_info=True)

    # request a fresh update (will run after any restored data is available)
    try:
        await coord.async_request_refresh()
    except Exception:
        _LOGGER.exception("Initial data refresh failed for entry %s", entry.entry_id)

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coord

    # forward entry to sensors
    hass.async_create_task(
        hass.config_entries.async_forward_entry_setup(entry, "sensor")
    )

    return True


async def async_unload_entry(hass, entry):
    """Unload a config entry."""
    # forward unload to platform(s)
    try:
        unload_ok = await hass.config_entries.async_forward_entry_unload(entry, "sensor")
    except Exception:
        _LOGGER.exception("Error while forwarding unload for entry %s", entry.entry_id)
        unload_ok = False

    # cleanup coordinator from hass.data
    try:
        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
    except Exception:
        _LOGGER.debug("Failed to remove entry %s from hass.data", entry.entry_id, exc_info=True)

    return unload_ok