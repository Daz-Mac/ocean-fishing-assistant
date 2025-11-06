"""
Ocean Fishing Assistant - integration entry points.

This module defers heavy/optional imports until runtime in async_setup_entry to avoid
ImportErrors during config flow / integration discovery (e.g. when optional deps like
skyfield are not available at import time).
"""
import logging
import inspect

from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import aiohttp_client

from .const import DOMAIN, DEFAULT_UPDATE_INTERVAL, DEFAULT_SAFETY_LIMITS

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry):
    """Set up integration from a config entry.

    Heavy imports are deferred here so that Home Assistant can import the package
    (e.g. when running the config flow) without pulling in optional or heavy deps.
    """
    _LOGGER.debug("Starting async_setup_entry for entry %s", entry.entry_id)
    session = aiohttp_client.async_get_clientsession(hass)  # kept in case fetcher needs it
    _LOGGER.debug("Acquired aiohttp session: %s", session)

    # ensure a shared in-memory fetch cache exists so multiple entries at similar coords
    # can reuse Open-Meteo responses and reduce API calls
    fetch_cache = hass.data.setdefault(DOMAIN, {}).setdefault("fetch_cache", {})
    _LOGGER.debug("Fetch cache initialized for domain %s (current keys: %s)", DOMAIN, list(fetch_cache.keys())[:10])

    # require explicit coordinates per the development guide (no fallback to global HA location)
    lat = entry.data.get(CONF_LATITUDE)
    lon = entry.data.get(CONF_LONGITUDE)
    _LOGGER.debug("Config entry %s coordinates lat=%s lon=%s", entry.entry_id, lat, lon)
    if lat is None or lon is None:
        _LOGGER.error("Config entry missing latitude/longitude; aborting setup for entry %s", entry.entry_id)
        return False

    # Defer imports that may raise (optional deps, heavy modules)
    try:
        from .coordinator import OFACoordinator
        from .weather_fetcher import WeatherFetcher
        from .data_formatter import DataFormatter
        # import unit_helpers only when needed (kept for backward compatibility)
        from . import unit_helpers  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive in case optional deps missing
        _LOGGER.exception("Failed to import integration modules for entry %s: %s", entry.entry_id, exc)
        return False

    # Log information about imported modules to help debugging which code was loaded
    try:
        _LOGGER.debug(
            "Imported integration classes: OFACoordinator=%s (module=%s), WeatherFetcher=%s (module=%s), DataFormatter=%s (module=%s)",
            getattr(OFACoordinator, "__name__", repr(OFACoordinator)),
            getattr(OFACoordinator, "__module__", None),
            getattr(WeatherFetcher, "__name__", repr(WeatherFetcher)),
            getattr(WeatherFetcher, "__module__", None),
            getattr(DataFormatter, "__name__", repr(DataFormatter)),
            getattr(DataFormatter, "__module__", None),
        )
        # attempt to log file locations where possible (best-effort)
        _LOGGER.debug(
            "Module files: coordinator=%s, weather_fetcher=%s, data_formatter=%s",
            getattr(inspect.getmodule(OFACoordinator), "__file__", None) if inspect.getmodule(OFACoordinator) else None,
            getattr(inspect.getmodule(WeatherFetcher), "__file__", None) if inspect.getmodule(WeatherFetcher) else None,
            getattr(inspect.getmodule(DataFormatter), "__file__", None) if inspect.getmodule(DataFormatter) else None,
        )
    except Exception:
        _LOGGER.debug("Failed to introspect imported module file locations", exc_info=True)

    # Instantiate formatter; fetcher will be constructed after options are read
    formatter = DataFormatter()
    _LOGGER.debug("DataFormatter instantiated: %s", formatter)

    # Read canonical safety_limits already normalized by config flow; fall back to defaults
    units = entry.options.get("units", "metric") or "metric"
    safety_limits = entry.options.get("safety_limits") or {}
    _LOGGER.debug("Entry options for entry %s: units=%s, raw_safety_limits=%s", entry.entry_id, units, entry.options.get("safety_limits"))
    if safety_limits:
        for k, v in DEFAULT_SAFETY_LIMITS.items():
            safety_limits.setdefault(k, v)
        _LOGGER.debug("Merged provided safety_limits with defaults: %s", safety_limits)
    else:
        safety_limits = DEFAULT_SAFETY_LIMITS.copy()
        _LOGGER.debug("No safety_limits provided; using DEFAULT_SAFETY_LIMITS for entry %s: %s", entry.entry_id, safety_limits)

    # Determine wind unit preference: prefer explicit per-entry option, else follow units
    speed_unit = entry.options.get("wind_unit") or ("km/h" if units == "metric" else "mph")
    _LOGGER.debug("Chosen speed_unit for entry %s: %s", entry.entry_id, speed_unit)
    fetcher = WeatherFetcher(hass, lat, lon, speed_unit=speed_unit)
    _LOGGER.debug("WeatherFetcher instantiated: %s", fetcher)

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
    _LOGGER.debug("OFACoordinator created for entry %s: %s", entry.entry_id, coord)

    # try to load persisted last successful fetch before first refresh (fast recovery)
    if entry.options.get("persist_last_fetch", False):
        _LOGGER.debug("persist_last_fetch enabled for entry %s; attempting to load persisted data", entry.entry_id)
        try:
            await coord.async_load_from_store()
            _LOGGER.debug("Successfully loaded persisted fetch for entry %s", entry.entry_id)
        except Exception:
            _LOGGER.debug("Failed to load persisted fetch for entry %s", entry.entry_id, exc_info=True)
    else:
        _LOGGER.debug("persist_last_fetch not enabled for entry %s", entry.entry_id)

    # request a fresh update (will run after any restored data is available)
    try:
        _LOGGER.debug("Requesting initial data refresh for entry %s", entry.entry_id)
        await coord.async_request_refresh()
        _LOGGER.debug("Initial data refresh completed for entry %s", entry.entry_id)
    except Exception:
        _LOGGER.exception("Initial data refresh failed for entry %s", entry.entry_id)

    # store coordinator in hass.data for lookups by entry_id
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coord
    _LOGGER.debug("Stored coordinator in hass.data[%s][%s]", DOMAIN, entry.entry_id)
    try:
        current_fetch_cache_keys = list(hass.data.setdefault(DOMAIN, {}).get("fetch_cache", {}).keys())
        _LOGGER.debug("Post-setup fetch_cache keys (sample up to 10): %s", current_fetch_cache_keys[:10])
    except Exception:
        _LOGGER.debug("Unable to read fetch_cache keys after setup", exc_info=True)

    # forward entry to sensors (use plural API that accepts a list of platforms)
    try:
        _LOGGER.debug("Forwarding entry %s to sensor platform(s)", entry.entry_id)
        hass.async_create_task(
            hass.config_entries.async_forward_entry_setups(entry, ["sensor"])
        )
    except Exception:
        _LOGGER.exception("Failed to forward entry %s to platforms", entry.entry_id)

    _LOGGER.debug("async_setup_entry completed successfully for entry %s", entry.entry_id)
    return True


async def async_unload_entry(hass, entry):
    """Unload a config entry."""
    _LOGGER.debug("Starting async_unload_entry for entry %s", entry.entry_id)
    # forward unload to platform(s)
    try:
        unload_ok = await hass.config_entries.async_forward_entry_unload(entry, "sensor")
        _LOGGER.debug("Forwarded unload for entry %s to sensor platform, result=%s", entry.entry_id, unload_ok)
    except Exception:
        _LOGGER.exception("Error while forwarding unload for entry %s", entry.entry_id)
        unload_ok = False

    # cleanup coordinator from hass.data
    try:
        removed = hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
        _LOGGER.debug("Removed coordinator from hass.data for entry %s: %s", entry.entry_id, removed)
    except Exception:
        _LOGGER.debug("Failed to remove entry %s from hass.data", entry.entry_id, exc_info=True)

    _LOGGER.debug("async_unload_entry finished for entry %s, unload_ok=%s", entry.entry_id, unload_ok)
    return unload_ok