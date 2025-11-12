"""
Ocean Fishing Assistant - integration entry points (strict).
"""
import logging

from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import aiohttp_client

from .const import DOMAIN, DEFAULT_UPDATE_INTERVAL, DEFAULT_SAFETY_LIMITS

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, entry):
    """Set up integration from a config entry (strict)."""

    _LOGGER.debug("Starting async_setup_entry for entry %s", entry.entry_id)
    session = aiohttp_client.async_get_clientsession(hass)
    _LOGGER.debug("Acquired aiohttp session: %s", session)

    fetch_cache = hass.data.setdefault(DOMAIN, {}).setdefault("fetch_cache", {})
    _LOGGER.debug(
        "Fetch cache initialized for domain %s (current keys: %s)", DOMAIN, list(fetch_cache.keys())[:10]
    )

    lat = entry.data.get(CONF_LATITUDE)
    lon = entry.data.get(CONF_LONGITUDE)
    _LOGGER.debug("Config entry %s coordinates lat=%s lon=%s", entry.entry_id, lat, lon)
    if lat is None or lon is None:
        _LOGGER.error(
            "Config entry missing latitude/longitude; aborting setup for entry %s", entry.entry_id
        )
        return False

    try:
        from .coordinator import OFACoordinator
        from .weather_fetcher import WeatherFetcher
        from .data_formatter import DataFormatter
        from .species_loader import SpeciesLoader
    except Exception as exc:
        _LOGGER.exception(
            "Failed to import integration modules for entry %s: %s", entry.entry_id, exc
        )
        return False

    # Strictly validate packaged species profiles at startup; fail fast on schema problems
    try:
        loader = SpeciesLoader(hass)
        await loader.async_load_profiles()
        _LOGGER.debug(
            "Loaded and validated species profiles (count=%d)", len(loader.get_all_species())
        )
    except Exception as exc:
        _LOGGER.exception("species_profiles.json failed validation: %s", exc)
        # fail setup loudly (per project policy)
        return False

    formatter = DataFormatter()
    _LOGGER.debug("DataFormatter instantiated for entry %s", entry.entry_id)

    # Read canonical safety_limits already normalized by config flow; require units present
    units = entry.options.get("units")
    if not units:
        raise ValueError("Entry options missing 'units' (strict)")

    safety_limits = entry.options.get("safety_limits")
    if safety_limits is None:
        raise ValueError("Entry options missing 'safety_limits' (strict)")

    # Determine expected wind unit deterministically from the selected units
    expected_wind_unit = "km/h" if units == "metric" else "mph" if units == "imperial" else None
    if expected_wind_unit is None:
        raise ValueError(f"Unsupported entry.options['units']: {units!r} (strict)")

    # If wind_unit is missing or mismatched, migrate the entry options to the deterministic mapping.
    current_wind_unit = entry.options.get("wind_unit")
    if current_wind_unit != expected_wind_unit:
        _LOGGER.info(
            "Config entry %s wind_unit mismatch or missing (current=%s expected=%s). Migrating entry.options to deterministic wind unit.",
            entry.entry_id,
            current_wind_unit,
            expected_wind_unit,
        )
        # Build new options dict based on existing options but with corrected wind_unit
        new_options = dict(entry.options or {})
        new_options["wind_unit"] = expected_wind_unit
        # Persist update to the config entry so future setups use correct mapping
        hass.config_entries.async_update_entry(entry, options=new_options)
        # Use updated value going forward
        wind_unit = expected_wind_unit
    else:
        wind_unit = current_wind_unit

    # Final validation of wind_unit value
    if wind_unit not in ("km/h", "mph", "m/s"):
        raise ValueError(f"Invalid entry.options['wind_unit']: {wind_unit!r} (strict)")

    _LOGGER.debug("Chosen speed_unit for entry %s: %s", entry.entry_id, wind_unit)

    # Validate selected species (if set) exists in the packaged profiles
    selected_species = entry.options.get("species")
    selected_region = entry.options.get("species_region")
    if selected_species:
        try:
            # Synthetic selection pattern: "general_mixed_<region>"
            if isinstance(selected_species, str) and selected_species.startswith("general_mixed_"):
                region = selected_species.replace("general_mixed_", "")
                region_info = loader.get_region_info(region)
                if not region_info:
                    raise ValueError(
                        f"Selected synthetic region '{region}' not found in species_profiles.json (strict)"
                    )
                if region_info.get("habitat") != "ocean":
                    raise ValueError(
                        f"Selected synthetic region '{region}' is not an ocean region (strict)"
                    )
                # ok: synthetic general_mixed_<region> is valid
            elif selected_species == "general_mixed":
                # require explicit region to be present and valid
                if not selected_region:
                    raise ValueError(
                        "Selected species 'general_mixed' requires 'species_region' option to be set (strict)"
                    )
                region_info = loader.get_region_info(selected_region)
                if not region_info:
                    raise ValueError(
                        f"Selected species_region '{selected_region}' not found in species_profiles.json (strict)"
                    )
                if region_info.get("habitat") != "ocean":
                    raise ValueError(
                        f"Selected species_region '{selected_region}' is not an ocean region (strict)"
                    )
            else:
                # Validate a specific species id exists and is ocean habitat
                sp = loader.get_species(selected_species)
                if not sp:
                    raise ValueError(
                        f"Selected species '{selected_species}' not found in packaged species_profiles.json (strict)"
                    )
                if sp.get("habitat") != "ocean":
                    raise ValueError(
                        f"Selected species '{selected_species}' is not an ocean species (strict)"
                    )
        except Exception as exc:
            _LOGGER.exception("Species validation failed for entry %s: %s", entry.entry_id, exc)
            return False

    fetcher = WeatherFetcher(hass, lat, lon, speed_unit=wind_unit)
    _LOGGER.debug("WeatherFetcher instantiated for entry %s", entry.entry_id)

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
    _LOGGER.debug("OFACoordinator created for entry %s", entry.entry_id)

    # try to load persisted last successful fetch before first refresh (fast recovery)
    if entry.options.get("persist_last_fetch", False):
        _LOGGER.debug(
            "persist_last_fetch enabled for entry %s; attempting to load persisted data", entry.entry_id
        )
        try:
            loaded = await coord.async_load_from_store()
            if loaded:
                _LOGGER.debug("Successfully loaded persisted fetch for entry %s", entry.entry_id)
            else:
                _LOGGER.debug("No persisted fetch available for entry %s (first run)", entry.entry_id)
        except Exception:
            _LOGGER.exception("Failed to load persisted fetch for entry %s", entry.entry_id)
            # Treat failure to restore as non-fatal but log the stack.

    # request a fresh update (will run after any restored data is available)
    _LOGGER.debug("Requesting initial data refresh for entry %s", entry.entry_id)
    await coord.async_request_refresh()
    _LOGGER.debug("Initial data refresh requested for entry %s", entry.entry_id)

    # store coordinator in hass.data for lookups by entry_id
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coord
    _LOGGER.debug("Stored coordinator in hass.data[%s][%s]", DOMAIN, entry.entry_id)

    # forward entry to sensors
    hass.async_create_task(hass.config_entries.async_forward_entry_setups(entry, ["sensor"]))

    _LOGGER.debug("async_setup_entry completed for entry %s", entry.entry_id)
    return True


async def async_unload_entry(hass, entry):
    """Unload a config entry."""
    _LOGGER.debug("Starting async_unload_entry for entry %s", entry.entry_id)
    try:
        unload_ok = await hass.config_entries.async_forward_entry_unload(entry, "sensor")
        _LOGGER.debug(
            "Forwarded unload for entry %s to sensor platform, result=%s", entry.entry_id, unload_ok
        )
    except Exception:
        _LOGGER.exception("Error while forwarding unload for entry %s", entry.entry_id)
        unload_ok = False

    try:
        removed = hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
        _LOGGER.debug("Removed coordinator from hass.data for entry %s: %s", entry.entry_id, removed)
    except Exception:
        _LOGGER.exception("Failed to remove entry %s from hass.data", entry.entry_id)

    _LOGGER.debug(
        "async_unload_entry finished for entry %s, unload_ok=%s", entry.entry_id, unload_ok
    )
    return unload_ok