"""
Ocean Fishing Assistant - integration entry points (strict, ocean-only).

This implementation expects required values (including `units` and
`safety_limits`) to be present in the created config entry `data`.
If they are missing or invalid, setup fails loudly (ValueError / return False).
"""
import logging

from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import aiohttp_client

from .const import DOMAIN, DEFAULT_UPDATE_INTERVAL, DEFAULT_SAFETY_LIMITS, CONF_SPECIES_ID, CONF_SPECIES_REGION, CONF_THRESHOLDS, CONF_TIME_PERIODS

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

    # Coordinates come from entry.data (flow writes these into data)
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

    # --- Read required canonical values from entry.data (strict: no migrations) ---
    units = entry.data.get("units")
    if not units:
        _LOGGER.error("Config entry %s missing required 'units' in entry.data (strict)", entry.entry_id)
        raise ValueError("Entry data missing 'units' (strict)")

    safety_limits = entry.data.get("safety_limits")
    if safety_limits is None:
        _LOGGER.error("Config entry %s missing required 'safety_limits' in entry.data (strict)", entry.entry_id)
        raise ValueError("Entry data missing 'safety_limits' (strict)")

    # Deterministic wind unit mapping — ensure the flow stored wind_unit consistent with units
    expected_wind_unit = "km/h" if units == "metric" else "mph" if units == "imperial" else None
    if expected_wind_unit is None:
        _LOGGER.error("Config entry %s has unsupported units=%r (strict)", entry.entry_id, units)
        raise ValueError(f"Unsupported entry.data['units']: {units!r} (strict)")

    wind_unit = entry.data.get("wind_unit")
    if wind_unit != expected_wind_unit:
        _LOGGER.error(
            "Config entry %s wind_unit mismatch or missing (found=%r expected=%r) — entry must be created with correct wind_unit (strict)",
            entry.entry_id,
            wind_unit,
            expected_wind_unit,
        )
        raise ValueError("Entry data 'wind_unit' missing or mismatched (strict)")

    # Validate allowed wind unit values
    if wind_unit not in ("km/h", "mph", "m/s"):
        _LOGGER.error("Config entry %s has invalid wind_unit=%r (strict)", entry.entry_id, wind_unit)
        raise ValueError(f"Invalid entry.data['wind_unit']: {wind_unit!r} (strict)")

    _LOGGER.debug("Chosen speed_unit for entry %s: %s", entry.entry_id, wind_unit)

    # Validate selected species (if set) exists in the packaged profiles
    selected_species = entry.data.get(CONF_SPECIES_ID)
    selected_region = entry.data.get(CONF_SPECIES_REGION)

    # Resolve species into either:
    #  - a concrete species profile dict (preferred), or
    #  - a synthetic string ("general_mixed" or "general_mixed_<region>") which the integration will treat specially.
    resolved_species = None
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
                # Keep the synthetic string (formatter/scoring can handle it as a string marker)
                resolved_species = selected_species
            elif selected_species == "general_mixed":
                # require explicit region to be present and valid
                if not selected_region:
                    raise ValueError(
                        "Selected species 'general_mixed' requires 'species_region' in entry.data (strict)"
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
                # keep the 'general_mixed' marker
                resolved_species = "general_mixed"
            else:
                # Load concrete species profile dict and pass that into coordinator/formatter so scoring receives a dict
                sp = loader.get_species(selected_species)
                if not sp:
                    raise ValueError(
                        f"Selected species '{selected_species}' not found in packaged species_profiles.json (strict)"
                    )
                if sp.get("habitat") != "ocean":
                    raise ValueError(
                        f"Selected species '{selected_species}' is not an ocean species (strict)"
                    )
                resolved_species = sp
        except Exception as exc:
            _LOGGER.exception("Species validation failed for entry %s: %s", entry.entry_id, exc)
            return False

    # Create WeatherFetcher and coordinator using values from entry.data
    fetcher = WeatherFetcher(hass, lat, lon, speed_unit=wind_unit)
    _LOGGER.debug("WeatherFetcher instantiated for entry %s", entry.entry_id)

    time_periods_mode = entry.data.get(CONF_TIME_PERIODS, "full_day")

    coord = OFACoordinator(
        hass,
        entry.entry_id,
        fetcher=fetcher,
        formatter=formatter,
        lat=lat,
        lon=lon,
        update_interval=entry.data.get("update_interval", DEFAULT_UPDATE_INTERVAL),
        store_enabled=entry.data.get("persist_last_fetch", False),
        ttl=entry.data.get("persist_ttl", 3600),
        species=resolved_species,
        units=units,
        safety_limits=safety_limits,
        time_periods_mode=time_periods_mode,
    )
    _LOGGER.debug("OFACoordinator created for entry %s", entry.entry_id)

    # Ensure persisted tide coefficients (if any) are loaded into the TideProxy before the first refresh.
    try:
        await coord._tide_proxy.async_load_persisted_coeffs()
    except Exception:
        _LOGGER.exception("Failed to load persisted tide coefficients into TideProxy")

    # try to load persisted last successful fetch before first refresh (fast recovery)
    if entry.data.get("persist_last_fetch", False):
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
    try:
        await hass.config_entries.async_forward_entry_setups(entry, ["sensor"])
        _LOGGER.debug("Forwarded entry setups for entry %s to platforms", entry.entry_id)
    except Exception:
        _LOGGER.exception(
            "Failed to forward entry setups for entry %s to sensor platform", entry.entry_id
        )
        return False

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