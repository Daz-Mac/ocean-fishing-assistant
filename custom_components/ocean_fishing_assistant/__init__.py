"""
Ocean Fishing Assistant - integration entry points (strict, ocean-only).

This implementation expects required values (including `units` and
`safety_limits`) to be present in the created config entry `data`.
If they are missing or invalid, setup fails loudly (ValueError / return False).
"""
import logging

from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import aiohttp_client

from .const import (
    DOMAIN,
    DEFAULT_UPDATE_INTERVAL,
    DEFAULT_SAFETY_LIMITS,
    CONF_SPECIES_ID,
    CONF_SPECIES_REGION,
    CONF_TIME_PERIODS,
    CONF_FETCH_CACHE_TTL,
    CONF_TIDE_TTL,
    CONF_WEATHER_CACHE_TTL,
    FETCH_CACHE_TTL,
    TIDE_PROXY_TTL_DEFAULT,
    WEATHER_FETCHER_CACHE_TTL_DEFAULT,
    # tide phase offset
    CONF_TIDE_PHASE_OFFSET_MINUTES,
    TIDE_PHASE_OFFSET_MINUTES_DEFAULT,
)

# import unit helpers to validate/convert options on updates
from .unit_helpers import convert_safety_display_to_metric, validate_and_normalize_safety_limits

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

    formatter = DataFormatter(config_entry_data=entry.data)
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

    # Validate selected species (if set) exists in the packaged profiles
    selected_species = entry.data.get(CONF_SPECIES_ID)
    selected_region = entry.data.get(CONF_SPECIES_REGION)

    resolved_species = None
    if selected_species:
        try:
            # Resolve either a general profile (top-level "general_profiles") or a specific species id.
            general_profile = loader.get_general_profile(selected_species)
            if general_profile:
                # No habitat checks — accept the general profile as-is.
                resolved_species = dict(general_profile)
                _LOGGER.debug("Resolved selected general profile '%s' to dict for entry %s", selected_species, entry.entry_id)
            else:
                # Not a general profile — resolve as a specific species id
                sp = loader.get_species(selected_species)
                if not sp:
                    raise ValueError(
                        f"Selected species '{selected_species}' not found in packaged species_profiles.json (strict)"
                    )
                # No habitat checks — accept the species profile.
                resolved_species = sp
                _LOGGER.debug("Resolved selected species id '%s' to species dict for entry %s", selected_species, entry.entry_id)
        except Exception as exc:
            _LOGGER.exception("Species validation failed for entry %s: %s", entry.entry_id, exc)
            return False

    # Read TTL overrides from entry.data (strict)
    fetch_cache_ttl = int(entry.data.get(CONF_FETCH_CACHE_TTL, FETCH_CACHE_TTL))
    tide_ttl = int(entry.data.get(CONF_TIDE_TTL, TIDE_PROXY_TTL_DEFAULT))
    weather_cache_ttl = int(entry.data.get(CONF_WEATHER_CACHE_TTL, WEATHER_FETCHER_CACHE_TTL_DEFAULT))

    # Read configured tide phase offset minutes (strict storing of the value)
    tide_phase_offset_minutes = int(entry.data.get(CONF_TIDE_PHASE_OFFSET_MINUTES, TIDE_PHASE_OFFSET_MINUTES_DEFAULT))

    # Create WeatherFetcher and coordinator using values from entry.data
    fetcher = WeatherFetcher(hass, lat, lon, speed_unit=wind_unit, cache_ttl_seconds=weather_cache_ttl)
    _LOGGER.debug("WeatherFetcher instantiated for entry %s (cache_ttl=%s)", entry.entry_id, weather_cache_ttl)

    time_periods_mode = entry.data.get(CONF_TIME_PERIODS, "full_day")

    coord = OFACoordinator(
        hass,
        entry.entry_id,
        fetcher=fetcher,
        formatter=formatter,
        lat=lat,
        lon=lon,
        update_interval=entry.data.get("update_interval", DEFAULT_UPDATE_INTERVAL),
        species=resolved_species,
        units=units,
        safety_limits=safety_limits,
        time_periods_mode=time_periods_mode,
        fetch_cache_ttl=fetch_cache_ttl,
        tide_ttl=tide_ttl,
        tide_phase_offset_minutes=tide_phase_offset_minutes,
    )
    _LOGGER.debug("OFACoordinator created for entry %s (fetch_cache_ttl=%s tide_ttl=%s tide_phase_offset_minutes=%s)", entry.entry_id, fetch_cache_ttl, tide_ttl, tide_phase_offset_minutes)

    # Instantiate TimezoneFinder in executor and resolve the location timezone (strict)
    try:
        await coord.async_init()  # ensures TimezoneFinder is created off the event loop
        tz_name = await coord.resolve_location_tz(lat, lon)
        if not tz_name:
            _LOGGER.error("Failed to resolve IANA timezone for lat=%s lon=%s (strict)", lat, lon)
            return False
        coord.location_tz = tz_name
        _LOGGER.debug("Resolved location_tz=%s for entry %s", coord.location_tz, entry.entry_id)
    except Exception as exc:
        _LOGGER.exception("Timezone resolution failed during setup_entry for %s,%s: %s", lat, lon, exc)
        return False

    # Request a fresh update (will run after any restored data is available)
    _LOGGER.debug("Requesting initial data refresh for entry %s", entry.entry_id)
    await coord.async_request_refresh()
    _LOGGER.debug("Initial data refresh requested for entry %s", entry.entry_id)

    # store coordinator in hass.data for lookups by entry_id
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coord
    _LOGGER.debug("Stored coordinator in hass.data[%s][%s]", DOMAIN, entry.entry_id)

    # Register an options update listener (strict application of updated options)
    async def _async_entry_options_updated(hass_inner, entry_inner):
        """Apply updated options into the running coordinator (strict)."""
        _LOGGER.debug("Applying updated options for entry %s", entry_inner.entry_id)
        coord_inner = hass_inner.data.get(DOMAIN, {}).get(entry_inner.entry_id)
        if coord_inner is None:
            _LOGGER.debug("Coordinator for entry %s not found when applying options", entry_inner.entry_id)
            return

        opts = entry_inner.options or {}
        # Expect top-level option keys (strict). Build safety_display from option keys.
        safety_display = {
            "safety_max_wind": opts.get("max_wind_speed"),
            "safety_max_gust": opts.get("max_gust_speed"),
            "safety_max_wave_height": opts.get("max_wave_height"),
            "safety_min_visibility": opts.get("min_visibility"),
            "safety_min_swell_period": opts.get("min_swell_period"),
            "safety_max_precip_chance": opts.get("max_precip_chance"),
        }

        # Validate/convert strictly (no fallbacks). If validation fails, raise to surface the error.
        try:
            canonical = convert_safety_display_to_metric(safety_display, entry_units=entry_inner.data.get("units", "metric"))
            normalized_limits, warnings = validate_and_normalize_safety_limits(canonical, strict=True)
            coord_inner.safety_limits = normalized_limits or {}
            _LOGGER.debug("Applied new safety_limits to coordinator for entry %s: %s (warnings=%s)", entry_inner.entry_id, normalized_limits, warnings)

            # Apply tide phase offset if provided in options: rebuild TideProxy with new offset
            try:
                new_phase_min = int(opts.get(CONF_TIDE_PHASE_OFFSET_MINUTES, entry_inner.data.get(CONF_TIDE_PHASE_OFFSET_MINUTES, TIDE_PHASE_OFFSET_MINUTES_DEFAULT)))
            except Exception:
                new_phase_min = int(TIDE_PHASE_OFFSET_MINUTES_DEFAULT)
            try:
                new_phase_hours = float(new_phase_min) / 60.0
                # preserve tide_ttl from entry.data if present otherwise use existing _tide_proxy._ttl
                tide_ttl_local = int(entry_inner.data.get(CONF_TIDE_TTL, getattr(coord_inner._tide_proxy, "_ttl", TIDE_PROXY_TTL_DEFAULT)))
                coord_inner._tide_proxy = TideProxy(hass_inner, coord_inner.lat, coord_inner.lon, ttl=tide_ttl_local, phase_offset_hours=new_phase_hours)
                _LOGGER.debug("Rebuilt TideProxy for entry %s with phase_offset_minutes=%s", entry_inner.entry_id, new_phase_min)
            except Exception:
                _LOGGER.exception("Failed to apply updated tide phase offset for entry %s", entry_inner.entry_id)
                raise

            # Apply expose_raw top-level option into stored data/options if present (no migration)
            # (Note: options are already saved by HA; we just apply runtime effect)
            await coord_inner.async_request_refresh()
        except Exception:
            _LOGGER.exception("Failed to apply/validate updated options for entry %s; raising (strict)", entry_inner.entry_id)
            # Fail loudly as per strict policy
            raise

    # Register the listener (this will cause a strict apply when options change)
    try:
        entry.add_update_listener(_async_entry_options_updated)
    except Exception:
        _LOGGER.debug("Failed to register entry update listener for entry %s", entry.entry_id)

    # forward entry to platforms
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