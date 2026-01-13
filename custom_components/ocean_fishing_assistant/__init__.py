"""
Ocean Fishing Assistant - integration entry points (strict, ocean-only).

This implementation expects required values (including `units` and
`safety_limits`) to be present in the created config entry `data`.
If they are missing or invalid, setup fails loudly (ValueError / return False).
"""
import logging
import time
from typing import Any, Dict

from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE, EVENT_HOMEASSISTANT_STOP
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    DEFAULT_UPDATE_INTERVAL,
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
    # world tides api key
    CONF_WORLD_TIDES_API_KEY,
    # shared tide cache keys
    SHARED_TIDE_CACHE_KEY,
    SHARED_TIDE_INFLIGHT_KEY,
)

# import unit helpers to validate/convert options on updates
from .unit_helpers import convert_safety_display_to_metric, validate_and_normalize_safety_limits

_LOGGER = logging.getLogger(__name__)

# --- persistence store config ---
_STORE_VERSION = 1
_STORE_KEY = "ocean_fishing_assistant_persisted_cache_v1"

# keys used in the stored JSON blob
_STORE_COORDINATOR_FETCH_KEY = "coordinator_fetch_cache"
_STORE_WEATHER_FETCH_KEY = "weather_fetch_cache"
_STORE_TIDE_CACHE_KEY = "tide_api_cache"

# Helpers to (de)serialize coordinator tuple keys used in memory.
# Coordinator uses keys like: (lat_r, lon_r, "hourly", int(days))
def _serialize_coord_cache_key(key: Any) -> str:
    try:
        if isinstance(key, (list, tuple)) and len(key) == 4:
            lat, lon, mode, days = key
            return f"fetch|{float(lat)}|{float(lon)}|{str(mode)}|{int(days)}"
    except Exception:
        pass
    return str(key)


def _deserialize_coord_cache_key(k: str):
    # expects "fetch|{lat}|{lon}|{mode}|{days}" or fallback to raw repr string
    try:
        if k.startswith("fetch|"):
            _, lat_s, lon_s, mode_s, days_s = k.split("|", 4)
            return (float(lat_s), float(lon_s), mode_s, int(days_s))
    except Exception:
        pass
    # last-resort: return the string key as-is
    return k


async def async_setup_entry(hass, entry):
    """Set up integration from a config entry (strict)."""

    _LOGGER.debug("Starting async_setup_entry for entry %s", entry.entry_id)
    session = aiohttp_client.async_get_clientsession(hass)
    _LOGGER.debug("Acquired aiohttp session: %s", session)

    # Ensure the base domain dict exists
    domain_store = hass.data.setdefault(DOMAIN, {})

    # Only initialize/load persisted caches once per HA process (first entry setup).
    if not domain_store.get("_persist_initialized"):
        store = Store(hass, _STORE_VERSION, _STORE_KEY)
        try:
            persisted = await store.async_load() or {}
        except Exception:
            _LOGGER.exception("Failed to load persisted cache store; continuing with empty caches")
            persisted = {}

        # --- Coordinator shared fetch cache ---
        raw_coord_cache = persisted.get(_STORE_COORDINATOR_FETCH_KEY) or {}
        coord_cache: Dict[Any, Any] = {}
        try:
            now_ts = int(dt_util.now().timestamp())
            for k_str, v in raw_coord_cache.items():
                try:
                    tup_key = _deserialize_coord_cache_key(k_str)
                    # expect v to be dict with 'fetched_at' (float) and 'data'
                    fetched_at = float(v.get("fetched_at", 0))
                    # prune if older than FETCH_CACHE_TTL
                    if (now_ts - int(fetched_at)) < int(FETCH_CACHE_TTL):
                        coord_cache[tup_key] = v
                except Exception:
                    _LOGGER.debug("Skipping malformed coordinator cache entry on load: %s", k_str)
        except Exception:
            _LOGGER.exception("Failed to reconstruct persisted coordinator fetch cache")

        domain_store.setdefault("fetch_cache", coord_cache)

        # --- Tide shared cache (entries include 'expires' ints) ---
        raw_tide_cache = persisted.get(_STORE_TIDE_CACHE_KEY) or {}
        tide_cache: Dict[str, Any] = {}
        try:
            now_ts = int(dt_util.now().timestamp())
            for k_str, v in raw_tide_cache.items():
                # v expected: {"expires": <int>, ...}
                try:
                    expires = int(v.get("expires", 0))
                    if expires >= now_ts:
                        tide_cache[k_str] = v
                    else:
                        # expired; skip
                        pass
                except Exception:
                    _LOGGER.debug("Skipping malformed tide cache entry on load: %s", k_str)
        except Exception:
            _LOGGER.exception("Failed to reconstruct persisted tide cache")

        domain_store.setdefault(SHARED_TIDE_CACHE_KEY, tide_cache)
        # ensure inflight key exists (not persisted)
        domain_store.setdefault(SHARED_TIDE_INFLIGHT_KEY, {})

        # --- WeatherFetcher cache (stored under a different hass.data key historically) ---
        # WeatherFetcher entries are like: { cache_key: {"data": ..., "time": "<ISO str>"} }
        raw_weather_cache = persisted.get(_STORE_WEATHER_FETCH_KEY) or {}
        weather_cache: Dict[str, Any] = {}
        try:
            for k_str, v in raw_weather_cache.items():
                try:
                    # parse time back into datetime if present
                    entry_time = v.get("time")
                    valid = False
                    if entry_time:
                        parsed = dt_util.parse_datetime(str(entry_time))
                        if parsed:
                            # WeatherFetcher expects a datetime object in memory
                            weather_cache[k_str] = {"data": v.get("data"), "time": parsed}
                            valid = True
                    # if no time field / parse failed, skip if stale; else attempt to accept numeric timestamp
                    if not valid:
                        # skip entries we can't parse
                        _LOGGER.debug("Skipping malformed/old weather cache entry on load: %s", k_str)
                except Exception:
                    _LOGGER.debug("Skipping malformed weather cache entry on load: %s", k_str)
        except Exception:
            _LOGGER.exception("Failed to reconstruct persisted weather fetch cache")

        # WeatherFetcher historically uses a top-level key "ocean_fishing_assistant_fetch_cache"
        hass.data.setdefault("ocean_fishing_assistant_fetch_cache", weather_cache)

        # store Store instance and mark initialized
        domain_store["_persist_store"] = store
        domain_store["_persist_initialized"] = True

        # Register save-on-stop to persist caches. Use async_listen_once so we save on clean shutdown.
        async def _save_on_stop(event):
            try:
                s = domain_store.get("_persist_store") or store
                to_save: Dict[str, Any] = {}

                # Serialize coordinator fetch cache: convert tuple keys -> strings
                coord_cache_local = domain_store.get("fetch_cache", {}) or {}
                coord_serialized = {}
                for k, v in coord_cache_local.items():
                    try:
                        kstr = _serialize_coord_cache_key(k)
                        coord_serialized[kstr] = v
                    except Exception:
                        _LOGGER.debug("Skipping non-serializable coordinator fetch_cache key on save: %r", k)
                to_save[_STORE_COORDINATOR_FETCH_KEY] = coord_serialized

                # Serialize tide cache (already JSON-friendly; just copy)
                tide_cache_local = domain_store.get(SHARED_TIDE_CACHE_KEY, {}) or {}
                to_save[_STORE_TIDE_CACHE_KEY] = tide_cache_local

                # Serialize weather fetch cache: convert datetime -> ISO string
                weather_cache_local = hass.data.get("ocean_fishing_assistant_fetch_cache", {}) or {}
                weather_serialized = {}
                for k, v in weather_cache_local.items():
                    try:
                        # expect v like {"data": ..., "time": datetime}
                        tval = v.get("time")
                        t_iso = None
                        if tval is not None:
                            try:
                                t_iso = tval.isoformat()
                            except Exception:
                                # fallback: numeric timestamp
                                try:
                                    t_iso = str(int(time.mktime(tval.timetuple())))
                                except Exception:
                                    t_iso = None
                        weather_serialized[k] = {"data": v.get("data"), "time": t_iso}
                    except Exception:
                        _LOGGER.debug("Skipping non-serializable weather cache entry on save: %s", k)
                to_save[_STORE_WEATHER_FETCH_KEY] = weather_serialized

                await s.async_save(to_save)
                _LOGGER.debug("Persisted ocean_fishing_assistant caches on stop: coord=%d tide=%d weather=%d",
                              len(coord_serialized), len(tide_cache_local), len(weather_serialized))
            except Exception:
                _LOGGER.exception("Failed to persist ocean_fishing_assistant caches on stop")

        try:
            hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, _save_on_stop)
            _LOGGER.debug("Registered Home Assistant stop listener to persist caches")
        except Exception:
            _LOGGER.exception("Failed to register stop listener for cache persistence")

    else:
        # ensure the keys exist so other modules can rely on them
        domain_store.setdefault("fetch_cache", {})
        domain_store.setdefault(SHARED_TIDE_CACHE_KEY, {})
        domain_store.setdefault(SHARED_TIDE_INFLIGHT_KEY, {})
        hass.data.setdefault("ocean_fishing_assistant_fetch_cache", {})

    fetch_cache = domain_store.get("fetch_cache")
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
        from .tide_proxy import TideProxy
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

    # Read configured tide phase offset minutes (prefer options if present, fallback to entry.data)
    tide_phase_offset_minutes = int(
        entry.options.get(
            CONF_TIDE_PHASE_OFFSET_MINUTES,
            entry.data.get(CONF_TIDE_PHASE_OFFSET_MINUTES, TIDE_PHASE_OFFSET_MINUTES_DEFAULT),
        )
    )
    _LOGGER.debug(
        "Resolved tide_phase_offset_minutes at startup for entry %s: options=%s data=%s chosen=%s",
        entry.entry_id,
        entry.options,
        entry.data.get(CONF_TIDE_PHASE_OFFSET_MINUTES),
        tide_phase_offset_minutes,
    )

    # Read World Tides API key from entry.data (strict)
    world_tides_api_key = entry.data.get(CONF_WORLD_TIDES_API_KEY)
    if not world_tides_api_key:
        _LOGGER.error("Config entry %s missing required World Tides API key in entry.data (strict)", entry.entry_id)
        return False

    # Create WeatherFetcher and coordinator using values from entry.data
    fetcher = WeatherFetcher(hass, lat, lon, speed_unit=wind_unit, cache_ttl_seconds=weather_cache_ttl)
    _LOGGER.debug("WeatherFetcher instantiated for entry %s (cache_ttl=%s)", entry.entry_id, weather_cache_ttl)

    # Prefer runtime options (entry.options) for time period selection, fall back to entry.data (strict)
    time_periods_mode = entry.options.get(CONF_TIME_PERIODS, entry.data.get(CONF_TIME_PERIODS, "full_day"))
    _LOGGER.debug(
        "Resolved time_periods_mode at startup for entry %s: options=%s data=%s chosen=%s",
        entry.entry_id,
        entry.options,
        entry.data.get(CONF_TIME_PERIODS),
        time_periods_mode,
    )

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
        tide_api_key=world_tides_api_key,
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
        _LOGGER.debug(
            "Options update callback invoked for entry %s: entry.options=%s entry.data_tide=%s",
            entry_inner.entry_id,
            entry_inner.options,
            entry_inner.data.get(CONF_TIDE_PHASE_OFFSET_MINUTES),
        )
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

            # Apply time_periods_mode update from options (strict)
            try:
                new_time_periods_mode = entry_inner.options.get(CONF_TIME_PERIODS, entry_inner.data.get(CONF_TIME_PERIODS, "full_day"))
                # Strict validation of allowed modes
                allowed_modes = {"full_day", "dawn_dusk"}
                if not isinstance(new_time_periods_mode, str) or new_time_periods_mode not in allowed_modes:
                    raise ValueError(f"Invalid time_periods_mode option: {new_time_periods_mode!r} (allowed: {allowed_modes})")
                old_mode = getattr(coord_inner, "time_periods_mode", None)
                coord_inner.time_periods_mode = new_time_periods_mode
                _LOGGER.debug("Applied time_periods_mode change for entry %s: old=%s new=%s", entry_inner.entry_id, old_mode, new_time_periods_mode)
            except Exception:
                _LOGGER.exception("Failed to apply time_periods_mode option for entry %s (strict)", entry_inner.entry_id)
                # Fail loudly to surface invalid options
                raise

            # Apply tide phase offset if provided in options: rebuild TideProxy with new offset
            try:
                new_phase_min = int(opts.get(CONF_TIDE_PHASE_OFFSET_MINUTES, entry_inner.data.get(CONF_TIDE_PHASE_OFFSET_MINUTES, TIDE_PHASE_OFFSET_MINUTES_DEFAULT)))
            except Exception:
                new_phase_min = int(TIDE_PHASE_OFFSET_MINUTES_DEFAULT)
            try:
                new_phase_hours = float(new_phase_min) / 60.0
                # preserve tide_ttl from entry.data if present otherwise use existing _tide_proxy._ttl
                tide_ttl_local = int(entry_inner.data.get(CONF_TIDE_TTL, getattr(coord_inner._tide_proxy, "_ttl", TIDE_PROXY_TTL_DEFAULT)))
                # Rebuild TideProxy with the stored world tides key in entry.data via helper
                world_key = entry_inner.data.get(CONF_WORLD_TIDES_API_KEY)
                coord_inner.rebuild_tide_proxy(tide_ttl=tide_ttl_local, phase_offset_hours=new_phase_hours, api_key=world_key)
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