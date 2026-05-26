"""
Strict coordinator: ensures fetcher configured using user-selected units and propagates strict errors
"""

from datetime import timedelta
import async_timeout
import logging
import time
from typing import Optional
import functools

from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import FETCH_CACHE_TTL, DOMAIN, CONF_TIDE_TTL, TIDE_PROXY_TTL_DEFAULT, CONF_TIDE_PHASE_OFFSET_MINUTES, TIDE_PHASE_OFFSET_MINUTES_DEFAULT, CONF_WORLD_TIDES_API_KEY
from .tide_proxy import TideProxy
from . import unit_helpers

# We'll instantiate TimezoneFinder lazily inside the executor to avoid blocking the event loop
# (importing TimezoneFinder at module import time can trigger package metadata I/O).
_LOGGER = logging.getLogger(__name__)


class OFACoordinator(DataUpdateCoordinator):
    def __init__(
        self,
        hass,
        entry_id: str,
        fetcher,
        formatter,
        lat: float,
        lon: float,
        update_interval: int,
        species: Optional[dict] = None,
        units: str = "metric",
        safety_limits: Optional[dict] = None,
        time_periods_mode: str = "full_day",
        *,
        fetch_cache_ttl: Optional[int] = None,
        tide_ttl: Optional[int] = None,
        tide_phase_offset_minutes: Optional[int] = None,
        tide_api_key: Optional[str] = None,
    ):
        super().__init__(
            hass,
            _LOGGER,
            name="ocean_fishing_assistant",
            update_interval=timedelta(seconds=update_interval),
        )
        self.entry_id = entry_id
        self.fetcher = fetcher
        self.formatter = formatter
        self.lat = lat
        self.lon = lon

        # Do NOT instantiate TimezoneFinder() synchronously here (it does blocking file I/O).
        # Instead, create lazily in the executor via async_init() and resolve timezone via resolve_location_tz().
        self._tf = None  # will hold TimezoneFinder instance (created in executor)
        self.location_tz: Optional[str] = None  # resolved IANA timezone name (set during setup in async_setup_entry)

        # Enforce strict contract for species: allow None or a resolved dict only.
        self.species = species
        if self.species is not None:
            if not isinstance(self.species, dict):
                raise ValueError(
                    "Coordinator requires 'species' to be a resolved dict (no fallbacks). "
                    "Pass a species dict from SpeciesLoader.get_species or get_general_profile."
                )
            if "id" not in self.species:
                raise ValueError("Provided species dict missing required 'id' key (strict)")

        self.units = units or "metric"
        # Normalize safety limits into canonical metric keys (e.g. max_wind_m_s, max_gust_m_s)
        # Accept either canonical keys (max_wind_m_s...) or legacy/display keys (safety_* from UI).
        raw_limits = safety_limits or {}
        try:
            # If the provided dict looks like UI/display keys (prefixed with "safety_"), convert -> canonical metric
            if any(str(k).startswith("safety_") for k in raw_limits.keys()):
                canonical = unit_helpers.convert_safety_display_to_metric(raw_limits, entry_units=self.units)
            else:
                canonical = dict(raw_limits)  # assume already canonical
            # Validate & normalize numeric ranges (raises on invalid if strict True)
            normalized, warnings = unit_helpers.validate_and_normalize_safety_limits(canonical, strict=True)
            self.safety_limits = normalized or {}
            if warnings:
                _LOGGER.debug("Safety limits normalized with warnings: %s", warnings)
        except Exception as exc:
            _LOGGER.exception("Failed to normalize/validate safety_limits: %s", exc)
            # Fail fast in strict mode — do not allow coordinator to start with ambiguous thresholds
            raise

        # Instance-level TTLs (allow override from entry.data)
        self._fetch_cache_ttl = int(fetch_cache_ttl) if fetch_cache_ttl is not None else int(FETCH_CACHE_TTL)

        # Phase offset (minutes) to pass into TideProxy (convert to hours)
        phase_offset_minutes = int(tide_phase_offset_minutes) if tide_phase_offset_minutes is not None else int(TIDE_PHASE_OFFSET_MINUTES_DEFAULT)
        phase_offset_hours = float(phase_offset_minutes) / 60.0

        # Store the tide API key used for TideProxy so the coordinator retains which key it's using.
        # This allows other coordinator methods (or future rebuilds) to reference the same key.
        self._tide_api_key = tide_api_key

        # create TideProxy using provided tide_ttl and phase offset (falls back to TideProxy defaults if None)
        if tide_ttl is not None:
            self._tide_proxy = TideProxy(
                hass,
                self.lat,
                self.lon,
                ttl=int(tide_ttl),
                phase_offset_hours=phase_offset_hours,
                api_key=self._tide_api_key,
            )
        else:
            self._tide_proxy = TideProxy(
                hass,
                self.lat,
                self.lon,
                phase_offset_hours=phase_offset_hours,
                api_key=self._tide_api_key,
            )

        self.time_periods_mode = time_periods_mode or "full_day"

        # Validate fetcher speed unit matches the configured units selection (strict enforcement)
        expected_speed_unit = None
        if self.units == "metric":
            expected_speed_unit = "km/h"
        elif self.units == "imperial":
            expected_speed_unit = "mph"
        else:
            expected_speed_unit = self.units

        fetcher_speed = getattr(self.fetcher, "speed_unit", None)
        if fetcher_speed is None:
            raise ValueError("Fetcher instance missing 'speed_unit' attribute; fetcher must be created with explicit units (strict)")
        if fetcher_speed != expected_speed_unit:
            raise ValueError(f"Fetcher speed_unit '{fetcher_speed}' does not match coordinator expected '{expected_speed_unit}' (strict)")

    def rebuild_tide_proxy(self, *, tide_ttl: Optional[int] = None, phase_offset_hours: Optional[float] = None, api_key: Optional[str] = None) -> None:
        """
        Helper to rebuild the TideProxy instance with the provided parameters.
        - tide_ttl: optional TTL in seconds to pass to TideProxy (falls back to existing)
        - phase_offset_hours: optional phase offset in hours
        - api_key: world tides API key (strict: must be provided)
        This updates self._tide_proxy and self._tide_api_key.
        """
        if api_key is None:
            api_key = self._tide_api_key
        if not api_key:
            raise RuntimeError("World Tides API key is required to rebuild TideProxy (strict)")

        # compute phase offset hours: preserve existing if not provided
        if phase_offset_hours is None:
            old = getattr(self._tide_proxy, "_phase_offset_hours", 0.0)
            phase_offset_hours = float(old)

        ttl_use = int(tide_ttl) if tide_ttl is not None else getattr(self._tide_proxy, "_ttl", TIDE_PROXY_TTL_DEFAULT)

        # instantiate new TideProxy; this will raise if api_key missing / invalid per strict policy
        self._tide_proxy = TideProxy(self.hass, self.lat, self.lon, ttl=int(ttl_use), phase_offset_hours=float(phase_offset_hours), api_key=api_key)
        self._tide_api_key = api_key
        _LOGGER.debug("Coordinator rebuilt TideProxy with ttl=%s phase_offset_hours=%.3f", ttl_use, phase_offset_hours)

    # Async helper to instantiate heavy objects in executor
    async def async_init(self) -> None:
        """Instantiate blocking/time-consuming helper objects in the executor.

        Call once from async_setup_entry (or lazily before first resolve_location_tz).
        """
        if self._tf is None:
            # Import and instantiate TimezoneFinder inside a worker thread to avoid blocking the event loop.
            def _make_timezonefinder():
                # local import to avoid doing file I/O at module import time
                from timezonefinder import TimezoneFinder
                return TimezoneFinder()

            self._tf = await self.hass.async_add_executor_job(_make_timezonefinder)

    async def resolve_location_tz(self, lat: float, lon: float) -> Optional[str]:
        """Resolve an IANA timezone name for lat/lon using TimezoneFinder in executor.

        Returns timezone name string or None if resolution fails. This method
        is strict — caller should decide how to handle a missing tz (we prefer fail-fast).
        """
        if self._tf is None:
            await self.async_init()

        # timezone_at is keyword-only in recent TimezoneFinder versions; use functools.partial
        try:
            func = functools.partial(self._tf.timezone_at, lat=lat, lng=lon)
            tz_name = await self.hass.async_add_executor_job(func)
            return str(tz_name) if tz_name else None
        except Exception:
            _LOGGER.exception("TimezoneFinder raised while resolving tz for %s,%s", lat, lon)
            return None

    async def _async_update_data(self):
        """Fetch weather, attach mandatory marine and tide data, run formatter. All errors propagate."""
        async with async_timeout.timeout(60):
            raw = await self._fetch_weather()
            await self._merge_marine(raw)
            timestamps = await self._fetch_tide(raw)
            period_indices = await self._compute_periods(timestamps)
            await self._fetch_current(raw)
            raw["location_tz"] = self.location_tz
            return self._run_formatter(raw, period_indices)

    async def _fetch_weather(self):
        """Fetch Open-Meteo forecast with shared cache."""
        cache_dict = self.hass.data.setdefault(DOMAIN, {}).setdefault("fetch_cache", {})
        days = 5
        lat_r, lon_r = unit_helpers.round_coords(self.lat, self.lon)
        cache_key = (lat_r, lon_r, "hourly", int(days))
        cached = cache_dict.get(cache_key)
        if cached and (time.time() - float(cached.get("fetched_at", 0))) < self._fetch_cache_ttl:
            return cached.get("data")
        raw = await self.fetcher.fetch(self.lat, self.lon, mode="hourly", days=days)
        cache_dict[cache_key] = {"fetched_at": time.time(), "data": raw}
        return raw

    async def _merge_marine(self, raw):
        """Fetch marine data from Open-Meteo Marine and merge into raw payload."""
        if not hasattr(self.fetcher, "fetch_marine_direct"):
            raise RuntimeError("Fetcher does not implement fetch_marine_direct (marine required)")
        if not isinstance(raw, dict) or "hourly" not in raw or not isinstance(raw["hourly"], dict):
            raise RuntimeError("Raw forecast payload missing required 'hourly' arrays (strict)")

        days = 5
        marine = await self.fetcher.fetch_marine_direct(days=days)
        if not isinstance(marine, dict) or "hourly" not in marine or not isinstance(marine["hourly"], dict):
            raise RuntimeError("Marine payload invalid (strict)")

        ref_time = raw["hourly"]["time"]
        if not isinstance(ref_time, (list, tuple)):
            raise ValueError("Raw hourly 'time' is not a list (strict)")
        ref_len = len(ref_time)
        for k, arr in marine["hourly"].items():
            if k == "time":
                continue
            if not isinstance(arr, (list, tuple)):
                raise ValueError(f"Marine hourly key '{k}' is not an array (strict)")
            if len(arr) != ref_len:
                raise ValueError(f"Marine hourly array '{k}' length {len(arr)} does not match forecast time length {ref_len} (strict)")
            raw["hourly"][k] = list(arr)

    async def _fetch_tide(self, raw):
        """Fetch tide data via TideProxy and normalize next_high/next_low entries."""
        timestamps = raw["hourly"]["time"]
        if not self.location_tz:
            raise RuntimeError("Coordinator missing resolved location_tz (strict)")
        tide = await self._tide_proxy.get_tide_for_timestamps(timestamps, location_tz=self.location_tz)

        def _normalize_entry(entry):
            if isinstance(entry, dict):
                entry_copy = dict(entry)
                entry_copy.pop("height_m", None)
                entry_copy.pop("height", None)
                entry_copy.pop("height_meters", None)
                return entry_copy
            if isinstance(entry, str):
                return {"timestamp": entry}
            if entry is None:
                return None
            raise ValueError(f"Unexpected tide entry type: {type(entry)} (strict)")

        tide["next_high"] = _normalize_entry(tide.get("next_high"))
        tide["next_low"] = _normalize_entry(tide.get("next_low"))
        raw["tide"] = tide
        return timestamps

    async def _compute_periods(self, timestamps):
        """Compute dawn/dusk or 4-period day indices via TideProxy and Skyfield."""
        _LOGGER.debug(
            "compute_period_indices: mode=%s count=%d first=%s last=%s tz=%s",
            self.time_periods_mode,
            len(timestamps) if timestamps is not None else 0,
            timestamps[0] if timestamps else None,
            timestamps[-1] if timestamps else None,
            self.location_tz,
        )
        try:
            return await self._tide_proxy.compute_period_indices_for_timestamps(
                timestamps,
                mode=self.time_periods_mode,
                dawn_window_hours=1.0,
                location_tz=self.location_tz,
            )
        except Exception:
            _LOGGER.exception("Failed to compute time-period indices from Skyfield (strict)")
            raise

    async def _fetch_current(self, raw):
        """Construct and validate the current-conditions snapshot."""
        if not hasattr(self.fetcher, "get_weather_data"):
            raise RuntimeError("Fetcher does not implement get_weather_data (strict)")
        try:
            current = await self.fetcher.get_weather_data()
        except Exception as exc:
            _LOGGER.exception("Failed to construct current snapshot for %s,%s: %s", self.lat, self.lon, exc)
            raise RuntimeError("Failed to construct current weather snapshot from hourly arrays (strict)") from exc

        required_current = ["temperature", "wind_speed", "wind_gust", "cloud_cover", "precipitation_probability", "pressure", "wind_unit"]
        missing_current = [k for k in required_current if not (isinstance(current, dict) and current.get(k) is not None)]
        if missing_current:
            _LOGGER.error("Current snapshot missing required fields for %s,%s: missing=%s", self.lat, self.lon, missing_current)
            raise RuntimeError(f"Current snapshot missing required fields (strict): {missing_current}")

        raw["current"] = current

    def _run_formatter(self, raw, period_indices):
        """Run strict DataFormatter validation and scoring."""
        return self.formatter.validate(
            raw,
            species_profile=self.species,
            units=self.units,
            safety_limits=self.safety_limits,
            precomputed_period_indices=period_indices,
        )