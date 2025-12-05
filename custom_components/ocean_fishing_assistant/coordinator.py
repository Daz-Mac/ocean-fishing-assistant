# custom_components/ocean_fishing_assistant/coordinator.py
"""
Strict coordinator: ensures fetcher configured using user-selected units and propagates strict errors
"""

from datetime import timedelta
import async_timeout
import logging
import time
from typing import Optional

from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import FETCH_CACHE_TTL, DOMAIN
from .tide_proxy import TideProxy
from . import unit_helpers

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
        species: Optional[str] = None,
        units: str = "metric",
        safety_limits: Optional[dict] = None,
        time_periods_mode: str = "full_day",
        *,
        fetch_cache_ttl: Optional[int] = None,
        tide_ttl: Optional[int] = None,
    ):
        """
        - fetcher must be constructed with the user-selected wind unit (strict).
        - coordinator validates fetcher.speed_unit matches options.
        - time_periods_mode: one of the CONF_TIME_PERIODS values (full_day/dawn_dusk)
        - fetch_cache_ttl: per-entry override for the shared in-memory fetch cache TTL (seconds)
        - tide_ttl: TTL (seconds) to pass into TideProxy instance
        """
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
        self.species = species
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

        # create TideProxy using provided tide_ttl (falls back to TideProxy default if None)
        if tide_ttl is not None:
            self._tide_proxy = TideProxy(hass, self.lat, self.lon, ttl=int(tide_ttl))
        else:
            self._tide_proxy = TideProxy(hass, self.lat, self.lon)

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

    async def _async_update_data(self):
        """Fetch weather, attach mandatory marine and tide data, run formatter. All errors propagate."""
        async with async_timeout.timeout(60):
            cache_dict = self.hass.data.setdefault(DOMAIN, {}).setdefault("fetch_cache", {})
            # Use an explicit 'days' variable — ensures cache key matches fetch parameters.
            days = 5
            # Use centralized rounding helper so coordinate rounding precision is consistent across modules
            lat_r, lon_r = unit_helpers.round_coords(self.lat, self.lon)
            cache_key = (lat_r, lon_r, "hourly", int(days))
            cached = cache_dict.get(cache_key)
            raw = None
            if cached and (time.time() - float(cached.get("fetched_at", 0))) < self._fetch_cache_ttl:
                raw = cached.get("data")
            else:
                # fetch raw Open-Meteo payload strictly (may raise)
                raw = await self.fetcher.fetch(self.lat, self.lon, mode="hourly", days=days)
                cache_dict[cache_key] = {"fetched_at": time.time(), "data": raw}

            # Fetch marine variables (STRICT: marine is required for ocean assistant)
            if not hasattr(self.fetcher, "fetch_marine_direct"):
                raise RuntimeError("Fetcher does not implement fetch_marine_direct (marine required)")

            marine = await self.fetcher.fetch_marine_direct(days=days)  # will raise on failure
            if not isinstance(marine, dict) or "hourly" not in marine or not isinstance(marine["hourly"], dict):
                raise RuntimeError("Marine payload invalid (strict)")

            # Only attach marine arrays that align exactly with raw['hourly']['time']
            if not isinstance(raw, dict) or "hourly" not in raw or not isinstance(raw["hourly"], dict):
                raise RuntimeError("Raw forecast payload missing required 'hourly' arrays (strict)")
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

            # Attach tide strictly (tide proxy must return dict with arrays aligned to timestamps)
            timestamps = raw["hourly"]["time"]
            tide = await self._tide_proxy.get_tide_for_timestamps(timestamps)
            if not isinstance(tide, dict):
                raise ValueError("TideProxy returned invalid shape (strict)")

            # --- Normalization for strict canonical shape ---
            try:
                nh = tide.get("next_high")
                nl = tide.get("next_low")
                nh_h = tide.get("next_high_height_m")
                nl_h = tide.get("next_low_height_m")

                def _normalize_entry(entry, height_scalar):
                    # If already dict, pass through
                    if isinstance(entry, dict):
                        return entry
                    # If a simple timestamp string, convert to strict dict and attach possible height
                    if isinstance(entry, str):
                        try:
                            h_val = None if height_scalar is None else float(height_scalar)
                        except Exception:
                            h_val = None
                        return {"timestamp": entry, "height_m": h_val}
                    # Unknown or missing -> None
                    return None

                nh_obj = _normalize_entry(nh, nh_h)
                nl_obj = _normalize_entry(nl, nl_h)

                # Remove legacy separate height keys to avoid duplication/ambiguity downstream
                tide.pop("next_high_height_m", None)
                tide.pop("next_low_height_m", None)

                # Overwrite canonical keys with normalized objects (may be None)
                tide["next_high"] = nh_obj
                tide["next_low"] = nl_obj
            except Exception:
                # If normalization fails for any reason, log but continue — downstream strict checks will catch it.
                _LOGGER.exception("Failed to normalize tide 'next_high'/'next_low' payload; continuing with original tide object")

            # only attach tide arrays if they are same length as timestamps; attach scalars as well
            for k, v in tide.items():
                if isinstance(v, (list, tuple)):
                    if len(v) != len(timestamps):
                        raise ValueError(f"Tide array '{k}' length {len(v)} does not match timestamps length {len(timestamps)} (strict)")
                    raw.setdefault("tide", {})[k] = list(v)
                else:
                    raw.setdefault("tide", {})[k] = v

            # Precompute period indices using TideProxy + Skyfield (strict)
            # Use dawn/dusk window ±1 hour by default for dawn_dusk mode.
            try:
                period_indices = await self._tide_proxy.compute_period_indices_for_timestamps(
                    timestamps,
                    mode=self.time_periods_mode,
                    dawn_window_hours=1.0,
                )
            except Exception:
                _LOGGER.exception("Failed to compute time-period indices from Skyfield (strict)")
                # propagate strict failure
                raise

            # Attach current snapshot (strict)
            if not hasattr(self.fetcher, "get_weather_data"):
                raise RuntimeError("Fetcher does not implement get_weather_data (strict)")
            try:
                current = await self.fetcher.get_weather_data()  # will raise on failure
            except Exception as exc:
                # Log with context for easier debugging, then re-raise as a strict error.
                _LOGGER.exception(
                    "Failed to construct current snapshot from Open-Meteo hourly data for %s,%s: %s",
                    self.lat,
                    self.lon,
                    exc,
                )
                raise RuntimeError("Failed to construct current weather snapshot from hourly arrays (strict)") from exc

            # STRICT sanity check: ensure current contains required keys (all required under strict policy)
            required_current = ["temperature", "wind_speed", "wind_gust", "cloud_cover", "precipitation_probability", "pressure", "wind_unit"]
            missing_current = [k for k in required_current if not (isinstance(current, dict) and current.get(k) is not None)]
            if missing_current:
                _LOGGER.error(
                    "Constructed current snapshot missing required fields for %s,%s: missing=%s current=%s",
                    self.lat,
                    self.lon,
                    missing_current,
                    current,
                )
                raise RuntimeError(f"Constructed current snapshot missing required fields (strict): {missing_current}")

            raw["current"] = current

            # Run strict formatter (errors propagate). Pass precomputed period indices so DataFormatter uses them.
            data = self.formatter.validate(
                raw,
                species_profile=self.species,
                units=self.units,
                safety_limits=self.safety_limits,
                precomputed_period_indices=period_indices,
            )

            # No disk persistence anymore
            return data