# (full contents — replace your existing file)
# Strict coordinator: ensures fetcher configured using user-selected units and propagates strict errors

from datetime import timedelta, datetime, timezone
import async_timeout
import logging
import time
from typing import Optional

from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import STORE_KEY, STORE_VERSION, FETCH_CACHE_TTL, DOMAIN
from .tide_proxy import TideProxy

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
        store_enabled: bool = False,
        ttl: int = 3600,
        species: Optional[str] = None,
        units: str = "metric",
        safety_limits: Optional[dict] = None,
    ):
        """
        - fetcher must be already constructed with the user-selected units (strict).
        - units here should correspond to that selection; coordinator validates fetcher speed unit consistency.
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
        self.safety_limits = safety_limits or {}
        self._store = Store(hass, STORE_VERSION, f"{STORE_KEY}_{entry_id}") if store_enabled else None
        self._ttl = int(ttl)
        self._tide_proxy = TideProxy(hass, self.lat, self.lon)

        # Validate fetcher speed unit matches the configured units selection:
        # mapping: 'metric' -> 'km/h' (requires fetcher.speed_unit == 'km/h');
        # if caller passed a more explicit units string like 'km/h'/'mph'/'m/s', accept that.
        expected_speed_unit = None
        if self.units == "metric":
            expected_speed_unit = "km/h"
        elif self.units == "imperial":
            expected_speed_unit = "mph"
        else:
            # allow direct units strings
            expected_speed_unit = self.units

        fetcher_speed = getattr(self.fetcher, "speed_unit", None)
        if fetcher_speed is None:
            raise ValueError("Fetcher instance missing 'speed_unit' attribute; fetcher must be created with explicit units (strict)")
        if fetcher_speed != expected_speed_unit:
            raise ValueError(f"Fetcher speed_unit '{fetcher_speed}' does not match coordinator expected '{expected_speed_unit}' (strict)")

    async def _async_update_data(self):
        """Fetch weather (strict), attach tide data, run formatter. All validation errors propagate."""
        async with async_timeout.timeout(60):
            cache_dict = self.hass.data.setdefault(DOMAIN, {}).setdefault("fetch_cache", {})
            cache_key = (round(float(self.lat), 4), round(float(self.lon), 4), "hourly", int(5))
            cached = cache_dict.get(cache_key)
            raw = None
            if cached and (time.time() - float(cached.get("fetched_at", 0))) < FETCH_CACHE_TTL:
                raw = cached.get("data")
            else:
                raw = await self.fetcher.fetch(self.lat, self.lon, mode="hourly", days=5)
                cache_dict[cache_key] = {"fetched_at": time.time(), "data": raw}

            # Try to fetch marine variables and align strictly if provided (errors propagate)
            if hasattr(self.fetcher, "fetch_marine_direct"):
                try:
                    marine = await self.fetcher.fetch_marine_direct(days=5)
                except Exception:
                    # treat marine as optional but do not silently modify payload shapes; only attach when shapes align exactly
                    marine = None
                if isinstance(marine, dict) and isinstance(marine.get("hourly"), dict) and isinstance(raw, dict) and isinstance(raw.get("hourly"), dict):
                    # only attach keys that have identical lengths to raw['hourly']['time']
                    ref_time = raw["hourly"]["time"]
                    if not isinstance(ref_time, (list, tuple)):
                        raise ValueError("Raw hourly 'time' is not a list (strict)")
                    ref_len = len(ref_time)
                    for k, arr in marine["hourly"].items():
                        if k == "time":
                            continue
                        if isinstance(arr, (list, tuple)) and len(arr) == ref_len:
                            raw["hourly"][k] = list(arr)

            # Attach tide strictly (tide proxy must return dict with arrays aligned to timestamps)
            if isinstance(raw, dict) and isinstance(raw.get("hourly"), dict):
                timestamps = raw["hourly"]["time"]
            else:
                timestamps = raw.get("timestamps") or raw.get("time")
            if timestamps:
                tide = await self._tide_proxy.get_tide_for_timestamps(timestamps)
                if not isinstance(tide, dict):
                    raise ValueError("TideProxy returned invalid shape (strict)")
                # only attach tide arrays if they are same length as timestamps; attach scalars as well
                for k, v in tide.items():
                    if isinstance(v, (list, tuple)) and len(v) == len(timestamps):
                        raw.setdefault("tide", {})[k] = list(v)
                    elif not isinstance(v, (list, tuple)):
                        # attach scalar metadata (e.g., tide_phase, tide_strength, source, confidence)
                        raw.setdefault("tide", {})[k] = v

            # Attach 'current' snapshot from fetcher (if present). fetcher.get_weather_data is strict and may raise.
            if hasattr(self.fetcher, "get_weather_data"):
                try:
                    current = await self.fetcher.get_weather_data()
                    raw["current"] = current
                except Exception:
                    # If get_weather_data fails, propagate: it's a strict condition
                    raise

            # Run strict formatter (errors propagate)
            data = self.formatter.validate(
                raw,
                species_profile=self.species,
                units=self.units,
                safety_limits=self.safety_limits,
            )

            # persist
            if self._store:
                await self._store.async_save({"fetched_at": time.time(), "data": data})

            return data