# Strict coordinator: ensures fetcher configured using user-selected units and propagates strict errors

from datetime import timedelta
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
        - fetcher must be constructed with the user-selected wind unit (strict).
        - coordinator validates fetcher.speed_unit matches options.
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
            cache_key = (round(float(self.lat), 4), round(float(self.lon), 4), "hourly", int(5))
            cached = cache_dict.get(cache_key)
            raw = None
            if cached and (time.time() - float(cached.get("fetched_at", 0))) < FETCH_CACHE_TTL:
                raw = cached.get("data")
            else:
                # fetch raw Open-Meteo payload strictly (may raise)
                raw = await self.fetcher.fetch(self.lat, self.lon, mode="hourly", days=5)
                cache_dict[cache_key] = {"fetched_at": time.time(), "data": raw}

            # Fetch marine variables (STRICT: marine is required for ocean assistant)
            if not hasattr(self.fetcher, "fetch_marine_direct"):
                raise RuntimeError("Fetcher does not implement fetch_marine_direct (marine required)")

            marine = await self.fetcher.fetch_marine_direct(days=5)  # will raise on failure
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
            # only attach tide arrays if they are same length as timestamps; attach scalars as well
            for k, v in tide.items():
                if isinstance(v, (list, tuple)):
                    if len(v) != len(timestamps):
                        raise ValueError(f"Tide array '{k}' length {len(v)} does not match timestamps length {len(timestamps)} (strict)")
                    raw.setdefault("tide", {})[k] = list(v)
                else:
                    raw.setdefault("tide", {})[k] = v

            # Attach current snapshot (strict)
            if not hasattr(self.fetcher, "get_weather_data"):
                raise RuntimeError("Fetcher does not implement get_weather_data (strict)")
            current = await self.fetcher.get_weather_data()  # will raise on failure
            raw["current"] = current

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