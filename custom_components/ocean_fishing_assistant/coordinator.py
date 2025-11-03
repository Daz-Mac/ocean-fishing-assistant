from datetime import timedelta
import async_timeout
import logging
import time
from typing import Optional

from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.helpers.storage import Store

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
        """Coordinator that fetches, aligns tide, formats data and persists last fetch."""
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
        # per-entry options
        self.species = species
        self.units = units or "metric"
        # canonicalized safety limits in metric units (e.g. m/s, m, km, s)
        self.safety_limits = safety_limits or {}
        # per-entry store key to avoid cross-entry collisions
        self._store = Store(hass, STORE_VERSION, f"{STORE_KEY}_{entry_id}") if store_enabled else None
        self._ttl = int(ttl)
        # instantiate TideProxy for this sensor coords
        self._tide_proxy = TideProxy(hass, self.lat, self.lon)

    async def _async_update_data(self):
        """Fetch weather (cached), attach tide data, run formatter to precompute forecasts."""
        async with async_timeout.timeout(60):
            # build robust cache access via hass.data
            cache_dict = self.hass.data.setdefault(DOMAIN, {}).setdefault("fetch_cache", {})
            cache_key = (round(float(self.lat), 4), round(float(self.lon), 4), "hourly", int(5))
            cached = cache_dict.get(cache_key)
            raw = None
            try:
                if cached and (time.time() - float(cached.get("fetched_at", 0))) < FETCH_CACHE_TTL:
                    raw = cached.get("data")
                else:
                    raw = await self.fetcher.fetch(self.lat, self.lon, mode="hourly", days=5)
                    try:
                        cache_dict[cache_key] = {"fetched_at": time.time(), "data": raw}
                    except Exception:
                        _LOGGER.debug("Failed to update in-memory fetch cache", exc_info=True)
            except Exception:
                # fallback to fetching directly if cache logic fails
                raw = await self.fetcher.fetch(self.lat, self.lon, mode="hourly", days=5)

            # attempt to align tide to weather timestamps if available
            timestamps = None
            if isinstance(raw, dict) and isinstance(raw.get("hourly"), dict):
                timestamps = raw.get("hourly", {}).get("time") or raw.get("timestamps") or raw.get("time") or []
            else:
                timestamps = raw.get("timestamps") or raw.get("time") or []

            if timestamps:
                try:
                    tide = await self._tide_proxy.get_tide_for_timestamps(timestamps)
                    # Keep a top-level tide dict for the formatter to consume
                    raw.setdefault("tide", {}).update(tide)
                    # Backwards-compatible convenience keys
                    if "tide_height_m" not in raw and tide.get("tide_height_m") is not None:
                        raw["tide_height_m"] = tide.get("tide_height_m")
                    if "tide_phase" not in raw and tide.get("tide_phase") is not None:
                        raw["tide_phase"] = tide.get("tide_phase")
                    if "tide_strength" not in raw and tide.get("tide_strength") is not None:
                        raw["tide_strength"] = tide.get("tide_strength")
                except Exception:
                    _LOGGER.debug("TideProxy failed; continuing without tide", exc_info=True)

            # Attach convenience 'current' snapshot if the fetcher can provide it
            try:
                if hasattr(self.fetcher, "get_weather_data"):
                    current = await self.fetcher.get_weather_data()
                    if current:
                        raw["current"] = current
            except Exception:
                _LOGGER.debug("Failed to get current snapshot from fetcher", exc_info=True)

            # IMPORTANT: pass per-entry species, units and safety_limits into the formatter
            data = self.formatter.validate(
                raw,
                species_profile=self.species,
                units=self.units,
                safety_limits=self.safety_limits,
            )

            # persist if store enabled (wrap with timestamp for TTL checks)
            if self._store:
                try:
                    await self._store.async_save({"fetched_at": time.time(), "data": data})
                except Exception:
                    _LOGGER.debug("Failed to persist fetch to store", exc_info=True)

            return data