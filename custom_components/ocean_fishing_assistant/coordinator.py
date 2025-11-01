from datetime import timedelta
import async_timeout
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.helpers.storage import Store

from .const import STORE_KEY, STORE_VERSION
from .tide_proxy import TideProxy

class OFACoordinator(DataUpdateCoordinator):
    def __init__(self, hass, entry_id, fetcher, formatter, lat, lon, update_interval, store_enabled=False, ttl=3600):
        super().__init__(
            hass,
            _LOGGER := hass.helpers.logging.getLogger(__name__),
            name="ocean_fishing_assistant",
            update_interval=timedelta(seconds=update_interval),
        )
        self.entry_id = entry_id
        self.fetcher = fetcher
        self.formatter = formatter
        self.lat = lat
        self.lon = lon
        # per-entry store key to avoid cross-entry collisions
        self._store = Store(hass, STORE_VERSION, f"{STORE_KEY}_{entry_id}") if store_enabled else None
        self._ttl = int(ttl)
        # instantiate TideProxy for this sensor coords
        self._tide_proxy = TideProxy(hass, self.lat, self.lon)

    async def _async_update_data(self):
        # fetch weather -> fetch tide -> merge -> format -> return
        async with async_timeout.timeout(60):
            raw = await self.fetcher.fetch(self.lat, self.lon, mode="hourly", days=5)
            # attempt to align tide to weather timestamps if available
            timestamps = raw.get("timestamps") or raw.get("time")
            if timestamps:
                try:
                    tide = await self._tide_proxy.get_tide_for_timestamps(timestamps)
                    # merge tide arrays into raw payload
                    raw["tide_height_m"] = tide.get("tide_height_m")
                    raw["tide_phase"] = tide.get("tide_phase")
                    raw["tide_strength"] = tide.get("tide_strength")
                except Exception:
                    _LOGGER.debug("TideProxy failed; continuing without tide", exc_info=True)
            data = self.formatter.validate(raw)
            # persist if store enabled (wrap with timestamp for TTL checks)
            if self._store:
                try:
                    import time
                    await self._store.async_save({"fetched_at": time.time(), "data": data})
                except Exception:
                    _LOGGER.debug("Failed to persist fetch to store", exc_info=True)
            return data

    async def async_load_from_store(self):
        """Load persisted data from the per-entry store if within TTL.
        Returns True if data restored, False otherwise.
        """
        if not self._store:
            return False
        try:
            raw = await self._store.async_load()
            if not raw:
                return False
            fetched_at = raw.get("fetched_at")
            stored = raw.get("data")
            if not fetched_at or not stored:
                return False
            import time
            now = time.time()
            if (now - float(fetched_at)) > float(self._ttl):
                return False
            # accept stored data and mark last_update_success
            self._data = stored
            self.last_update_success = True
            return True
        except Exception:
            _LOGGER.debug("Failed to load from store", exc_info=True)
            return False