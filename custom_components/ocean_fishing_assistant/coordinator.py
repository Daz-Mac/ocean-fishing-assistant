from datetime import timedelta
import async_timeout
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.helpers.storage import Store
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE

from .const import STORE_KEY, STORE_VERSION

class OFACoordinator(DataUpdateCoordinator):
    def __init__(self, hass, fetcher, formatter, lat, lon, update_interval, store_enabled=False, ttl=3600):
        super().__init__(
            hass,
            _LOGGER := hass.helpers.logging.getLogger(__name__),
            name="ocean_fishing_assistant",
            update_interval=timedelta(seconds=update_interval),
        )
        self.fetcher = fetcher
        self.formatter = formatter
        self.lat = lat
        self.lon = lon
        self._store = Store(hass, STORE_VERSION, STORE_KEY) if store_enabled else None
        self._ttl = ttl

    async def _async_update_data(self):
        # fetch -> format -> return
        async with async_timeout.timeout(30):
            raw = await self.fetcher.fetch(self.lat, self.lon, mode="hourly")
            data = self.formatter.validate(raw)
            # persist if store enabled
            if self._store:
                await self._store.async_save(data)
            return data

    async def async_load_from_store(self):
        if not self._store:
            return
        stored = await self._store.async_load()
        if stored:
            self.data = stored
            self._last_update_success = True