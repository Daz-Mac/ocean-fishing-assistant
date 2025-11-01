import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import config_validation as cv
from .const import DOMAIN, DEFAULT_UPDATE_INTERVAL

class OFAConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        errors = {}
        if user_input is not None:
            data = {
                CONF_LATITUDE: user_input.get(CONF_LATITUDE),
                CONF_LONGITUDE: user_input.get(CONF_LONGITUDE),
            }
            options = {
                "update_interval": user_input.get("update_interval", DEFAULT_UPDATE_INTERVAL),
                "persist_last_fetch": user_input.get("persist_last_fetch", False),
                "persist_ttl": user_input.get("persist_ttl", 3600),
            }
            return self.async_create_entry(title="Ocean Fishing Assistant", data=data, options=options)

        schema = vol.Schema(
            {
                vol.Required(CONF_LATITUDE): cv.latitude,
                vol.Required(CONF_LONGITUDE): cv.longitude,
                vol.Optional("update_interval", default=DEFAULT_UPDATE_INTERVAL): cv.positive_int,
                vol.Optional("persist_last_fetch", default=False): bool,
                vol.Optional("persist_ttl", default=3600): cv.positive_int,
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)