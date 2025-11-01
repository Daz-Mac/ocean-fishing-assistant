import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import config_validation as cv
from typing import Any, Dict, Optional

from .const import DOMAIN, DEFAULT_UPDATE_INTERVAL

class OFAConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: Optional[Dict[str, Any]] = None):
        errors = {}
        if user_input is not None:
            # Store coordinates in data; options include runtime flags and required safety limits
            data = {
                CONF_LATITUDE: user_input.get(CONF_LATITUDE),
                CONF_LONGITUDE: user_input.get(CONF_LONGITUDE),
            }
            # Collect safety limits into a single dict stored under options
            safety_limits = {
                "max_wind": float(user_input.get("safety_max_wind")),
                "max_wave_height": float(user_input.get("safety_max_wave_height")),
                "min_visibility": float(user_input.get("safety_min_visibility")),
                "max_swell_period": float(user_input.get("safety_max_swell_period")),
            }
            options = {
                "update_interval": user_input.get("update_interval", DEFAULT_UPDATE_INTERVAL),
                "persist_last_fetch": user_input.get("persist_last_fetch", False),
                "persist_ttl": user_input.get("persist_ttl", 3600),
                "species": user_input.get("species"),
                "units": user_input.get("units", "metric"),
                "safety_limits": safety_limits,
            }
            return self.async_create_entry(title="Ocean Fishing Assistant", data=data, options=options)

        # Require users to provide explicit safety limits at setup. Units determine how wind is entered (km/h or mph)
        schema = vol.Schema(
            {
                vol.Required(CONF_LATITUDE): cv.latitude,
                vol.Required(CONF_LONGITUDE): cv.longitude,
                vol.Optional("update_interval", default=DEFAULT_UPDATE_INTERVAL): cv.positive_int,
                vol.Optional("persist_last_fetch", default=False): bool,
                vol.Optional("persist_ttl", default=3600): cv.positive_int,
                vol.Optional("species", default=""): cv.string,
                vol.Required("units", default="metric"): vol.In(["metric", "imperial"]),
                vol.Required("safety_max_wind"): vol.Coerce(float),
                vol.Required("safety_max_wave_height"): vol.Coerce(float),
                vol.Required("safety_min_visibility"): vol.Coerce(float),
                vol.Required("safety_max_swell_period"): vol.Coerce(float),
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    @staticmethod
    def async_get_options_flow(config_entry):
        return OFAOptionsFlowHandler(config_entry)


class OFAOptionsFlowHandler(config_entries.OptionsFlow):
    def __init__(self, config_entry):
        self.config_entry = config_entry

    async def async_step_init(self, user_input: Optional[Dict[str, Any]] = None):
        errors = {}
        current = dict(self.config_entry.options or {})

        if user_input is not None:
            # validate species exists in species_profiles.json if provided
            species = user_input.get("species") or None
            if species:
                # Attempt to load species list from file packaged with integration
                try:
                    import json, pkgutil
                    raw = pkgutil.get_data(__name__, "species_profiles.json")
                    profiles = json.loads(raw.decode("utf-8")) if raw else {}
                    if species not in profiles:
                        errors["species"] = "invalid_species"
                except Exception:
                    # if we can't validate, accept but warn (no blocking error)
                    pass

            if not errors:
                # Preserve and update safety limits from the options flow
                new_safety = {
                    "max_wind": float(user_input.get("safety_max_wind")) if user_input.get("safety_max_wind") is not None else current.get("safety_limits", {}).get("max_wind"),
                    "max_wave_height": float(user_input.get("safety_max_wave_height")) if user_input.get("safety_max_wave_height") is not None else current.get("safety_limits", {}).get("max_wave_height"),
                    "min_visibility": float(user_input.get("safety_min_visibility")) if user_input.get("safety_min_visibility") is not None else current.get("safety_limits", {}).get("min_visibility"),
                    "max_swell_period": float(user_input.get("safety_max_swell_period")) if user_input.get("safety_max_swell_period") is not None else current.get("safety_limits", {}).get("max_swell_period"),
                }

                new_options = {
                    "update_interval": int(user_input.get("update_interval", current.get("update_interval", DEFAULT_UPDATE_INTERVAL))),
                    "persist_last_fetch": bool(user_input.get("persist_last_fetch", current.get("persist_last_fetch", False))),
                    "persist_ttl": int(user_input.get("persist_ttl", current.get("persist_ttl", 3600))),
                    "species": user_input.get("species", current.get("species")),
                    "units": user_input.get("units", current.get("units", "metric")),
                    "safety_limits": new_safety,
                }
                return self.async_create_entry(title="", data=new_options)

        # Build defaults for schema using currently saved options (if present)
        saved_safety = current.get("safety_limits", {}) or {}
        schema = vol.Schema(
            {
                vol.Optional("update_interval", default=current.get("update_interval", DEFAULT_UPDATE_INTERVAL)): cv.positive_int,
                vol.Optional("persist_last_fetch", default=current.get("persist_last_fetch", False)): bool,
                vol.Optional("persist_ttl", default=current.get("persist_ttl", 3600)): cv.positive_int,
                vol.Optional("species", default=current.get("species", "")): cv.string,
                vol.Optional("units", default=current.get("units", "metric")): vol.In(["metric", "imperial"]),
                vol.Optional("safety_max_wind", default=saved_safety.get("max_wind")): vol.Any(vol.Coerce(float), None),
                vol.Optional("safety_max_wave_height", default=saved_safety.get("max_wave_height")): vol.Any(vol.Coerce(float), None),
                vol.Optional("safety_min_visibility", default=saved_safety.get("min_visibility")): vol.Any(vol.Coerce(float), None),
                vol.Optional("safety_max_swell_period", default=saved_safety.get("max_swell_period")): vol.Any(vol.Coerce(float), None),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema, errors=errors)