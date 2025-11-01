import logging
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import selector
from typing import Any, Dict, Optional

from .const import DOMAIN, DEFAULT_UPDATE_INTERVAL
from .unit_helpers import convert_safety_display_to_metric, validate_and_normalize_safety_limits

_LOGGER = logging.getLogger(__name__)

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

            # Collect raw UI safety inputs and convert to canonical metric keys
            display_safety = {
                "safety_max_wind": user_input.get("safety_max_wind"),
                "safety_max_wave_height": user_input.get("safety_max_wave_height"),
                "safety_min_visibility": user_input.get("safety_min_visibility"),
                "safety_max_swell_period": user_input.get("safety_max_swell_period"),
            }
            entry_units = user_input.get("units", "metric")
            metric_safety = convert_safety_display_to_metric(display_safety, entry_units=entry_units)

            # Validate & normalize (warn-and-clamp by default)
            normalized_safety, warnings = validate_and_normalize_safety_limits(metric_safety, strict=False)
            for w in warnings:
                _LOGGER.warning("Safety limits normalization: %s", w)

            options = {
                "update_interval": user_input.get("update_interval", DEFAULT_UPDATE_INTERVAL),
                "persist_last_fetch": user_input.get("persist_last_fetch", False),
                "persist_ttl": user_input.get("persist_ttl", 3600),
                "species": user_input.get("species"),
                "units": entry_units,
                "safety_limits": normalized_safety,
            }
            return self.async_create_entry(title="Ocean Fishing Assistant", data=data, options=options)

        # Require users to provide explicit safety limits at setup. We show sliders (NumberSelector) with metric defaults.
        schema = vol.Schema(
            {
                vol.Required(CONF_LATITUDE): cv.latitude,
                vol.Required(CONF_LONGITUDE): cv.longitude,
                vol.Optional("update_interval", default=DEFAULT_UPDATE_INTERVAL): cv.positive_int,
                vol.Optional("persist_last_fetch", default=False): bool,
                vol.Optional("persist_ttl", default=3600): cv.positive_int,
                vol.Optional("species", default=""): cv.string,
                vol.Required("units", default="metric"): vol.In(["metric", "imperial"]),
                vol.Required("safety_max_wind"): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=10, max=100, step=1, unit_of_measurement="km/h", mode="slider")
                ),
                vol.Required("safety_max_wave_height"): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.5, max=10.0, step=0.1, unit_of_measurement="m", mode="slider")
                ),
                vol.Required("safety_min_visibility"): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0, max=200, step=1, unit_of_measurement="km", mode="slider")
                ),
                vol.Required("safety_max_swell_period"): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=1, max=120, step=1, unit_of_measurement="s", mode="slider")
                ),
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
                    raw = pkgutil.get_data(__package__, "species_profiles.json")
                    profiles = json.loads(raw.decode("utf-8")) if raw else {}
                    if species not in profiles:
                        errors["species"] = "invalid_species"
                except Exception:
                    # if we can't validate, accept but warn (no blocking error)
                    pass

            if not errors:
                # Convert display values (from the options flow) into canonical metric keys
                entry_units = user_input.get("units", current.get("units", "metric"))
                display_safety = {
                    "safety_max_wind": user_input.get("safety_max_wind"),
                    "safety_max_wave_height": user_input.get("safety_max_wave_height"),
                    "safety_min_visibility": user_input.get("safety_min_visibility"),
                    "safety_max_swell_period": user_input.get("safety_max_swell_period"),
                }
                metric_safety = convert_safety_display_to_metric(display_safety, entry_units=entry_units)
                normalized_safety, warnings = validate_and_normalize_safety_limits(metric_safety, strict=False)
                for w in warnings:
                    _LOGGER.warning("Safety limits normalization (options flow): %s", w)

                new_options = {
                    "update_interval": int(user_input.get("update_interval", current.get("update_interval", DEFAULT_UPDATE_INTERVAL))),
                    "persist_last_fetch": bool(user_input.get("persist_last_fetch", current.get("persist_last_fetch", False))),
                    "persist_ttl": int(user_input.get("persist_ttl", current.get("persist_ttl", 3600))),
                    "species": user_input.get("species", current.get("species")),
                    "units": entry_units,
                    "safety_limits": normalized_safety,
                }
                return self.async_create_entry(title="", data=new_options)

        # Build defaults for schema using currently saved options (if present)
        saved_safety = current.get("safety_limits", {}) or {}
        # Support old key names if present (backwards compatibility)
        # canonical metric keys we store: max_wind_m_s, max_wave_height_m, min_visibility_km, max_swell_period_s
        def _pick(key_variants, target=None):
            for k in key_variants:
                if k in saved_safety and saved_safety.get(k) is not None:
                    return saved_safety.get(k)
            return target

        entry_units = current.get("units", "metric")
        # derive display defaults (convert metric stored values to UI units)
        wind_metric = _pick(["max_wind_m_s", "max_wind"], None)
        if wind_metric is not None:
            if entry_units == "metric":
                wind_default = (float(wind_metric) * 3.6)
                wind_unit = "km/h"
            else:
                wind_default = (float(wind_metric) * 2.2369362920544)
                wind_unit = "mph"
        else:
            wind_default = None
            wind_unit = "km/h" if entry_units == "metric" else "mph"

        wave_metric = _pick(["max_wave_height_m", "max_wave_height"], None)
        if wave_metric is not None:
            if entry_units == "metric":
                wave_default = float(wave_metric)
                wave_unit = "m"
            else:
                wave_default = float(wave_metric) / 0.3048
                wave_unit = "ft"
        else:
            wave_default = None
            wave_unit = "m" if entry_units == "metric" else "ft"

        vis_metric = _pick(["min_visibility_km", "min_visibility"], None)
        if vis_metric is not None:
            if entry_units == "metric":
                vis_default = float(vis_metric)
                vis_unit = "km"
            else:
                vis_default = float(vis_metric) / 1.609344
                vis_unit = "mi"
        else:
            vis_default = None
            vis_unit = "km" if entry_units == "metric" else "mi"

        swell_default = _pick(["max_swell_period_s", "max_swell_period", "max_swell_period_s"], None)

        schema = vol.Schema(
            {
                vol.Optional("update_interval", default=current.get("update_interval", DEFAULT_UPDATE_INTERVAL)): cv.positive_int,
                vol.Optional("persist_last_fetch", default=current.get("persist_last_fetch", False)): bool,
                vol.Optional("persist_ttl", default=current.get("persist_ttl", 3600)): cv.positive_int,
                vol.Optional("species", default=current.get("species", "")): cv.string,
                vol.Optional("units", default=entry_units): vol.In(["metric", "imperial"]),
                vol.Optional("safety_max_wind", default=wind_default): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=10, max=100, step=1, unit_of_measurement=wind_unit, mode="slider")
                ),
                vol.Optional("safety_max_wave_height", default=wave_default): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.5, max=10.0, step=0.1, unit_of_measurement=wave_unit, mode="slider")
                ),
                vol.Optional("safety_min_visibility", default=vis_default): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0, max=200, step=1, unit_of_measurement=vis_unit, mode="slider")
                ),
                vol.Optional("safety_max_swell_period", default=swell_default): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=1, max=120, step=1, unit_of_measurement="s", mode="slider")
                ),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema, errors=errors)