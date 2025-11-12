"""Config flow for Ocean Fishing Assistant (strict, ocean-only)."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
import homeassistant.helpers.config_validation as cv

from .const import (
    DOMAIN,
    MODE_OCEAN,
    CONF_NAME,
    CONF_LATITUDE,
    CONF_LONGITUDE,
    CONF_SPECIES_ID,
    CONF_SPECIES_REGION,
    CONF_HABITAT_PRESET,
    CONF_TIME_PERIODS,
    CONF_THRESHOLDS,
    CONF_TIMEZONE,
    CONF_ELEVATION,
    CONF_MODE,
    CONF_AUTO_APPLY_THRESHOLDS,
    CONF_TIDE_MODE,
    CONF_MARINE_ENABLED,
    TIDE_MODE_PROXY,
    HABITAT_PRESETS,
    TIME_PERIODS_FULL_DAY,
    TIME_PERIODS_DAWN_DUSK,
    DEFAULT_NAME,
    HABITAT_ROCKY_POINT,
)

from .species_loader import SpeciesLoader

_LOGGER = logging.getLogger(__name__)


class OceanFishingConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Ocean Fishing Assistant (ocean-only)."""

    VERSION = 1

    def __init__(self) -> None:
        self.ocean_config: dict[str, Any] = {}
        self.species_loader: SpeciesLoader | None = None

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Start the flow by jumping to mode select where the user must pick ocean."""
        if user_input is None:
            return await self.async_step_select_mode()
        # If user provided a direct payload with mode, handle accordingly
        if user_input.get(CONF_MODE) == MODE_OCEAN:
            return await self.async_step_ocean_location()
        # If mode not provided or not ocean, force mode selection page
        return await self.async_step_select_mode()

    async def async_step_select_mode(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Force selection of Ocean mode (explicit)."""
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                if user_input.get(CONF_MODE) == MODE_OCEAN:
                    return await self.async_step_ocean_location()
                errors["base"] = "must_select_ocean"
            except Exception as exc:
                _LOGGER.exception("Error in select_mode: %s", exc)
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="select_mode",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MODE, default=MODE_OCEAN): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": MODE_OCEAN, "label": "🌊 Ocean/Shore Fishing"},
                            ],
                            mode="list",
                        )
                    )
                }
            ),
            errors=errors,
        )

    async def async_step_ocean_location(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure ocean location (name + coordinates)."""
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                lat = float(user_input[CONF_LATITUDE])
                lon = float(user_input[CONF_LONGITUDE])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    errors["base"] = "invalid_coordinates"
            except (ValueError, KeyError):
                errors["base"] = "invalid_coordinates"

            if not errors:
                self.ocean_config.update(user_input)
                return await self.async_step_ocean_species()

        default_name = user_input.get(CONF_NAME, "") if user_input else ""
        default_lat = user_input.get(CONF_LATITUDE, "") if user_input else ""
        default_lon = user_input.get(CONF_LONGITUDE, "") if user_input else ""

        return self.async_show_form(
            step_id="ocean_location",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NAME, default=default_name): str,
                    vol.Required(CONF_LATITUDE, default=default_lat): cv.latitude,
                    vol.Required(CONF_LONGITUDE, default=default_lon): cv.longitude,
                }
            ),
            errors=errors,
        )

    async def async_step_ocean_species(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Choose species/region for ocean mode (strict)."""
        # Ensure loader is available and loaded for species lists
        if self.species_loader is None:
            self.species_loader = SpeciesLoader(self.hass)
            await self.species_loader.async_load_profiles()

        if user_input is not None:
            species_id = user_input[CONF_SPECIES_ID]
            # handle the 'general_mixed_<region>' synthetic selection
            if species_id.startswith("general_mixed_"):
                species_region = species_id.replace("general_mixed_", "")
                species_id = "general_mixed"
            else:
                species_profile = self.species_loader.get_species(species_id)
                if species_profile:
                    available_regions = species_profile.get("regions", ["global"])
                    species_region = available_regions[0]
                else:
                    _LOGGER.error("Selected species_id %s not found in profiles", species_id)
                    raise RuntimeError("Selected species_id not found")

            self.ocean_config[CONF_SPECIES_ID] = species_id
            self.ocean_config[CONF_SPECIES_REGION] = species_region
            return await self.async_step_ocean_habitat()

        # Build options strictly; abort if missing
        if not getattr(self.species_loader, "_profiles", None):
            _LOGGER.error("Species profiles missing when building ocean species list; aborting flow.")
            raise RuntimeError("Missing species profiles for ocean species selection")

        regions = self.species_loader.get_regions_by_type("ocean")
        if not regions:
            _LOGGER.error("No ocean regions found in species_profiles.json; aborting flow.")
            raise RuntimeError("No ocean regions available")

        species_options: list[dict[str, str]] = []

        # SECTION: General regional mixed profiles
        species_options.append({"value": "separator_regions", "label": "━━━━━ 🎣 GENERAL REGION PROFILES ━━━━━"})
        for region in regions:
            region_id = region["id"]
            region_name = region.get("name", region_id)
            species_options.append({"value": f"general_mixed_{region_id}", "label": f"🎣 {region_name} - General Mixed Species"})

        # SECTION: Specific species
        species_options.append({"value": "separator_species", "label": "━━━━━ 🐟 TARGET SPECIFIC SPECIES ━━━━━"})

        # Collect all ocean-specific species across regions (dedupe)
        all_species: list[dict[str, Any]] = []
        for region in regions:
            region_id = region["id"]
            if region_id == "global":
                continue
            region_species = self.species_loader.get_species_by_region(region_id)
            for s in region_species:
                if s.get("habitat") != "ocean":
                    continue
                sid = s["id"]
                if not any(x["id"] == sid for x in all_species):
                    all_species.append(s)

        all_species.sort(key=lambda s: s.get("name", s["id"]))
        for species in all_species:
            emoji = species.get("emoji", "🐟")
            name = species.get("name", species["id"])
            species_id = species["id"]
            active_months = species.get("active_months", [])
            if len(active_months) == 12:
                season_info = "Year-round"
            elif len(active_months) > 0:
                season_info = f"Active: {len(active_months)} months"
            else:
                season_info = ""
            label = f"{emoji} {name}"
            if season_info:
                label += f" ({season_info})"
            species_options.append({"value": species_id, "label": label})

        return self.async_show_form(
            step_id="ocean_species",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_SPECIES_ID): selector.SelectSelector(
                        selector.SelectSelectorConfig(options=species_options, mode="dropdown")
                    )
                }
            ),
            description_placeholders={"info": "Choose a general region profile for mixed species, or target a specific species."},
        )

    async def async_step_ocean_habitat(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Choose habitat preset for ocean mode."""
        if user_input is not None:
            try:
                raw_hp = user_input.get(CONF_HABITAT_PRESET, "")
                habitat_preset = str(raw_hp).strip() if raw_hp is not None else ""
                if not habitat_preset or habitat_preset not in HABITAT_PRESETS:
                    _LOGGER.error("Invalid or missing habitat_preset submitted: %s", habitat_preset)
                    raise ValueError("Invalid habitat_preset")
                self.ocean_config[CONF_HABITAT_PRESET] = habitat_preset
                return await self.async_step_ocean_time_periods()
            except Exception as exc:
                _LOGGER.exception("Unhandled exception in async_step_ocean_habitat: %s", exc)
                return self.async_show_form(
                    step_id="ocean_habitat",
                    data_schema=vol.Schema(
                        {
                            vol.Required(CONF_HABITAT_PRESET, default=HABITAT_ROCKY_POINT): selector.SelectSelector(
                                selector.SelectSelectorConfig(
                                    options=[
                                        {"value": "open_beach", "label": "🏖️ Open Sandy Beach"},
                                        {"value": "rocky_point", "label": "🪨 Rocky Point/Jetty"},
                                        {"value": "harbour", "label": "⚓ Harbour/Pier"},
                                        {"value": "reef", "label": "🪸 Offshore Reef"},
                                    ],
                                    mode="list",
                                )
                            )
                        }
                    ),
                    errors={"base": "unknown"},
                )

        return self.async_show_form(
            step_id="ocean_habitat",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_HABITAT_PRESET, default=HABITAT_ROCKY_POINT): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": "open_beach", "label": "🏖️ Open Sandy Beach"},
                                {"value": "rocky_point", "label": "🪨 Rocky Point/Jetty"},
                                {"value": "harbour", "label": "⚓ Harbour/Pier"},
                                {"value": "reef", "label": "🪸 Offshore Reef"},
                            ],
                            mode="list",
                        )
                    )
                }
            ),
        )

    async def async_step_ocean_time_periods(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Choose time periods for ocean monitoring."""
        if user_input is not None:
            errors: dict[str, str] = {}
            tp = user_input.get(CONF_TIME_PERIODS)
            valid = {TIME_PERIODS_FULL_DAY, TIME_PERIODS_DAWN_DUSK}
            if tp is None or tp not in valid:
                errors["base"] = "invalid_time_periods"
                return self.async_show_form(
                    step_id="ocean_time_periods",
                    data_schema=vol.Schema(
                        {
                            vol.Required(CONF_TIME_PERIODS, default=self.ocean_config.get(CONF_TIME_PERIODS, TIME_PERIODS_FULL_DAY)): selector.SelectSelector(
                                selector.SelectSelectorConfig(
                                    options=[
                                        {"value": TIME_PERIODS_FULL_DAY, "label": "🌅 Full Day (4 periods: Morning, Afternoon, Evening, Night)"},
                                        {"value": TIME_PERIODS_DAWN_DUSK, "label": "🌄 Dawn & Dusk Only (Prime fishing times: ±1hr sunrise/sunset)"},
                                    ],
                                    mode="list",
                                )
                            )
                        }
                    ),
                    errors=errors,
                    description_placeholders={"info": "Choose which time periods to monitor. Dawn & Dusk focuses on the most productive fishing times."},
                )

            self.ocean_config.update(user_input)
            return await self.async_step_ocean_thresholds()

        return self.async_show_form(
            step_id="ocean_time_periods",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_TIME_PERIODS, default=TIME_PERIODS_FULL_DAY): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": TIME_PERIODS_FULL_DAY, "label": "🌅 Full Day (4 periods: Morning, Afternoon, Evening, Night)"},
                                {"value": TIME_PERIODS_DAWN_DUSK, "label": "🌄 Dawn & Dusk Only (Prime fishing times: ±1hr sunrise/sunset)"},
                            ],
                            mode="list",
                        )
                    )
                }
            ),
            description_placeholders={"info": "Choose which time periods to monitor. Dawn & Dusk focuses on the most productive fishing times."},
        )

    async def async_step_ocean_thresholds(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure thresholds and finish ocean config (strict)."""
        if user_input is not None:
            try:
                # Ensure coordinates are present and valid
                if CONF_LATITUDE not in self.ocean_config or CONF_LONGITUDE not in self.ocean_config:
                    _LOGGER.error("Latitude/Longitude missing from ocean_config; aborting to surface the issue.")
                    raise RuntimeError("Missing latitude/longitude in ocean_config")

                lat_raw = self.ocean_config[CONF_LATITUDE]
                lon_raw = self.ocean_config[CONF_LONGITUDE]
                latitude = float(lat_raw)
                longitude = float(lon_raw)
                if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    _LOGGER.error("Latitude/longitude out of valid ranges: lat=%s lon=%s", latitude, longitude)
                    raise ValueError("Latitude/longitude out of range")

                habitat_preset = self.ocean_config.get(CONF_HABITAT_PRESET, HABITAT_ROCKY_POINT)

                final_config = {
                    CONF_MODE: MODE_OCEAN,
                    CONF_NAME: self.ocean_config.get(CONF_NAME, DEFAULT_NAME),
                    CONF_LATITUDE: latitude,
                    CONF_LONGITUDE: longitude,
                    CONF_SPECIES_ID: self.ocean_config.get(CONF_SPECIES_ID, "general_mixed"),
                    CONF_SPECIES_REGION: self.ocean_config.get(CONF_SPECIES_REGION, "global"),
                    CONF_HABITAT_PRESET: habitat_preset,
                    CONF_TIME_PERIODS: self.ocean_config.get(CONF_TIME_PERIODS, TIME_PERIODS_FULL_DAY),
                    CONF_AUTO_APPLY_THRESHOLDS: False,
                    CONF_TIDE_MODE: TIDE_MODE_PROXY,
                    CONF_MARINE_ENABLED: True,
                    CONF_THRESHOLDS: {
                        "max_wind_speed": user_input["max_wind_speed"],
                        "max_gust_speed": user_input["max_gust_speed"],
                        "max_wave_height": user_input["max_wave_height"],
                        "min_temperature": user_input["min_temperature"],
                        "max_temperature": user_input["max_temperature"],
                    },
                }

                final_config[CONF_TIMEZONE] = str(self.hass.config.time_zone)
                final_config[CONF_ELEVATION] = self.hass.config.elevation

                _LOGGER.debug("Creating ocean config entry with data keys: %s", list(final_config.keys()))
                return self.async_create_entry(title=final_config[CONF_NAME], data=final_config)
            except KeyError as ke:
                _LOGGER.exception("Missing expected key when building final ocean config: %s", ke)
                return self._show_ocean_thresholds_form(errors={"base": "unknown"})
            except Exception as exc:
                _LOGGER.exception("Unhandled exception in async_step_ocean_thresholds: %s", exc)
                return self._show_ocean_thresholds_form(errors={"base": "unknown"})

        # Show the form with defaults based on habitat preset
        return self._show_ocean_thresholds_form()

    def _show_ocean_thresholds_form(self, errors: dict[str, str] | None = None) -> FlowResult:
        habitat = HABITAT_PRESETS.get(self.ocean_config.get(CONF_HABITAT_PRESET, HABITAT_ROCKY_POINT), HABITAT_PRESETS.get(HABITAT_ROCKY_POINT, {}))
        return self.async_show_form(
            step_id="ocean_thresholds",
            data_schema=vol.Schema(
                {
                    vol.Required("max_wind_speed", default=habitat.get("max_wind_speed", 25)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=10, max=50, step=5, unit_of_measurement="km/h", mode="slider")
                    ),
                    vol.Required("max_gust_speed", default=habitat.get("max_gust_speed", 40)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=15, max=70, step=5, unit_of_measurement="km/h", mode="slider")
                    ),
                    vol.Required("max_wave_height", default=habitat.get("max_wave_height", 2.0)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0.5, max=5.0, step=0.5, unit_of_measurement="m", mode="slider")
                    ),
                    vol.Required("min_temperature", default=5): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=-10, max=20, step=1, unit_of_measurement="°C")
                    ),
                    vol.Required("max_temperature", default=35): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=20, max=50, step=1, unit_of_measurement="°C")
                    ),
                }
            ),
            errors=errors or {},
            description_placeholders={"info": "Set safe fishing limits based on your habitat and comfort level."},
        )

    # Options flow (simple)
    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry) -> config_entries.OptionsFlow:
        return OptionsFlowHandler(config_entry)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for Ocean Fishing Assistant."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        return await self.async_step_ocean_options()

    async def async_step_ocean_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        thresholds = self.config_entry.data.get(CONF_THRESHOLDS, {})
        return self.async_show_form(
            step_id="ocean_options",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_TIME_PERIODS, default=self.config_entry.data.get(CONF_TIME_PERIODS, TIME_PERIODS_FULL_DAY)): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": TIME_PERIODS_FULL_DAY, "label": "🌅 Full Day (4 periods)"},
                                {"value": TIME_PERIODS_DAWN_DUSK, "label": "🌄 Dawn & Dusk Only"},
                            ],
                            mode="dropdown",
                        )
                    ),
                    vol.Required("max_wind_speed", default=thresholds.get("max_wind_speed", 25)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=10, max=50, step=5, unit_of_measurement="km/h", mode="slider")
                    ),
                    vol.Required("max_wave_height", default=thresholds.get("max_wave_height", 2.0)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0.5, max=5.0, step=0.5, unit_of_measurement="m", mode="slider")
                    ),
                }
            ),
        )