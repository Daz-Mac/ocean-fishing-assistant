"""Config flow for Ocean Fishing Assistant"""

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
from .unit_helpers import convert_safety_display_to_metric, validate_and_normalize_safety_limits

_LOGGER = logging.getLogger(__name__)


class OceanFishingConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Ocean Fishing Assistant."""

    VERSION = 1

    def __init__(self) -> None:
        self.ocean_config: dict[str, Any] = {}
        self.species_loader: SpeciesLoader | None = None

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Start the ocean-only flow; forward to ocean_location step."""
        return await self.async_step_ocean_location(user_input)

    # ----
    # Ocean location
    # ----
    async def async_step_ocean_location(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure ocean location (name and coordinates)."""
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

    # ----
    # Ocean species/region selection
    # ----
    async def async_step_ocean_species(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Choose species/region for ocean mode (strict)."""
        # Ensure loader is available and loaded for species lists
        if self.species_loader is None:
            self.species_loader = SpeciesLoader(self.hass)
            await self.species_loader.async_load_profiles()

        if user_input is not None:
            species_id = user_input[CONF_SPECIES_ID]

            # If the user picked a general profile id, resolve its region(s)
            general_profile = self.species_loader.get_general_profile(species_id)
            if general_profile:
                # use the first available region for the selected general profile
                available_regions = general_profile.get("regions", ["global"])
                species_region = available_regions[0] if available_regions else "global"
            else:
                # otherwise treat as a specific species id and resolve its first region
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

        # SECTION: General regional mixed profiles (sourced from general_profiles)
        species_options.append({"value": "separator_regions", "label": "━━━━ 🎣 GENERAL REGION PROFILES ━━━━"})
        # get all general profiles and include only those whose region is ocean
        general_profiles = self.species_loader.get_general_profiles()
        # present general profiles in order by friendly name (common_name or id)
        general_profiles.sort(key=lambda g: g.get("common_name", g.get("id", "")))
        for gp in general_profiles:
            # ensure it belongs to an ocean region (or has habitat ocean)
            gp_habitat = gp.get("habitat")
            # fallback to checking the first region's habitat via region metadata
            if gp_habitat != "ocean":
                # attempt to detect via region info if available
                gp_regions = gp.get("regions", []) or []
                if gp_regions:
                    # if any region is ocean include it
                    include = False
                    for r in gp_regions:
                        rinfo = self.species_loader.get_region_info(r)
                        if rinfo and rinfo.get("habitat") == "ocean":
                            include = True
                            break
                    if not include:
                        continue
                else:
                    # no explicit habitat/regions — skip
                    continue
            gid = gp.get("id")
            gname = gp.get("common_name", gid)
            emoji = gp.get("emoji", "🎣")
            label = f"{emoji} {gname}"
            species_options.append({"value": gid, "label": label})

        # SECTION: Specific species
        species_options.append({"value": "separator_species", "label": "━━━━ 🐟 TARGET SPECIFIC SPECIES ━━━━"})

        # Collect all ocean-specific species across regions (dedupe, include 'global')
        all_species: list[dict[str, Any]] = []
        for region in regions:
            region_id = region["id"]
            region_species = self.species_loader.get_species_by_region(region_id)
            for s in region_species:
                if s.get("habitat") != "ocean":
                    continue
                sid = s["id"]
                if not any(x["id"] == sid for x in all_species):
                    all_species.append(s)

        all_species.sort(key=lambda s: s.get("common_name", s.get("id")))
        for species in all_species:
            emoji = species.get("emoji", "🐟")
            name = species.get("common_name", species["id"])
            species_id = species["id"]
            active_months = species.get("active_months", []) or species.get("preferred_months", [])
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

    # ----
    # Habitat selection
    # ----
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

    # ----
    # Time periods
    # ----
    async def async_step_ocean_time_periods(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Choose time periods for ocean monitoring."""
        if user_input is not None:
            errors: dict[str, str] = {}
            tp = user_input.get(CONF_TIME_PERIODS)
            valid = {TIME_PERIODS_FULL_DAY, TIME_PERIODS_DAWN_DUSK}
            if tp is None or tp not in valid:
                errors["base"] = "invalid_time_periods"
            if errors:
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
            # Ask units next so that thresholds form can render correct unit labels.
            return await self.async_step_ocean_units()

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

    # ----
    # Units selection (NEW) - must be chosen before thresholds are rendered
    # ----
    async def async_step_ocean_units(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Ask user which display units they want (metric/imperial)."""
        if user_input is not None:
            units = user_input.get("units")
            if units not in ("metric", "imperial"):
                return self.async_show_form(
                    step_id="ocean_units",
                    data_schema=vol.Schema({vol.Required("units", default="metric"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": "metric", "label": "Metric (m, km/h, °C)"},
                                {"value": "imperial", "label": "Imperial (ft, mph, °F)"},
                            ],
                            mode="dropdown",
                        )
                    )}),
                    errors={"base": "invalid_units"},
                )
            self.ocean_config["units"] = units
            return await self.async_step_ocean_thresholds()

        return self.async_show_form(
            step_id="ocean_units",
            data_schema=vol.Schema(
                {
                    vol.Required("units", default="metric"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": "metric", "label": "Metric (m, km/h, °C)"},
                                {"value": "imperial", "label": "Imperial (ft, mph, °F)"},
                            ],
                            mode="dropdown",
                        )
                    )
                }
            ),
        )

    # ----
    # Thresholds & finish
    # ----
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
                units = self.ocean_config.get("units", "metric")
                wind_unit = "km/h" if units == "metric" else "mph"

                # Build a display->canonical safety dict then convert -> validate/normalize
                safety_display = {
                    "safety_max_wind": user_input["max_wind_speed"],
                    "safety_max_gust": user_input.get("max_gust_speed"),
                    "safety_max_wave_height": user_input["max_wave_height"],
                    "safety_min_visibility": user_input.get("min_visibility"),
                    "safety_min_swell_period": user_input.get("min_swell_period"),
                    # NEW: precipitation chance display key (percent)
                    "safety_max_precip_chance": user_input.get("max_precip_chance"),
                }
                canonical = convert_safety_display_to_metric(safety_display, entry_units=units)
                normalized_limits, warnings = validate_and_normalize_safety_limits(canonical, strict=True)

                # Build canonical safety_limits to store
                safety_limits = normalized_limits

                final_config = {
                    # integration is ocean-only
                    CONF_NAME: self.ocean_config.get(CONF_NAME, DEFAULT_NAME),
                    CONF_LATITUDE: latitude,
                    CONF_LONGITUDE: longitude,
                    CONF_SPECIES_ID: self.ocean_config.get(CONF_SPECIES_ID, "general_mixed_global"),
                    CONF_SPECIES_REGION: self.ocean_config.get(CONF_SPECIES_REGION, "global"),
                    CONF_HABITAT_PRESET: habitat_preset,
                    CONF_TIME_PERIODS: self.ocean_config.get(CONF_TIME_PERIODS, TIME_PERIODS_FULL_DAY),
                    CONF_AUTO_APPLY_THRESHOLDS: False,
                    CONF_TIDE_MODE: TIDE_MODE_PROXY,
                    CONF_MARINE_ENABLED: True,
                    # store both user thresholds (for UI/options) and canonical safety_limits (for runtime)
                    CONF_THRESHOLDS: {
                        "max_wind_speed": user_input["max_wind_speed"],
                        "max_gust_speed": user_input.get("max_gust_speed"),
                        "max_wave_height": user_input["max_wave_height"],
                        "min_visibility": user_input.get("min_visibility"),
                        "min_temperature": user_input.get("min_temperature"),
                        "max_temperature": user_input.get("max_temperature"),
                        "min_swell_period": user_input.get("min_swell_period"),
                        # NEW: include precip chance in stored thresholds for options/UI
                        "max_precip_chance": user_input.get("max_precip_chance"),
                        # Preserve user selection here for visibility; async_setup_entry will populate entry.options
                        "expose_raw": bool(user_input.get("expose_raw", False)),
                    },
                    # Strict runtime keys required by async_setup_entry
                    "units": units,
                    "wind_unit": wind_unit,
                    "safety_limits": safety_limits,
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

        return self._show_ocean_thresholds_form()

    def _show_ocean_thresholds_form(self, errors: dict[str, str] | None = None) -> FlowResult:
        habitat = HABITAT_PRESETS.get(self.ocean_config.get(CONF_HABITAT_PRESET, HABITAT_ROCKY_POINT), HABITAT_PRESETS.get(HABITAT_ROCKY_POINT, {}))
        units = self.ocean_config.get("units", "metric")
        # map display units for selector labels
        wind_unit_label = "km/h" if units == "metric" else "mph"
        wave_unit_label = "m" if units == "metric" else "ft"
        temp_unit_label = "°C" if units == "metric" else "°F"
        vis_unit_label = "km" if units == "metric" else "miles"

        return self.async_show_form(
            step_id="ocean_thresholds",
            data_schema=vol.Schema(
                {
                    vol.Required("max_wind_speed", default=habitat.get("max_wind_speed", 25)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=10, max=50, step=5, unit_of_measurement=wind_unit_label, mode="slider")
                    ),
                    vol.Required("max_gust_speed", default=habitat.get("max_gust_speed", 40)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=15, max=80, step=5, unit_of_measurement=wind_unit_label, mode="slider")
                    ),
                    vol.Required("max_wave_height", default=habitat.get("max_wave_height", 2.0)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0.5, max=10.0, step=0.5, unit_of_measurement=wave_unit_label, mode="slider")
                    ),
                    # NEW: maximum precipitation chance (percentage)
                    vol.Required("max_precip_chance", default=habitat.get("max_precip_chance", 80)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0, max=100, step=5, unit_of_measurement="%", mode="slider")
                    ),
                    vol.Required("min_swell_period", default=3): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0, max=30, step=1, unit_of_measurement="s")
                    ),
                    vol.Required("min_visibility", default=1): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0, max=50, step=1, unit_of_measurement=vis_unit_label)
                    ),
                    vol.Required("min_temperature", default=5): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=-30, max=50, step=1, unit_of_measurement=temp_unit_label)
                    ),
                    vol.Required("max_temperature", default=35): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=-10, max=122, step=1, unit_of_measurement=temp_unit_label)
                    ),
                    # RESTORE: expose_raw option at setup time
                    vol.Required("expose_raw", default=habitat.get("expose_raw", False)): selector.BooleanSelector(
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
        # Do NOT assign to `self.config_entry` (deprecated). Use a private attribute instead.
        self._config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        return await self.async_step_ocean_options()

    async def async_step_ocean_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        thresholds = self._config_entry.data.get(CONF_THRESHOLDS, {})
        # show units-driven labels based on stored units in config_entry.data
        units = self._config_entry.data.get("units", "metric")
        wind_unit_label = "km/h" if units == "metric" else "mph"
        wave_unit_label = "m" if units == "metric" else "ft"
        vis_unit_label = "km" if units == "metric" else "miles"

        return self.async_show_form(
            step_id="ocean_options",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_TIME_PERIODS, default=self._config_entry.data.get(CONF_TIME_PERIODS, TIME_PERIODS_FULL_DAY)): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": TIME_PERIODS_FULL_DAY, "label": "🌅 Full Day (4 periods)"},
                                {"value": TIME_PERIODS_DAWN_DUSK, "label": "🌄 Dawn & Dusk Only"},
                            ],
                            mode="dropdown",
                        )
                    ),
                    vol.Required("max_wind_speed", default=thresholds.get("max_wind_speed", 25)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=10, max=50, step=5, unit_of_measurement=wind_unit_label, mode="slider")
                    ),
                    vol.Required("max_gust_speed", default=thresholds.get("max_gust_speed", 40)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=15, max=80, step=5, unit_of_measurement=wind_unit_label, mode="slider")
                    ),
                    vol.Required("max_wave_height", default=thresholds.get("max_wave_height", 2.0)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0.5, max=10.0, step=0.5, unit_of_measurement=wave_unit_label, mode="slider")
                    ),
                    # NEW: options flow exposure for precipitation chance
                    vol.Required("max_precip_chance", default=thresholds.get("max_precip_chance", 80)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0, max=100, step=5, unit_of_measurement="%", mode="slider")
                    ),
                    vol.Required("min_swell_period", default=thresholds.get("min_swell_period", 3)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0, max=30, step=1, unit_of_measurement="s")
                    ),
                    vol.Required("min_visibility", default=thresholds.get("min_visibility", 1)): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0, max=50, step=1, unit_of_measurement=vis_unit_label, mode="slider")
                    ),
                    # RESTORE: expose_raw option in Options flow
                    vol.Required("expose_raw", default=thresholds.get("expose_raw", False)): selector.BooleanSelector(
                    ),                    
                }
            ),
        )