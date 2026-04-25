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
    CONF_FACTOR_WEIGHTS,
    CONF_TIMEZONE,
    CONF_ELEVATION,
    HABITAT_PRESETS,
    TIME_PERIODS_FULL_DAY,
    TIME_PERIODS_DAWN_DUSK,
    DEFAULT_NAME,
    HABITAT_ROCKY_POINT,
    DEFAULT_UPDATE_INTERVAL,
    FETCH_CACHE_TTL,
    CONF_FETCH_CACHE_TTL,
    CONF_TIDE_TTL,
    CONF_WEATHER_CACHE_TTL,
    TIDE_PROXY_TTL_DEFAULT,
    WEATHER_FETCHER_CACHE_TTL_DEFAULT,
    # new tide phase offset constants
    CONF_TIDE_PHASE_OFFSET_MINUTES,
    TIDE_PHASE_OFFSET_MINUTES_DEFAULT,
    # World Tides API key constant
    CONF_WORLD_TIDES_API_KEY,
    # wind direction constants
    CONF_PREFERRED_WIND_DIRECTIONS,
    WIND_DIRECTIONS,
)

from .species_loader import SpeciesLoader
from .unit_helpers import convert_safety_display_to_metric, validate_and_normalize_safety_limits
from .ocean_scoring import FACTOR_WEIGHTS, _validate_and_normalize_factor_weights

_LOGGER = logging.getLogger(__name__)


class OceanFishingConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Ocean Fishing Assistant."""

    VERSION = 1

    def __init__(self) -> None:
        self.ocean_config: dict[str, Any] = {}
        self.species_loader: SpeciesLoader | None = None

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Start the flow; forward to location step."""
        return await self.async_step_ocean_location(user_input)

    # ----
    # Ocean location (mode, name, coordinates, World Tides API key)
    # ----
    async def async_step_ocean_location(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure ocean location (mode, name, coordinates) and capture the World Tides API key (required)."""
        errors: dict[str, str] = {}
        if user_input is not None:
            # Validate coordinates and title
            try:
                lat = float(user_input[CONF_LATITUDE])
                lon = float(user_input[CONF_LONGITUDE])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    errors["base"] = "invalid_coordinates"
            except (ValueError, KeyError):
                errors["base"] = "invalid_coordinates"

            # Validate API key presence (required once per integration)
            try:
                api_key_submitted = str(user_input.get(CONF_WORLD_TIDES_API_KEY, "")).strip()
                if not api_key_submitted:
                    errors["base"] = "missing_world_tides_api_key"
            except Exception:
                errors["base"] = "missing_world_tides_api_key"

            # Duplicate title check
            if not errors:
                submitted_title = str(user_input.get(CONF_NAME, "")).strip()
                if submitted_title:
                    existing_entries = self.hass.config_entries.async_entries(DOMAIN)
                    for e in existing_entries:
                        if e.title == submitted_title:
                            _LOGGER.debug(
                                "Attempt to create entry with duplicate title '%s' rejected at location step",
                                submitted_title,
                            )
                            errors["base"] = "title_exists"
                            break

            if not errors:
                # Save basic values including the API key
                self.ocean_config.update(user_input)
                # Default setup_mode -> normal unless explicitly provided
                self.ocean_config.setdefault("setup_mode", "normal")
                if self.ocean_config.get("setup_mode") == "advanced":
                    return await self.async_step_advanced_config()
                return await self.async_step_ocean_species()

        default_name = user_input.get(CONF_NAME, "") if user_input else ""
        default_lat = user_input.get(CONF_LATITUDE, "") if user_input else ""
        default_lon = user_input.get(CONF_LONGITUDE, "") if user_input else ""
        default_mode = user_input.get("setup_mode", "normal") if user_input else "normal"
        default_world_key = user_input.get(CONF_WORLD_TIDES_API_KEY, "") if user_input else ""

        return self.async_show_form(
            step_id="ocean_location",
            data_schema=vol.Schema(
                {
                    vol.Required("setup_mode", default=default_mode): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": "normal", "label": "Normal mode (simple setup)"},
                                {"value": "advanced", "label": "Advanced mode (intervals, TTLs, factor weights)"},
                            ],
                            mode="list",
                        )
                    ),
                    vol.Required(CONF_NAME, default=default_name): str,
                    vol.Required(CONF_LATITUDE, default=default_lat): cv.latitude,
                    vol.Required(CONF_LONGITUDE, default=default_lon): cv.longitude,
                    vol.Required(CONF_WORLD_TIDES_API_KEY, default=default_world_key): selector.TextSelector(),
                }
            ),
            errors=errors,
        )

    # ----
    # Advanced configuration step (only when user picks advanced)
    # ----
    async def async_step_advanced_config(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Advanced options: update_interval, TTLs, expose_raw. Factor weights live on their own screen."""
        default_update_interval = self.ocean_config.get("update_interval", DEFAULT_UPDATE_INTERVAL)
        default_expose_raw = self.ocean_config.get("expose_raw", False)
        default_fetch_cache_ttl = self.ocean_config.get(CONF_FETCH_CACHE_TTL, FETCH_CACHE_TTL)
        default_tide_ttl = self.ocean_config.get(CONF_TIDE_TTL, TIDE_PROXY_TTL_DEFAULT)
        default_weather_cache_ttl = self.ocean_config.get(CONF_WEATHER_CACHE_TTL, WEATHER_FETCHER_CACHE_TTL_DEFAULT)
        default_phase_offset = int(self.ocean_config.get(CONF_TIDE_PHASE_OFFSET_MINUTES, TIDE_PHASE_OFFSET_MINUTES_DEFAULT))

        if user_input is not None:
            errors: dict[str, str] = {}
            try:
                ui_update_interval = int(user_input.get("update_interval", default_update_interval))
                if ui_update_interval < 30:
                    errors["base"] = "update_interval_too_small"
                ui_expose_raw = bool(user_input.get("expose_raw", default_expose_raw))
                ui_fetch_ttl = int(user_input.get(CONF_FETCH_CACHE_TTL, default_fetch_cache_ttl))
                ui_tide_ttl = int(user_input.get(CONF_TIDE_TTL, default_tide_ttl))
                ui_weather_ttl = int(user_input.get(CONF_WEATHER_CACHE_TTL, default_weather_cache_ttl))
                ui_phase_offset = int(user_input.get(CONF_TIDE_PHASE_OFFSET_MINUTES, default_phase_offset))
                if ui_fetch_ttl < 30 or ui_tide_ttl < 10 or ui_weather_ttl < 30:
                    errors["base"] = "ttl_too_small"
                if ui_phase_offset < -180 or ui_phase_offset > 180:
                    errors["base"] = "phase_offset_out_of_range"
            except Exception:
                errors["base"] = "invalid_advanced_values"

            if errors:
                return self.async_show_form(
                    step_id="advanced_config",
                    data_schema=vol.Schema(
                        {
                            vol.Required("update_interval", default=default_update_interval): selector.NumberSelector(
                                selector.NumberSelectorConfig(min=30, max=86400, step=30, unit_of_measurement="s")
                            ),
                            vol.Required("expose_raw", default=default_expose_raw): selector.BooleanSelector(),
                            vol.Required(CONF_FETCH_CACHE_TTL, default=default_fetch_cache_ttl): selector.NumberSelector(
                                selector.NumberSelectorConfig(min=30, max=86400, step=30, unit_of_measurement="s")
                            ),
                            vol.Required(CONF_TIDE_TTL, default=default_tide_ttl): selector.NumberSelector(
                                selector.NumberSelectorConfig(min=10, max=86400, step=10, unit_of_measurement="s")
                            ),
                            vol.Required(CONF_WEATHER_CACHE_TTL, default=default_weather_cache_ttl): selector.NumberSelector(
                                selector.NumberSelectorConfig(min=30, max=86400, step=30, unit_of_measurement="s")
                            ),
                            vol.Required(CONF_TIDE_PHASE_OFFSET_MINUTES, default=default_phase_offset): selector.NumberSelector(
                                selector.NumberSelectorConfig(min=-180, max=180, step=1, unit_of_measurement="min", mode="slider")
                            ),
                        }
                    ),
                    errors=errors,
                    description_placeholders={"info": "Configure advanced options: how often to fetch and cache TTLs, and a local tide phase offset (minutes)."},
                )

            # Save advanced options
            self.ocean_config["update_interval"] = ui_update_interval
            self.ocean_config["expose_raw"] = ui_expose_raw
            self.ocean_config[CONF_FETCH_CACHE_TTL] = ui_fetch_ttl
            self.ocean_config[CONF_TIDE_TTL] = ui_tide_ttl
            self.ocean_config[CONF_WEATHER_CACHE_TTL] = ui_weather_ttl
            self.ocean_config[CONF_TIDE_PHASE_OFFSET_MINUTES] = ui_phase_offset

            # After advanced options, go to factor weights step (separate screen)
            return await self.async_step_factor_weights()

        # First-time show advanced form (no factor sliders here)
        return self.async_show_form(
            step_id="advanced_config",
            data_schema=vol.Schema(
                {
                    vol.Required("update_interval", default=default_update_interval): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=30, max=86400, step=30, unit_of_measurement="s")
                    ),
                    vol.Required("expose_raw", default=default_expose_raw): selector.BooleanSelector(),
                    vol.Required(CONF_FETCH_CACHE_TTL, default=default_fetch_cache_ttl): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=30, max=86400, step=30, unit_of_measurement="s")
                    ),
                    vol.Required(CONF_TIDE_TTL, default=default_tide_ttl): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=10, max=86400, step=10, unit_of_measurement="s")
                    ),
                    vol.Required(CONF_WEATHER_CACHE_TTL, default=default_weather_cache_ttl): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=30, max=86400, step=30, unit_of_measurement="s")
                    ),
                    vol.Required(CONF_TIDE_PHASE_OFFSET_MINUTES, default=default_phase_offset): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=-180, max=180, step=1, unit_of_measurement="min", mode="slider")
                    ),
                }
            ),
            description_placeholders={"info": "Configure advanced options: how often to fetch and cache TTLs, and a local tide phase offset (minutes)."},
        )

    # ----
    # Factor weights (own screen)
    # ----
    async def async_step_factor_weights(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Show scoring factor sliders on their own step and normalize them on submit."""
        existing_weights = self.ocean_config.get(CONF_FACTOR_WEIGHTS)
        try:
            normalized_defaults = (
                existing_weights
                if isinstance(existing_weights, dict)
                else _validate_and_normalize_factor_weights(None)
            )
        except Exception:
            normalized_defaults = _validate_and_normalize_factor_weights(None)

        factor_defaults_percent: dict[str, int] = {
            k: int(round((normalized_defaults.get(k, 0.0) * 100))) for k in FACTOR_WEIGHTS.keys()
        }
        total_default = sum(factor_defaults_percent.values())

        if user_input is not None:
            try:
                ui_weights_raw: dict[str, float] = {}
                for k in FACTOR_WEIGHTS.keys():
                    key_name = f"factor_{k}"
                    val = user_input.get(key_name, factor_defaults_percent.get(k, 0))
                    try:
                        fv = float(val)
                    except Exception:
                        fv = 0.0
                    ui_weights_raw[k] = fv

                total = float(sum(ui_weights_raw.values()))

                if abs(total - 100.0) > 0.5:
                    schema_fields: dict = {
                        vol.Required("_factors_total", default=int(round(total))): selector.NumberSelector(
                            selector.NumberSelectorConfig(min=0, max=1000, step=1, unit_of_measurement="%", mode="box")
                        )
                    }
                    for k in FACTOR_WEIGHTS.keys():
                        schema_fields[vol.Required(f"factor_{k}", default=int(round(ui_weights_raw.get(k, 0))))] = selector.NumberSelector(
                            selector.NumberSelectorConfig(min=0, max=100, step=1, unit_of_measurement="%", mode="slider")
                        )

                    return self.async_show_form(
                        step_id="factor_weights",
                        data_schema=vol.Schema(schema_fields),
                        errors={"base": "sum_not_100"},
                        description_placeholders={
                            "info": f"Adjust scoring weights. Current total is {total:.1f}%. Values must add to 100%."
                        },
                    )

                normalized_weights = {k: ui_weights_raw[k] / 100.0 for k in ui_weights_raw.keys()}
                normalized_weights = _validate_and_normalize_factor_weights(normalized_weights)

                self.ocean_config[CONF_FACTOR_WEIGHTS] = normalized_weights

                return await self.async_step_ocean_species()
            except ValueError as ve:
                _LOGGER.debug("Factor weights validation failed: %s", ve)
                schema_fields: dict = {
                    vol.Required("_factors_total", default=total_default): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0, max=1000, step=1, unit_of_measurement="%", mode="box")
                    )
                }
                for k in FACTOR_WEIGHTS.keys():
                    schema_fields[vol.Required(f"factor_{k}", default=factor_defaults_percent.get(k, 0))] = selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0, max=100, step=1, unit_of_measurement="%", mode="slider")
                    )

                return self.async_show_form(
                    step_id="factor_weights",
                    data_schema=vol.Schema(schema_fields),
                    errors={"base": "invalid_factor_weights"},
                    description_placeholders={
                        "info": f"Adjust scoring weights. Current default total is {total_default}%. Values must add to 100%."
                    },
                )
            except Exception as exc:
                _LOGGER.exception("Unhandled exception in factor_weights: %s", exc)
                schema_fields: dict = {
                    vol.Required("_factors_total", default=total_default): selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0, max=1000, step=1, unit_of_measurement="%", mode="box")
                    )
                }
                for k in FACTOR_WEIGHTS.keys():
                    schema_fields[vol.Required(f"factor_{k}", default=factor_defaults_percent.get(k, 0))] = selector.NumberSelector(
                        selector.NumberSelectorConfig(min=0, max=100, step=1, unit_of_measurement="%", mode="slider")
                    )
                return self.async_show_form(
                    step_id="factor_weights",
                    data_schema=vol.Schema(schema_fields),
                    errors={"base": "unknown"},
                    description_placeholders={
                        "info": f"Adjust scoring weights. Current default total is {total_default}%. Values must add to 100%."
                    },
                )

        # Build the default schema for the factor weights form
        schema_fields: dict = {}
        schema_fields[vol.Required("_factors_total", default=total_default)] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, max=1000, step=1, unit_of_measurement="%", mode="box")
        )

        for k in FACTOR_WEIGHTS.keys():
            schema_fields[vol.Required(f"factor_{k}", default=factor_defaults_percent.get(k, 0))] = selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=100, step=1, unit_of_measurement="%", mode="slider")
            )

        return self.async_show_form(
            step_id="factor_weights",
            data_schema=vol.Schema(schema_fields),
            description_placeholders={
                "info": f"Adjust scoring weights. Current total is {total_default}%. Values must add to 100%."
            },
        )

    # ----
    # Select a general profile
    # ----
    async def async_step_ocean_species(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Ask whether the user wants a general profile or a species profile."""
        if self.species_loader is None:
            self.species_loader = SpeciesLoader(self.hass)
            await self.species_loader.async_load_profiles()

        if user_input is not None:
            choice = user_input.get("profile_type")
            if choice == "general":
                return await self.async_step_select_general_profile()
            elif choice == "species":
                return await self.async_step_select_region()
            else:
                return self.async_show_form(
                    step_id="ocean_species",
                    data_schema=vol.Schema(
                        {
                            vol.Required("profile_type"): selector.SelectSelector(
                                selector.SelectSelectorConfig(
                                    options=[
                                        {"value": "general", "label": "General / mixed profile"},
                                        {"value": "species", "label": "Target a specific species"},
                                    ],
                                    mode="list",
                                )
                            )
                        }
                    ),
                    errors={"base": "invalid_selection"},
                )

        return self.async_show_form(
            step_id="ocean_species",
            data_schema=vol.Schema(
                {
                    vol.Required("profile_type"): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": "general", "label": "General / mixed profile"},
                                {"value": "species", "label": "Target a specific species"},
                            ],
                            mode="list",
                        )
                    )
                }
            ),
            description_placeholders={"info": "Choose whether to use a general regional profile or target a specific species."},
        )

    # ----
    # Select General Profile
    # ----
    async def async_step_select_general_profile(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Show list of general profiles for the user to choose from."""
        if self.species_loader is None:
            self.species_loader = SpeciesLoader(self.hass)
            await self.species_loader.async_load_profiles()

        if user_input is not None:
            profile_id = user_input.get(CONF_SPECIES_ID)
            gp = self.species_loader.get_general_profile(profile_id)
            if not gp:
                _LOGGER.error("Selected general profile id %s not found", profile_id)
                return self.async_abort(reason="profile_not_found")
            regions = gp.get("regions", ["global"]) or ["global"]
            region = regions[0]
            self.ocean_config[CONF_SPECIES_ID] = profile_id
            self.ocean_config[CONF_SPECIES_REGION] = region
            return await self.async_step_ocean_habitat()

        general_profiles = self.species_loader.get_general_profiles()
        if not general_profiles:
            _LOGGER.error("No general profiles available in species_profiles.json")
            return self.async_abort(reason="no_general_profiles")

        general_profiles.sort(key=lambda g: g.get("common_name", g.get("id", "")))
        options = []
        for gp in general_profiles:
            gid = gp.get("id")
            gname = gp.get("common_name", gid)
            emoji = gp.get("emoji", "🎣")
            options.append({"value": gid, "label": f"{emoji} {gname}"})

        return self.async_show_form(
            step_id="select_general_profile",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_SPECIES_ID): selector.SelectSelector(
                        selector.SelectSelectorConfig(options=options, mode="dropdown")
                    )
                }
            ),
            description_placeholders={"info": "Choose a mixed/general regional profile."},
        )

    # ---- Remaining steps unchanged (habitat, time periods, units, thresholds) ----
    # (The rest of this file has not been changed except for ensuring final_config stores the new CONF_TIDE_PHASE_OFFSET_MINUTES and unit strings)
    async def async_step_select_region(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Let the user pick a region first, then show species available in that region."""
        if self.species_loader is None:
            self.species_loader = SpeciesLoader(self.hass)
            await self.species_loader.async_load_profiles()

        regions = self.species_loader.get_regions()
        if not regions:
            _LOGGER.error("No regions present in species_profiles.json")
            return self.async_abort(reason="no_regions")

        if user_input is not None:
            region_id = user_input.get(CONF_SPECIES_REGION)
            if not any(r.get("id") == region_id for r in regions):
                _LOGGER.error("Selected region %s not valid", region_id)
                return self.async_show_form(
                    step_id="select_region",
                    data_schema=vol.Schema(
                        {
                            vol.Required(CONF_SPECIES_REGION): selector.SelectSelector(
                                selector.SelectSelectorConfig(
                                    options=[{"value": r["id"], "label": r.get("name", r["id"])} for r in regions],
                                    mode="dropdown",
                                )
                            )
                        }
                    ),
                    errors={"base": "invalid_region"},
                )

            self.ocean_config[CONF_SPECIES_REGION] = region_id
            return await self.async_step_select_species_for_region()

        region_options = [{"value": r["id"], "label": r.get("name", r["id"])} for r in regions]

        return self.async_show_form(
            step_id="select_region",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_SPECIES_REGION): selector.SelectSelector(
                        selector.SelectSelectorConfig(options=region_options, mode="dropdown")
                    )
                }
            ),
            description_placeholders={"info": "Choose the region you will fish in — species list will be filtered by this region."},
        )

    async def async_step_select_species_for_region(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Show species filtered by the chosen region."""
        if self.species_loader is None:
            self.species_loader = SpeciesLoader(self.hass)
            await self.species_loader.async_load_profiles()

        region_id = self.ocean_config.get(CONF_SPECIES_REGION)
        if not region_id:
            _LOGGER.error("Region not selected before species-for-region step")
            return self.async_abort(reason="region_missing")

        if user_input is not None:
            species_id = user_input.get(CONF_SPECIES_ID)
            species_list = self.species_loader.get_species_by_region(region_id)
            if not any(s.get("id") == species_id for s in species_list):
                _LOGGER.error("Species %s not available in region %s", species_id, region_id)
                return self.async_show_form(
                    step_id="select_species_for_region",
                    data_schema=vol.Schema(
                        {
                            vol.Required(CONF_SPECIES_ID): selector.SelectSelector(
                                selector.SelectSelectorConfig(
                                    options=[
                                        {"value": s["id"], "label": f'{s.get("emoji","🐟")} {s.get("common_name", s["id"])}'}
                                        for s in species_list
                                    ],
                                    mode="dropdown",
                                )
                            )
                        }
                    ),
                    errors={"base": "invalid_species_for_region"},
                )

            self.ocean_config[CONF_SPECIES_ID] = species_id
            return await self.async_step_ocean_habitat()

        species_list = self.species_loader.get_species_by_region(region_id)
        if not species_list:
            _LOGGER.error("No species defined for region %s", region_id)
            return self.async_abort(reason="no_species_for_region")

        species_list.sort(key=lambda s: s.get("common_name", s.get("id")))
        options = []
        for s in species_list:
            options.append({"value": s["id"], "label": f'{s.get("emoji","🐟")} {s.get("common_name", s["id"])}'})

        return self.async_show_form(
            step_id="select_species_for_region",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_SPECIES_ID): selector.SelectSelector(
                        selector.SelectSelectorConfig(options=options, mode="dropdown")
                    )
                }
            ),
            description_placeholders={"info": "Choose the specific species you want to target in this region."},
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
                                        {"value": "open_beach", "label": "Open Sandy Beach"},
                                        {"value": "rocky_point", "label": "Rocky Point / Jetty"},
                                        {"value": "harbour", "label": "Harbour / Pier"},
                                        {"value": "reef", "label": "Offshore Reef"},
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
                                {"value": "open_beach", "label": "Open Sandy Beach"},
                                {"value": "rocky_point", "label": "Rocky Point / Jetty"},
                                {"value": "harbour", "label": "Harbour / Pier"},
                                {"value": "reef", "label": "Offshore Reef"},
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
            if errors:
                return self.async_show_form(
                    step_id="ocean_time_periods",
                    data_schema=vol.Schema(
                        {
                            vol.Required(
                                CONF_TIME_PERIODS,
                                default=self.ocean_config.get(CONF_TIME_PERIODS, TIME_PERIODS_FULL_DAY),
                            ): selector.SelectSelector(
                                selector.SelectSelectorConfig(
                                    options=[
                                        {"value": TIME_PERIODS_FULL_DAY, "label": "Full Day (4 periods)"},
                                        {"value": TIME_PERIODS_DAWN_DUSK, "label": "Dawn & Dusk only"},
                                    ],
                                    mode="list",
                                )
                            )
                        }
                    ),
                    errors=errors,
                    description_placeholders={
                        "info": "Choose which time periods to monitor. Dawn & dusk focuses on the most productive fishing times."
                    },
                )

            self.ocean_config.update(user_input)
            return await self.async_step_wind_direction_config()

        return self.async_show_form(
            step_id="ocean_time_periods",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_TIME_PERIODS, default=TIME_PERIODS_FULL_DAY): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                {"value": TIME_PERIODS_FULL_DAY, "label": "Full Day (4 periods)"},
                                {"value": TIME_PERIODS_DAWN_DUSK, "label": "Dawn & Dusk only"},
                            ],
                            mode="list",
                        )
                    )
                }
            ),
            description_placeholders={"info": "Choose which time periods to monitor. Dawn & dusk focuses on the most productive fishing times."},
        )

    async def async_step_wind_direction_config(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure preferred wind directions."""
        if user_input is not None:
            is_important = user_input.get("wind_direction_important", False)
            if is_important:
                return await self.async_step_wind_direction_select()
            else:
                # Not important - save empty and continue to units
                self.ocean_config[CONF_PREFERRED_WIND_DIRECTIONS] = ""
                return await self.async_step_ocean_units()

        return self.async_show_form(
            step_id="wind_direction_config",
            data_schema=vol.Schema(
                {
                    vol.Required("wind_direction_important", default=False): selector.BooleanSelector(),
                }
            ),
            description_placeholders={
                "info": "Would you like wind direction to be a factor in the fishing score? If enabled, you can select preferred wind directions (e.g., offshore winds)."
            },
        )

    async def async_step_wind_direction_select(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Select preferred wind directions (multi-select)."""
        if user_input is not None:
            directions = user_input.get("preferred_wind_directions", [])
            if directions:
                self.ocean_config[CONF_PREFERRED_WIND_DIRECTIONS] = ",".join(directions)
            else:
                self.ocean_config[CONF_PREFERRED_WIND_DIRECTIONS] = ""
            return await self.async_step_ocean_units()

        direction_options = [
            {"value": "N", "label": "North"},
            {"value": "NNE", "label": "North-Northeast"},
            {"value": "NE", "label": "Northeast"},
            {"value": "ENE", "label": "East-Northeast"},
            {"value": "E", "label": "East"},
            {"value": "ESE", "label": "East-Southeast"},
            {"value": "SE", "label": "Southeast"},
            {"value": "SSE", "label": "South-Southeast"},
            {"value": "S", "label": "South"},
            {"value": "SSW", "label": "South-Southwest"},
            {"value": "SW", "label": "Southwest"},
            {"value": "WSW", "label": "West-Southwest"},
            {"value": "W", "label": "West"},
            {"value": "WNW", "label": "West-Northwest"},
            {"value": "NW", "label": "Northwest"},
            {"value": "NNW", "label": "North-Northwest"},
        ]

        return self.async_show_form(
            step_id="wind_direction_select",
            data_schema=vol.Schema(
                {
                    vol.Required("preferred_wind_directions", default=[]): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=direction_options,
                            mode="dropdown",
                            multiple=True,
                        )
                    ),
                }
            ),
            description_placeholders={
                "info": "Select the wind directions that are most favorable for fishing at your location. You can select multiple directions."
            },
        )

    async def async_step_ocean_units(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Ask user which display units they want (metric/imperial)."""
        if user_input is not None:
            units = user_input.get("units")
            if units not in ("metric", "imperial"):
                return self.async_show_form(
                    step_id="ocean_units",
                    data_schema=vol.Schema(
                        {
                            vol.Required("units", default="metric"): selector.SelectSelector(
                                selector.SelectSelectorConfig(
                                    options=[
                                        {"value": "metric", "label": "Metric (km/h, m, °C)"},
                                        {"value": "imperial", "label": "Imperial (mph, ft, °F)"},
                                    ],
                                    mode="list",
                                )
                            )
                        }
                    ),
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
                                {"value": "metric", "label": "Metric (km/h, m, °C)"},
                                {"value": "imperial", "label": "Imperial (mph, ft, °F)"},
                            ],
                            mode="list",
                        )
                    )
                }
            ),
        )

    async def async_step_ocean_thresholds(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Configure thresholds and finish ocean config (strict)."""
        if user_input is not None:
            try:
                if CONF_LATITUDE not in self.ocean_config or CONF_LONGITUDE not in self.ocean_config:
                    _LOGGER.error("Latitude/Longitude missing from ocean_config; aborting.")
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

                # Build display->canonical safety dict then convert -> validate/normalize
                safety_display = {
                    "safety_max_wind": user_input["max_wind_speed"],
                    "safety_max_gust": user_input.get("max_gust_speed"),
                    "safety_max_wave_height": user_input["max_wave_height"],
                    "safety_min_visibility": user_input.get("min_visibility"),
                    "safety_min_swell_period": user_input.get("min_swell_period"),
                    "safety_max_precip_chance": user_input.get("max_precip_chance"),
                }
                canonical = convert_safety_display_to_metric(safety_display, entry_units=units)
                normalized_limits, warnings = validate_and_normalize_safety_limits(canonical, strict=True)
                safety_limits = normalized_limits

                # Build final_config (data). Keep canonical safety_limits here and
                # keep other core static fields required at setup.
                final_config = {
                    CONF_NAME: self.ocean_config.get(CONF_NAME, DEFAULT_NAME),
                    CONF_LATITUDE: latitude,
                    CONF_LONGITUDE: longitude,
                    CONF_SPECIES_ID: self.ocean_config.get(CONF_SPECIES_ID, "general_mixed_global"),
                    CONF_SPECIES_REGION: self.ocean_config.get(CONF_SPECIES_REGION, "global"),
                    CONF_HABITAT_PRESET: habitat_preset,
                    # store canonical safety limits in data (used by coordinator at setup)
                    "safety_limits": safety_limits,
                    # keep units (coordinator expects this in data)
                    "units": units,
                    # timezone and elevation
                    CONF_TIMEZONE: str(self.hass.config.time_zone),
                    CONF_ELEVATION: self.hass.config.elevation,
                }

                # Add unit strings expected downstream (wind_unit is required by coordinator)
                final_config["wind_unit"] = wind_unit
                final_config["wave_unit"] = "m" if units == "metric" else "ft"
                final_config["temp_unit"] = "°C" if units == "metric" else "°F"

                # Factor weights canonical stored in both data (for initial setup) and options
                final_config[CONF_FACTOR_WEIGHTS] = self.ocean_config.get(
                    CONF_FACTOR_WEIGHTS, _validate_and_normalize_factor_weights(None)
                )

                # Keep update interval & TTLs in data (these are used at coordinator creation)
                if "update_interval" in self.ocean_config:
                    final_config["update_interval"] = int(self.ocean_config.get("update_interval"))

                final_config[CONF_FETCH_CACHE_TTL] = int(self.ocean_config.get(CONF_FETCH_CACHE_TTL, DEFAULT_UPDATE_INTERVAL))
                final_config[CONF_TIDE_TTL] = int(self.ocean_config.get(CONF_TIDE_TTL, TIDE_PROXY_TTL_DEFAULT))
                final_config[CONF_WEATHER_CACHE_TTL] = int(self.ocean_config.get(CONF_WEATHER_CACHE_TTL, WEATHER_FETCHER_CACHE_TTL_DEFAULT))

                # Persist the World Tides API key in data (sensitive; stored in entry.data)
                if CONF_WORLD_TIDES_API_KEY in self.ocean_config:
                    final_config[CONF_WORLD_TIDES_API_KEY] = self.ocean_config.get(CONF_WORLD_TIDES_API_KEY)

                # Persist the chosen tide phase offset minutes in options (not data)
                # Build an options dict so that entry.options will contain the UI-editable settings
                options: dict[str, Any] = {}

                # persist the chosen units so the Options UI shows the correct unit labels
                options["units"] = units

                # time periods should be editable in options UI — write to options too
                options[CONF_TIME_PERIODS] = self.ocean_config.get(CONF_TIME_PERIODS, TIME_PERIODS_FULL_DAY)

                # copy the threshold-display values into options (UI shows/edits these)
                options["max_wind_speed"] = user_input["max_wind_speed"]
                options["max_gust_speed"] = user_input.get("max_gust_speed")
                options["max_wave_height"] = user_input["max_wave_height"]
                options["min_visibility"] = user_input.get("min_visibility")
                options["min_temperature"] = user_input.get("min_temperature")
                options["max_temperature"] = user_input.get("max_temperature")
                options["min_swell_period"] = user_input.get("min_swell_period")
                options["max_precip_chance"] = user_input.get("max_precip_chance")

                # expose_raw is configured earlier (advanced) — include in options
                options["expose_raw"] = bool(self.ocean_config.get("expose_raw", False))

                # add tide phase offset into options (so options UI shows it)
                options[CONF_TIDE_PHASE_OFFSET_MINUTES] = int(
                    self.ocean_config.get(CONF_TIDE_PHASE_OFFSET_MINUTES, TIDE_PHASE_OFFSET_MINUTES_DEFAULT)
                )

                # store normalized factor weights in options as well (UI uses factor_* sliders)
                options[CONF_FACTOR_WEIGHTS] = final_config[CONF_FACTOR_WEIGHTS]

                # Store wind direction preference in options
                options[CONF_PREFERRED_WIND_DIRECTIONS] = self.ocean_config.get(CONF_PREFERRED_WIND_DIRECTIONS, "")

                # Duplicate title check
                existing_entries = self.hass.config_entries.async_entries(DOMAIN)
                title = final_config[CONF_NAME]
                for e in existing_entries:
                    if e.title == title:
                        _LOGGER.debug("User attempted to create entry with duplicate title '%s'", title)
                        return self._show_ocean_thresholds_form(errors={"base": "title_exists"})

                # Use title as unique_id (legacy/simple approach)
                unique_id = title.strip() if isinstance(title, str) else str(title)
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()

                _LOGGER.debug(
                    "Creating ocean config entry with data keys: %s (unique_id=%s) and options keys: %s",
                    list(final_config.keys()),
                    unique_id,
                    list(options.keys()),
                )

                # Create the config entry with both data and options so Options UI immediately reflects setup values.
                return self.async_create_entry(title=final_config[CONF_NAME], data=final_config, options=options)
            except KeyError as ke:
                _LOGGER.exception("Missing expected key when building final ocean config: %s", ke)
                return self._show_ocean_thresholds_form(errors={"base": "unknown"})
            except Exception as exc:
                _LOGGER.exception("Unhandled exception in async_step_ocean_thresholds: %s", exc)
                return self._show_ocean_thresholds_form(errors={"base": "unknown"})

        return self._show_ocean_thresholds_form()

    def _show_ocean_thresholds_form(self, errors: dict[str, str] | None = None) -> FlowResult:
        habitat = HABITAT_PRESETS.get(
            self.ocean_config.get(CONF_HABITAT_PRESET, HABITAT_ROCKY_POINT),
            HABITAT_PRESETS.get(HABITAT_ROCKY_POINT, {}),
        )
        units = self.ocean_config.get("units", "metric")
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
        self._config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        return await self.async_step_ocean_options()

    async def async_step_wind_direction(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Step 1: Ask if wind direction is important."""
        if user_input is not None:
            is_important = user_input.get("wind_direction_important", False)
            if is_important:
                return await self.async_step_wind_direction_select()
            else:
                # Not important - save empty and continue to thresholds
                self._wind_dirs = ""
                return await self.async_step_ocean_thresholds()

        return self.async_show_form(
            step_id="wind_direction",
            data_schema=vol.Schema(
                {
                    vol.Required("wind_direction_important", default=False): selector.BooleanSelector(),
                }
            ),
            description_placeholders={
                "info": "Would you like wind direction to be a factor in the fishing score? If enabled, you can select preferred wind directions (e.g., offshore winds)."
            },
        )

    async def async_step_wind_direction_select(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Step 2: Select preferred wind directions (multi-select)."""
        if user_input is not None:
            directions = user_input.get("preferred_wind_directions", [])
            if directions:
                self._wind_dirs = ",".join(directions)
            else:
                self._wind_dirs = ""
            return await self.async_step_ocean_thresholds()

        direction_options = [
            {"value": "N", "label": "North"},
            {"value": "NNE", "label": "North-Northeast"},
            {"value": "NE", "label": "Northeast"},
            {"value": "ENE", "label": "East-Northeast"},
            {"value": "E", "label": "East"},
            {"value": "ESE", "label": "East-Southeast"},
            {"value": "SE", "label": "Southeast"},
            {"value": "SSE", "label": "South-Southeast"},
            {"value": "S", "label": "South"},
            {"value": "SSW", "label": "South-Southwest"},
            {"value": "SW", "label": "Southwest"},
            {"value": "WSW", "label": "West-Southwest"},
            {"value": "W", "label": "West"},
            {"value": "WNW", "label": "West-Northwest"},
            {"value": "NW", "label": "Northwest"},
            {"value": "NNW", "label": "North-Northwest"},
        ]

        # Get existing selection for default
        existing_dirs = self._wind_dirs if hasattr(self, "_wind_dirs") else ""
        existing_list = [d.strip() for d in existing_dirs.split(",") if d.strip()] if existing_dirs else []

        return self.async_show_form(
            step_id="wind_direction_select",
            data_schema=vol.Schema(
                {
                    vol.Required("preferred_wind_directions", default=existing_list): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=direction_options,
                            mode="dropdown",
                            multiple=True,
                        )
                    ),
                }
            ),
            description_placeholders={
                "info": "Select the wind directions that are most favorable for fishing at your location. You can select multiple directions."
            },
        )

    async def async_step_ocean_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:

        # Options flow reads/writes only from entry.options (no fallbacks/migrations here).
        current_opts = dict(self._config_entry.options or {})

        if user_input is not None:
            try:
                # Collect raw factor_* values (percent 0-100)
                weights_raw = {}
                for k in FACTOR_WEIGHTS.keys():
                    key_name = f"factor_{k}"
                    if key_name in user_input:
                        try:
                            weights_raw[k] = float(user_input.get(key_name, 0.0))
                        except Exception:
                            weights_raw[k] = 0.0

                if weights_raw:
                    total = float(sum(weights_raw.values()))
                    if abs(total - 100.0) > 0.5:
                        # Build schema to re-present the form with validation error,
                        # using values from current_opts or hardcoded defaults.
                        stored_fw = current_opts.get(CONF_FACTOR_WEIGHTS)
                        try:
                            norm_defaults = stored_fw if isinstance(stored_fw, dict) else _validate_and_normalize_factor_weights(None)
                        except Exception:
                            norm_defaults = _validate_and_normalize_factor_weights(None)
                        stored_defaults = {k: int(round(norm_defaults.get(k, 0.0) * 100)) for k in FACTOR_WEIGHTS.keys()}

                        # compute unit labels from current options so re-displayed form shows correct labels
                        units_local = current_opts.get("units", "metric")
                        wind_unit_label_local = "km/h" if units_local == "metric" else "mph"
                        wave_unit_label_local = "m" if units_local == "metric" else "ft"
                        vis_unit_label_local = "km" if units_local == "metric" else "miles"

                        schema_fields: dict = {}
                        schema_fields[vol.Required(CONF_TIME_PERIODS, default=current_opts.get(CONF_TIME_PERIODS, TIME_PERIODS_FULL_DAY))] = selector.SelectSelector(
                            selector.SelectSelectorConfig(
                                options=[
                                    {"value": TIME_PERIODS_FULL_DAY, "label": "Full Day (4 periods)"},
                                    {"value": TIME_PERIODS_DAWN_DUSK, "label": "Dawn & Dusk only"},
                                ],
                                mode="dropdown",
                            )
                        )
                        schema_fields[vol.Required("max_wind_speed", default=current_opts.get("max_wind_speed", 25))] = selector.NumberSelector(
                            selector.NumberSelectorConfig(min=10, max=50, step=5, unit_of_measurement=wind_unit_label_local, mode="slider")
                        )
                        schema_fields[vol.Required("max_gust_speed", default=current_opts.get("max_gust_speed", 40))] = selector.NumberSelector(
                            selector.NumberSelectorConfig(min=15, max=80, step=5, unit_of_measurement=wind_unit_label_local, mode="slider")
                        )
                        schema_fields[vol.Required("max_wave_height", default=current_opts.get("max_wave_height", 2.0))] = selector.NumberSelector(
                            selector.NumberSelectorConfig(min=0.5, max=10.0, step=0.5, unit_of_measurement=wave_unit_label_local, mode="slider")
                        )
                        schema_fields[vol.Required("max_precip_chance", default=current_opts.get("max_precip_chance", 80))] = selector.NumberSelector(
                            selector.NumberSelectorConfig(min=0, max=100, step=5, unit_of_measurement="%", mode="slider")
                        )
                        schema_fields[vol.Required("min_swell_period", default=current_opts.get("min_swell_period", 3))] = selector.NumberSelector(
                            selector.NumberSelectorConfig(min=0, max=30, step=1, unit_of_measurement="s")
                        )
                        schema_fields[vol.Required("min_visibility", default=current_opts.get("min_visibility", 1))] = selector.NumberSelector(
                            selector.NumberSelectorConfig(min=0, max=50, step=1, unit_of_measurement=vis_unit_label_local, mode="slider")
                        )
                        schema_fields[vol.Required("expose_raw", default=current_opts.get("expose_raw", False))] = selector.BooleanSelector()

                        # Include tide phase offset in options form (advanced tuning)
                        schema_fields[vol.Required(CONF_TIDE_PHASE_OFFSET_MINUTES, default=current_opts.get(CONF_TIDE_PHASE_OFFSET_MINUTES, TIDE_PHASE_OFFSET_MINUTES_DEFAULT))] = selector.NumberSelector(
                            selector.NumberSelectorConfig(min=-180, max=180, step=1, unit_of_measurement="min", mode="slider")
                        )

                        for k in FACTOR_WEIGHTS.keys():
                            schema_fields[vol.Required(f"factor_{k}", default=stored_defaults.get(k, 0))] = selector.NumberSelector(
                                selector.NumberSelectorConfig(min=0, max=100, step=1, unit_of_measurement="%", mode="slider")
                            )

                        return self.async_show_form(
                            step_id="ocean_options",
                            data_schema=vol.Schema(schema_fields),
                            errors={"base": "sum_not_100"},
                            description_placeholders={
                                "info": f"Adjust factor weights. Current total is {total:.1f}%. Values must add to 100%."
                            },
                        )

                    # Normalize factor weights into canonical normalized dict
                    normalized = {k: weights_raw.get(k, 0.0) / 100.0 for k in FACTOR_WEIGHTS.keys()}
                    normalized = _validate_and_normalize_factor_weights(normalized)
                    user_input[CONF_FACTOR_WEIGHTS] = normalized
                    for k in list(FACTOR_WEIGHTS.keys()):
                        user_input.pop(f"factor_{k}", None)

                # Handle wind direction preference
                wind_enabled = user_input.get("wind_direction_enabled", False)
                if wind_enabled:
                    # If enabled but no directions selected, keep existing or empty
                    existing_dirs = current_opts.get(CONF_PREFERRED_WIND_DIRECTIONS, "")
                    user_input[CONF_PREFERRED_WIND_DIRECTIONS] = existing_dirs
                else:
                    user_input[CONF_PREFERRED_WIND_DIRECTIONS] = ""
                user_input.pop("wind_direction_enabled", None)

                # Merge with existing options to avoid clobbering unrelated keys
                new_options = dict(current_opts)
                new_options.update(user_input)
                return self.async_create_entry(title="", data=new_options)
            except Exception as exc:
                _LOGGER.debug("Options flow factor weights normalization failed: %s", exc)

        # Build form defaults: prefer options only, then hardcoded defaults (no data fallback)
        units = current_opts.get("units", "metric")
        wind_unit_label = "km/h" if units == "metric" else "mph"
        wave_unit_label = "m" if units == "metric" else "ft"
        vis_unit_label = "km" if units == "metric" else "miles"

        stored_weights = current_opts.get(CONF_FACTOR_WEIGHTS)
        try:
            normalized_defaults = stored_weights if isinstance(stored_weights, dict) else _validate_and_normalize_factor_weights(None)
        except Exception:
            normalized_defaults = _validate_and_normalize_factor_weights(None)
        factor_defaults_percent: dict[str, int] = {
            k: int(round((normalized_defaults.get(k, 0.0) * 100))) for k in FACTOR_WEIGHTS.keys()
        }
        total_default = sum(factor_defaults_percent.values())

        schema_fields: dict = {}
        schema_fields[vol.Required(CONF_TIME_PERIODS, default=current_opts.get(CONF_TIME_PERIODS, TIME_PERIODS_FULL_DAY))] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    {"value": TIME_PERIODS_FULL_DAY, "label": "Full Day (4 periods)"},
                    {"value": TIME_PERIODS_DAWN_DUSK, "label": "Dawn & Dusk only"},
                ],
                mode="dropdown",
            )
        )
        schema_fields[vol.Required("max_wind_speed", default=current_opts.get("max_wind_speed", 25))] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=10, max=50, step=5, unit_of_measurement=wind_unit_label, mode="slider")
        )
        schema_fields[vol.Required("max_gust_speed", default=current_opts.get("max_gust_speed", 40))] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=15, max=80, step=5, unit_of_measurement=wind_unit_label, mode="slider")
        )
        schema_fields[vol.Required("max_wave_height", default=current_opts.get("max_wave_height", 2.0))] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=0.5, max=10.0, step=0.5, unit_of_measurement=wave_unit_label, mode="slider")
        )
        schema_fields[vol.Required("max_precip_chance", default=current_opts.get("max_precip_chance", 80))] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, max=100, step=5, unit_of_measurement="%", mode="slider")
        )
        schema_fields[vol.Required("min_swell_period", default=current_opts.get("min_swell_period", 3))] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, max=30, step=1, unit_of_measurement="s")
        )
        schema_fields[vol.Required("min_visibility", default=current_opts.get("min_visibility", 1))] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, max=50, step=1, unit_of_measurement=vis_unit_label, mode="slider")
        )
        schema_fields[vol.Required("expose_raw", default=current_opts.get("expose_raw", False))] = selector.BooleanSelector()

        # Include tide phase offset in options UI
        schema_fields[vol.Required(CONF_TIDE_PHASE_OFFSET_MINUTES, default=current_opts.get(CONF_TIDE_PHASE_OFFSET_MINUTES, TIDE_PHASE_OFFSET_MINUTES_DEFAULT))] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=-180, max=180, step=1, unit_of_measurement="min", mode="slider")
        )

        # Wind direction preference
        existing_wind_dirs = current_opts.get(CONF_PREFERRED_WIND_DIRECTIONS, "")
        wind_dir_enabled = bool(existing_wind_dirs)
        schema_fields[vol.Required("wind_direction_enabled", default=wind_dir_enabled)] = selector.BooleanSelector()

        for k in FACTOR_WEIGHTS.keys():
            schema_fields[vol.Required(f"factor_{k}", default=factor_defaults_percent.get(k, 0))] = selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=100, step=1, unit_of_measurement="%", mode="slider")
            )

        return self.async_show_form(
            step_id="ocean_options",
            data_schema=vol.Schema(schema_fields),
            description_placeholders={
                "info": f"Adjust factor weights. Current total is {total_default}%. Values must add to 100%."
            },
        )