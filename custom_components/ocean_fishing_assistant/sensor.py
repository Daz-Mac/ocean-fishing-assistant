# Strict sensor entity - surfaces errors loudly so callers see misconfiguration
from typing import Optional, Dict, Any

from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.const import ATTR_ATTRIBUTION
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN, CONF_NAME

ATTRIBUTION = "Data provided by Open-Meteo"


class OFASensor(CoordinatorEntity):
    def __init__(self, coordinator, name: str):
        """
        Strict CoordinatorEntity wrapper.

        - Fails loudly if coordinator lacks expected data.
        - Requires explicit `name` (no defaults or fallbacks).
        - The integration expects the configured name to be provided via
          the config entry data under the key CONF_NAME. The runtime
          name used for the entity will be prefixed with
          "ocean_fishing_assistant_" (unless the provided name already
          starts with that prefix).
        """
        if not name:
            raise RuntimeError("Sensor name must be provided (strict)")

        super().__init__(coordinator)

        # Enforce prefix deterministically; do not silently choose a name if missing.
        prefix = "ocean_fishing_assistant"
        if not name.startswith(prefix):
            name = f"{prefix}_{name}"

        self._attr_name = name

        # Provide a deterministic unique id so HA entities remain stable across restarts.
        # Include the entry/coordinator id and the configured name to avoid collisions
        # if multiple entries are created.
        try:
            safe_name = name.replace(" ", "_")
            self._attr_unique_id = f"{prefix}_{getattr(coordinator, 'entry_id', 'noentry')}_{safe_name}"
        except Exception:
            # If coordinator doesn't provide entry_id, still permit creation but flag absence via unique id being None.
            self._attr_unique_id = None

    @property
    def available(self) -> bool:
        return bool(self.coordinator.last_update_success and self.coordinator.data is not None)

    def _get_current_forecast(self) -> Optional[Dict[str, Any]]:
        data = self.coordinator.data or {}
        forecasts = data.get("per_timestamp_forecasts") or []
        if isinstance(forecasts, (list, tuple)) and len(forecasts) > 0:
            return forecasts[0]
        return None

    @property
    def state(self) -> Optional[int]:
        """Return state as integer 0..100 or raise on strict failures."""
        if not self.coordinator.data:
            raise RuntimeError("Coordinator data missing for OFA sensor (strict)")

        forecast = self._get_current_forecast()
        if forecast:
            sc = forecast.get("score_100")
            if sc is None:
                raise RuntimeError("Precomputed forecast present but score_100 is missing (strict)")
            return int(sc)

        raise RuntimeError("No precomputed per_timestamp_forecasts found in coordinator.data (strict)")

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """
        Strict attributes:

        - Fail loudly if coordinator.data is absent (consistent with strict policy).
        - Expose canonical fields required for debugging and integrations.
        - Add several current_* top-level convenience fields drawn from either
          forecast_raw.formatted_weather or the raw current snapshot.
        """
        if not self.coordinator.data:
            raise RuntimeError("Coordinator data missing when building attributes (strict)")

        attrs: Dict[str, Any] = {}
        forecasts = self.coordinator.data.get("per_timestamp_forecasts")
        if forecasts is not None:
            attrs["per_timestamp_forecasts"] = forecasts

        current = self._get_current_forecast()
        if current is not None:
            attrs["current_forecast"] = current

        raw = self.coordinator.data.get("raw_payload") or self.coordinator.data
        attrs["raw_payload"] = raw

        # Add profile and safety details coming from per-timestamp/current forecast where present
        attrs["profile_used"] = (current.get("profile_used") if current else None)
        # Coordinator holds chosen units and safety limits in strict coordinator implementation
        attrs["units"] = getattr(self.coordinator, "units", None)
        attrs["safety_limits"] = getattr(self.coordinator, "safety_limits", None)
        attrs["safety"] = (current.get("safety") if current else None)

        # Expose both reason codes and human-readable strings for safety reasons (if present)
        if current and isinstance(current.get("safety"), dict):
            safety_obj = current.get("safety") or {}
            # codes
            attrs["safety_reason_codes"] = list(safety_obj.get("reasons", []) or [])
            # human-friendly strings (compute_forecast supplies 'reason_strings')
            attrs["safety_reason_strings"] = list(safety_obj.get("reason_strings", []) or [])

        # Extract formatted-weather (preferred) and raw current snapshot (fallback)
        formatted = None
        try:
            if current and isinstance(current.get("forecast_raw"), dict):
                formatted = current.get("forecast_raw", {}).get("formatted_weather") or {}
        except Exception:
            formatted = {}

        raw_current = None
        try:
            if isinstance(raw, dict):
                raw_current = raw.get("current")
        except Exception:
            raw_current = None

        # Helper to read a key from formatted then raw_current (return None if missing)
        def _pick(*keys):
            # keys are names to look for in formatted first, then in raw_current
            for k in keys:
                try:
                    if formatted and k in formatted:
                        return formatted.get(k)
                except Exception:
                    pass
                try:
                    if raw_current and k in raw_current:
                        return raw_current.get(k)
                except Exception:
                    pass
            return None

        # Provide convenient top-level current_* attributes:
        attrs["current_temperature"] = _pick("temperature", "temp", "temperature_c")
        # wind speed: formatted uses "wind" (m/s canonical) while raw_current likely uses "wind_speed"
        attrs["current_wind_speed"] = _pick("wind", "wind_speed", "wind_m_s")
        # wind gust (duplicate of earlier current_wind_gust but included under canonical name)
        attrs["current_wind_gust"] = _pick("wind_gust", "wind_max_m_s", "windgusts_10m")
        # wind unit (from current snapshot)
        attrs["current_wind_unit"] = _pick("wind_unit", "wind_unit")
        # pressure: formatted uses 'pressure_hpa', raw_current may use 'pressure'
        attrs["current_pressure_hpa"] = _pick("pressure_hpa", "pressure", "pressure_hpa")
        # cloud cover
        attrs["current_cloud_cover"] = _pick("cloud_cover", "cloud", "cloudcover")
        # precipitation probability
        attrs["current_precipitation_probability"] = _pick("precipitation_probability", "precipitation_probability", "pop")
        # visibility (if present)
        attrs["current_visibility_km"] = _pick("visibility", "visibility_km")
        # Wave / swell metrics
        attrs["current_wave_height_m"] = _pick("wave_height_m", "wave_height", "wave_height")
        attrs["current_wave_period_s"] = _pick("wave_period_s", "wave_period", "wave_period_s")
        attrs["current_swell_height_m"] = _pick("swell_height_m", "swell_wave_height", "swell_height")
        attrs["current_swell_period_s"] = _pick("swell_period_s", "swell_period_s")

        # Keep legacy-named compact fields where applicable for backwards compat
        # (some consumers may still look for these)
        # current_wind_gust is already set above; set short-named variants too
        if attrs.get("current_wind_gust") is None:
            # try explicit fallback that some fetchers use
            attrs["current_wind_gust"] = _pick("wind_max_m_s", "wind_max_m_s")

        # Attribution to appear both as explicit key and HA-standard ATTR_ATTRIBUTION for older integrations
        attrs["attribution"] = ATTRIBUTION
        attrs[ATTR_ATTRIBUTION] = ATTRIBUTION
        return attrs


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities):
    """
    Entry setup for the single OFA sensor.

    Expects coordinator instance to be stored in hass.data[DOMAIN][entry.entry_id].

    Strict behavior:
      - Requires entry.data[CONF_NAME] to be set by the user during config.
      - No default or fallback names are used.
      - The configured name will be deterministically prefixed with "ocean_fishing_assistant_" if not already.
      - If the required key is missing, setup fails loudly by raising RuntimeError.
    """
    coordinator = hass.data[DOMAIN].get(entry.entry_id) if hass.data.get(DOMAIN) else None
    if coordinator is None:
        raise RuntimeError("Coordinator not found in hass.data for this config entry (strict)")

    try:
        # Read name from entry.data (strict — the config flow must have stored this)
        sensor_name = entry.data.get(CONF_NAME)
    except Exception:
        raise RuntimeError("Failed to read config entry data; 'name' (CONF_NAME) is required (strict)")

    if not sensor_name:
        raise RuntimeError(
            "Missing required name in config entry data; the Ocean Fishing Assistant integration requires the user to set a name during configuration (strict)."
        )

    # Ensure prefixing per policy (deterministic transformation).
    prefix = "ocean_fishing_assistant"
    final_name = sensor_name if sensor_name.startswith(prefix) else f"{prefix}_{sensor_name}"

    async_add_entities([OFASensor(coordinator, name=final_name)], update_before_add=True)