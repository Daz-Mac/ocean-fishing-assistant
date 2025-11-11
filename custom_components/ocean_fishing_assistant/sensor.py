# Strict sensor entity - surfaces errors loudly so callers see misconfiguration
from typing import Optional, Dict, Any

from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.const import ATTR_ATTRIBUTION
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN

ATTRIBUTION = "Data provided by Open-Meteo"


class OFASensor(CoordinatorEntity):
    def __init__(self, coordinator, name: str):
        """
        Strict CoordinatorEntity wrapper.

        - Fails loudly if coordinator lacks expected data.
        - Requires explicit `name` (no defaults or fallbacks).
        - The integration expects the configured name to be provided via
          config entry options under the key "sensor_name". The runtime
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

        # Attribution to appear both as explicit key and HA-standard ATTR_ATTRIBUTION for older integrations
        attrs["attribution"] = ATTRIBUTION
        attrs[ATTR_ATTRIBUTION] = ATTR_ATTRIBUTION
        return attrs


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities):
    """
    Entry setup for the single OFA sensor.

    Expects coordinator instance to be stored in hass.data[DOMAIN][entry.entry_id].

    Strict behavior:
      - Requires entry.options["sensor_name"] to be set by the user during config/options.
      - No default or fallback names are used.
      - The configured name will be deterministically prefixed with "ocean_fishing_assistant_" if not already.
      - If the required option is missing, setup fails loudly by raising RuntimeError.
    """
    coordinator = hass.data[DOMAIN].get(entry.entry_id) if hass.data.get(DOMAIN) else None
    if coordinator is None:
        raise RuntimeError("Coordinator not found in hass.data for this config entry (strict)")

    # Enforce presence of explicit configured sensor name (no fallbacks).
    try:
        opts = entry.options or {}
        sensor_name = opts.get("sensor_name")
    except Exception:
        raise RuntimeError("Failed to read config entry options; 'sensor_name' is required (strict)")

    if not sensor_name:
        raise RuntimeError(
            "Missing required option 'sensor_name' in config entry options; "
            "the Ocean Fishing Assistant integration requires the user to set a sensor name during configuration (strict)."
        )

    # Ensure prefixing per policy (deterministic transformation).
    prefix = "ocean_fishing_assistant"
    final_name = sensor_name if sensor_name.startswith(prefix) else f"{prefix}_{sensor_name}"

    async_add_entities([OFASensor(coordinator, name=final_name)], update_before_add=True)