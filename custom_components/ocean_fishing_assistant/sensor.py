# Strict sensor entity - surfaces errors loudly so callers see misconfiguration
from typing import Optional, Dict, Any

from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.const import ATTR_ATTRIBUTION

from .const import DEFAULT_NAME, DOMAIN

ATTRIBUTION = "Data provided by Open-Meteo"


class OFASensor(CoordinatorEntity):
    def __init__(self, coordinator, name: str = DEFAULT_NAME):
        """
        Strict CoordinatorEntity wrapper.

        - Fails loudly if coordinator lacks expected data.
        - Uses coordinator.units / coordinator.safety_limits for attributes.
        """
        super().__init__(coordinator)
        self._attr_name = name
        # Provide a deterministic unique id so HA entities remain stable across restarts
        try:
            self._attr_unique_id = f"ocean_fishing_assistant_{coordinator.entry_id}"
        except Exception:
            # If coordinator doesn't provide entry_id, still permit creation but warn via unique id absence.
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

        # Attribution to appear both as explicit key and HA-standard ATTR_ATTRIBUTION for older integrations
        attrs["attribution"] = ATTRIBUTION
        attrs[ATTR_ATTRIBUTION] = ATTRIBUTION
        return attrs


async def async_setup_entry(hass, entry, async_add_entities):
    """
    Entry setup for the single OFA sensor.

    Expects coordinator instance to be stored in hass.data[DOMAIN][entry.entry_id].
    """
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([OFASensor(coordinator)], update_before_add=True)