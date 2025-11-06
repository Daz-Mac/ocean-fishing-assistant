from typing import Optional, Dict, Any

from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.const import ATTR_ATTRIBUTION

from .const import DEFAULT_NAME, DOMAIN

ATTRIBUTION = "Data provided by Open-Meteo"


class OFASensor(CoordinatorEntity):
    def __init__(self, coordinator, name=DEFAULT_NAME):
        super().__init__(coordinator)
        self._attr_name = name

    @property
    def available(self) -> bool:
        return self.coordinator.last_update_success and self.coordinator.data is not None

    def _get_current_forecast(self) -> Optional[Dict[str, Any]]:
        """Return the precomputed forecast dict for index 0 if available, otherwise None."""
        data = self.coordinator.data or {}
        forecasts = data.get("per_timestamp_forecasts") or []
        if isinstance(forecasts, (list, tuple)) and len(forecasts) > 0:
            return forecasts[0]
        return None

    @property
    def state(self) -> Optional[int]:
        """Return state as integer 0..100 or raise on strict failures."""
        if not self.coordinator.data:
            # Strict: fail loudly when no coordinator data present
            raise RuntimeError("Coordinator data missing for OFA sensor")

        # Prefer precomputed forecast (from DataFormatter)
        forecast = self._get_current_forecast()
        if forecast:
            sc = forecast.get("score_100")
            if sc is None:
                # Strict: precomputed forecast exists but score missing -> surface error
                raise RuntimeError("Precomputed forecast present but score_100 is missing for index 0")
            return int(sc)

        # Strict: do not attempt silent fallbacks. Require precomputed forecasts.
        raise RuntimeError("No precomputed per_timestamp_forecasts found in coordinator.data")

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Expose the precomputed forecasts and helpful raw data for debugging/automation. Fail loudly if missing expected data."""
        if not self.coordinator.data:
            return {}

        attrs: Dict[str, Any] = {}
        # Expose the full per_timestamp_forecasts list (index-aware, hourly)
        forecasts = self.coordinator.data.get("per_timestamp_forecasts")
        if forecasts is not None:
            attrs["per_timestamp_forecasts"] = forecasts

        # Also expose the current forecast (duplicate for convenience)
        current = self._get_current_forecast()
        if current is not None:
            attrs["current_forecast"] = current

        # Keep raw values used by scoring (tide, wind, wave, etc.) for troubleshooting
        raw = self.coordinator.data.get("raw_payload") or self.coordinator.data
        attrs["raw_payload"] = raw

        attrs["profile_used"] = current.get("profile_used") if current else None
        attrs["units"] = getattr(self.coordinator, "units", None)
        # expose the canonical safety limits used for this config entry
        attrs["safety_limits"] = getattr(self.coordinator, "safety_limits", None)
        # also expose the safety evaluation for current index
        attrs["safety"] = current.get("safety") if current else None

        attrs["attribution"] = ATTRIBUTION
        attrs[ATTR_ATTRIBUTION] = ATTRIBUTION
        return attrs


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up sensor platform from a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([OFASensor(coordinator)], update_before_add=True)