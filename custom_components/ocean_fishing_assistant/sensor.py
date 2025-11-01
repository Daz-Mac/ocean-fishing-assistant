from typing import Optional, List, Dict, Any

from homeassistant.helpers.entity import CoordinatorEntity
from homeassistant.const import ATTR_ATTRIBUTION

from .const import DEFAULT_NAME, DOMAIN
from .ocean_scoring import compute_score, MissingDataError

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
        forecasts = data.get("forecasts") or []
        if isinstance(forecasts, (list, tuple)) and len(forecasts) > 0:
            return forecasts[0]
        return None

    @property
    def state(self) -> Optional[int]:
        """Return state as integer 0..100 or None."""
        if not self.coordinator.data:
            return None

        # Prefer precomputed forecast (from DataFormatter)
        forecast = self._get_current_forecast()
        if forecast:
            sc = forecast.get("score_100")
            return int(sc) if sc is not None else None

        # Fallback: compute from raw payload (index 0)
        try:
            result = compute_score(self.coordinator.data, species_profile=getattr(self.coordinator, "species", None), use_index=0)
            return int(result["score_100"])
        except MissingDataError:
            return None
        except Exception:
            return None

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Expose the precomputed forecasts and helpful raw data for debugging/automation."""
        if not self.coordinator.data:
            return {}

        attrs: Dict[str, Any] = {}
        # Expose the full forecasts list (index-aware, 5-day hourly)
        forecasts = self.coordinator.data.get("forecasts")
        if forecasts is not None:
            attrs["forecasts"] = forecasts

        # Also expose the current forecast (duplicate for convenience)
        current = self._get_current_forecast()
        if current is not None:
            attrs["current_forecast"] = current

        # Keep raw values used by scoring (tide, wind, wave, etc.) for troubleshooting
        raw = self.coordinator.data.get("raw_payload") or self.coordinator.data
        attrs["raw_payload"] = raw

        attrs["profile_used"] = current.get("profile_used") if current else None
        attrs["units"] = getattr(self.coordinator, "units", None)
        attrs["attribution"] = ATTRIBUTION
        attrs[ATTR_ATTRIBUTION] = ATTRIBUTION
        return attrs


async def async_setup_entry(hass, entry, async_add_entities):
    """Set up sensor platform from a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([OFASensor(coordinator)], update_before_add=True)