from homeassistant.helpers.entity import CoordinatorEntity
from homeassistant.const import ATTR_ATTRIBUTION

from .const import DEFAULT_NAME, DOMAIN
from .ocean_scoring import compute_score

ATTRIBUTION = "Data provided by Open-Meteo"

class OFASensor(CoordinatorEntity):
    def __init__(self, coordinator, name=DEFAULT_NAME):
        super().__init__(coordinator)
        self._attr_name = name

    @property
    def available(self):
        return self.coordinator.last_update_success and self.coordinator.data is not None

    @property
    def state(self):
        # Return "score" as integer or unknown (None) if missing
        try:
            result = compute_score(self.coordinator.data)
            return int(result["score"])
        except Exception:
            # Fail loudly: unknown state when inputs missing
            return None

    @property
    def extra_state_attributes(self):
        if not self.coordinator.data:
            return {}
        try:
            result = compute_score(self.coordinator.data)
            return {
                "components": result.get("components", {}),
                ATTR_ATTRIBUTION: ATTRIBUTION,
            }
        except Exception:
            return {}

async def async_setup_entry(hass, entry, async_add_entities):
    """Set up sensor platform from a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([OFASensor(coordinator)], update_before_add=True)