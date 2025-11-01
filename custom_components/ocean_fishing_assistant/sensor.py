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
    def available(self):
        return self.coordinator.last_update_success and self.coordinator.data is not None

    @property
    def state(self):
        # Return "score" as integer (0..100) or unknown (None) if missing
        if not self.coordinator.data:
            return None
        try:
            # compute current (index 0)
            result = compute_score(self.coordinator.data, use_index=0)
            return int(result["score_100"])
        except MissingDataError:
            # Fail loudly: unknown state when required inputs missing
            return None
        except Exception:
            return None

    @property
    def extra_state_attributes(self):
        if not self.coordinator.data:
            return {}
        try:
            result = compute_score(self.coordinator.data, use_index=0)
            attrs = {
                "score_10": result.get("score_10"),
                "score_100": result.get("score_100"),
                "components": result.get("components", {}),
                "profile_used": result.get("profile_used"),
                "raw": result.get("raw"),
                ATTR_ATTRIBUTION: ATTRIBUTION,
            }
            return attrs
        except Exception:
            return {}

async def async_setup_entry(hass, entry, async_add_entities):
    """Set up sensor platform from a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([OFASensor(coordinator)], update_before_add=True)