from datetime import timedelta

DOMAIN = "ocean_fishing_assistant"
DEFAULT_NAME = "Ocean Fishing Assistant"
DEFAULT_UPDATE_INTERVAL = 15 * 60  # seconds
OM_BASE = "https://api.open-meteo.com/v1/forecast"
STORE_VERSION = 1
STORE_KEY = f"{DOMAIN}_store"
DEFAULT_FORECAST_MODE = "hourly"  # or 'daily'


# Marine-specific Open-Meteo endpoint (provides wave / sea-surface variables)
OM_MARINE_BASE = "https://marine-api.open-meteo.com/v1/marine"

# Default safety limits applied per-config-entry as conservative fallbacks.
# Note: safety limits are required at config time and are stored per-entry; these
# defaults should not be relied upon in normal operation.
DEFAULT_SAFETY_LIMITS = {
    "max_wave_height_m": 2.5,
    "max_wind_m_s": 15.0,
}

FETCH_CACHE_TTL = 600  # seconds for shared in-memory Open-Meteo fetch cache