from datetime import timedelta

DOMAIN = "ocean_fishing_assistant"
DEFAULT_NAME = "Ocean Fishing Assistant"
DEFAULT_UPDATE_INTERVAL = 15 * 60  # seconds
OM_BASE = "https://api.open-meteo.com/v1/forecast"
STORE_VERSION = 1
STORE_KEY = f"{DOMAIN}_store"
DEFAULT_FORECAST_MODE = "hourly"  # or 'daily'

FETCH_CACHE_TTL = 600  # seconds for shared in-memory Open-Meteo fetch cache