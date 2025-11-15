"""Constants for Ocean Fishing Assistant (ocean-only, strict)."""

from datetime import timedelta

# Integration identity
DOMAIN = "ocean_fishing_assistant"
DEFAULT_NAME = "Ocean Fishing Assistant"

# Update interval (seconds) default used by coordinator
DEFAULT_UPDATE_INTERVAL = 15 * 60  # seconds

# Open-Meteo endpoints
OM_BASE = "https://api.open-meteo.com/v1/forecast"
OM_MARINE_BASE = "https://marine-api.open-meteo.com/v1/marine"

# Storage keys
STORE_VERSION = 1
STORE_KEY = f"{DOMAIN}_store"

# Default forecast mode (used where appropriate)
DEFAULT_FORECAST_MODE = "hourly"

# Default (canonical) safety limits stored per-entry/options.
# The integration requires explicit safety limits at config time; these are sane defaults
# but the config flow / options should ensure these are explicitly set by the user.
DEFAULT_SAFETY_LIMITS = {
    "max_wave_height_m": 2.5,
    "max_wind_m_s": 15.0,
    # other defaults intentionally omitted (they are optional / user-set): gust, visibility, swell, precip
    "max_gust_m_s": None,
    "min_visibility_km": None,
    "min_swell_period_s": None,
    # Precipitation chance default: None (disabled unless user sets)
    "max_precip_chance_pct": None,
}

FETCH_CACHE_TTL = 600  # seconds for shared in-memory Open-Meteo fetch cache

# ----- Config keys used by the flow and entry options -----
CONF_NAME = "name"
CONF_LATITUDE = "latitude"
CONF_LONGITUDE = "longitude"
CONF_SPECIES_ID = "species"  # canonical key used in entry.options for chosen species
CONF_SPECIES_REGION = "species_region"
CONF_HABITAT_PRESET = "habitat_preset"
CONF_TIME_PERIODS = "time_periods"
CONF_THRESHOLDS = "thresholds"
CONF_TIMEZONE = "timezone"
CONF_ELEVATION = "elevation"
CONF_AUTO_APPLY_THRESHOLDS = "auto_apply_thresholds"
CONF_TIDE_MODE = "tide_mode"
CONF_MARINE_ENABLED = "marine_enabled"

# Tide mode constants
TIDE_MODE_PROXY = "proxy"
TIDE_MODE_SENSOR = "sensor"

# Time period options
TIME_PERIODS_FULL_DAY = "full_day"
TIME_PERIODS_DAWN_DUSK = "dawn_dusk"

# Habitat presets and defaults used to seed the thresholds UI
HABITAT_ROCKY_POINT = "rocky_point"
HABITAT_OPEN_BEACH = "open_beach"
HABITAT_HARBOUR = "harbour"
HABITAT_REEF = "reef"

HABITAT_PRESETS = {
    HABITAT_ROCKY_POINT: {
        "name": "Rocky Point / Jetty",
        "max_wind_speed": 25,  # km/h (used in UI sliders)
        "max_gust_speed": 40,  # km/h
        "max_wave_height": 2.0,  # m
        "min_visibility": 5,  # km
        "min_swell_period": 10,  # s
        "max_precip_chance": 80,  # % default used to seed UI slider
    },
    HABITAT_OPEN_BEACH: {
        "name": "Open Sandy Beach",
        "max_wind_speed": 20,
        "max_gust_speed": 35,
        "max_wave_height": 1.5,
        "min_visibility": 6,
        "min_swell_period": 8,
        "max_precip_chance": 80,
    },
    HABITAT_HARBOUR: {
        "name": "Harbour / Pier",
        "max_wind_speed": 30,
        "max_gust_speed": 50,
        "max_wave_height": 0.8,
        "min_visibility": 4,
        "min_swell_period": 6,
        "max_precip_chance": 85,
    },
    HABITAT_REEF: {
        "name": "Offshore Reef",
        "max_wind_speed": 20,
        "max_gust_speed": 40,
        "max_wave_height": 3.0,
        "min_visibility": 8,
        "min_swell_period": 12,
        "max_precip_chance": 70,
    },
}