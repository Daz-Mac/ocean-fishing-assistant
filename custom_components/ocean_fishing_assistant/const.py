"""Constants for Ocean Fishing Assistant (ocean-only, strict)."""

# Integration identity
DOMAIN = "ocean_fishing_assistant"
DEFAULT_NAME = "Ocean Fishing Assistant"

# Update interval (seconds) default used by coordinator
DEFAULT_UPDATE_INTERVAL = 30 * 60  # seconds

# Open-Meteo endpoints
OM_BASE = "https://api.open-meteo.com/v1/forecast"
OM_MARINE_BASE = "https://marine-api.open-meteo.com/v1/marine"

# World Tides API base (configurable)
WORLD_TIDES_API_BASE = "https://www.worldtides.info/api/v3"
# Name of config entry field for World Tides API key (store in entry.data)
CONF_WORLD_TIDES_API_KEY = "world_tides_api_key"

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

# Coordinator shared in-memory fetch cache TTL (seconds)
# This is used by OFACoordinator to decide when to reuse a cached Open-Meteo payload keyed per-location.
FETCH_CACHE_TTL = DEFAULT_UPDATE_INTERVAL  # seconds (align with DEFAULT_UPDATE_INTERVAL)

# Separate defaults for the three TTLs (Option B)
TIDE_PROXY_TTL_DEFAULT = 24 * 60 * 60  # seconds (default 1 day)

# Network / World Tides HTTP defaults
WORLD_TIDES_TIMEOUT_SECONDS = 15  # per-request timeout (seconds)
WORLD_TIDES_MAX_RETRIES = 2       # number of retries on 5xx/429/network errors
WORLD_TIDES_RETRY_BACKOFF_SECONDS = 1  # base backoff seconds (exponential backoff)

# Shared cache bucketing and names
TIDE_CACHE_BUCKET_SECONDS = 300   # seconds for time-bucketing cache keys (5 minutes)
SHARED_TIDE_CACHE_KEY = "tide_api_cache_v1"
SHARED_TIDE_INFLIGHT_KEY = "tide_api_inflight_v1"

# Negative-cache TTL for failed requests
TIDE_NEGATIVE_TTL_DEFAULT = 360  # seconds (6 minutes)

# Tide numeric constants
TIDE_STRENGTH_NORMALIZATION_DENOM = 2.5  # denom used to normalize tide amplitude to 0..1
TIDE_MIN_REQUIRED_BRACKETING = True  # require both bracketing extremes for interpolation (strict)

# Canonical strings
TIDE_CONFIDENCE_SOURCE = "worldtides_api_extremes_only"
TIDE_SOURCE = "worldtides_api"

WEATHER_FETCHER_CACHE_TTL_DEFAULT = 60 * 60  # seconds (default 1 hour)

# Tide phase offset default (minutes) — allows per-location tuning of tide predictions
TIDE_PHASE_OFFSET_MINUTES_DEFAULT = 0

# ----- New: coordinate rounding precision used for cache keys and key-coalescing across components -----
# Use a single canonical rounding precision everywhere to ensure consistent cache keys and predictable hits.
COORD_ROUND_DECIMALS = 5  # recommended default: 5 decimals ≈ 1.1 meters

# ----- Config keys used by the flow and entry options -----
CONF_NAME = "name"
CONF_LATITUDE = "latitude"
CONF_LONGITUDE = "longitude"
CONF_SPECIES_ID = "species"  # canonical key used in entry.options for chosen species
CONF_SPECIES_REGION = "species_region"
CONF_HABITAT_PRESET = "habitat_preset"
CONF_TIME_PERIODS = "time_periods"
# CONF_THRESHOLDS removed (no nested thresholds object — thresholds are top-level keys)
CONF_TIMEZONE = "timezone"
CONF_ELEVATION = "elevation"
CONF_AUTO_APPLY_THRESHOLDS = "auto_apply_thresholds"

# NEW: per-entry configurable factor weights (normalized floats summing to 1.0)
CONF_FACTOR_WEIGHTS = "factor_weights"

# TTL config keys (Option B)
CONF_FETCH_CACHE_TTL = "fetch_cache_ttl"       # coordinator shared fetch cache TTL (seconds)
CONF_TIDE_TTL = "tide_ttl"                     # tide proxy ttl (seconds)
CONF_WEATHER_CACHE_TTL = "weather_cache_ttl"   # weather_fetcher internal cache duration (seconds)
CONF_TIDE_PHASE_OFFSET_MINUTES = "tide_phase_offset_minutes"  # tide phase offset (minutes)

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