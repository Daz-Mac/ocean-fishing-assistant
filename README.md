# Ocean Fishing Assistant

A Home Assistant integration that scores ocean fishing conditions by combining live weather, marine, tide, moon, and species data into a single 0-100 score.

![Integration](https://img.shields.io/badge/HA-custom_component-blue) [![HACS validation](https://github.com/Daz-Mac/ocean-fishing-assistant/actions/workflows/validate.yml/badge.svg)](https://github.com/Daz-Mac/ocean-fishing-assistant/actions/workflows/validate.yml) [![Home Assistant validation](https://github.com/Daz-Mac/ocean-fishing-assistant/actions/workflows/hassfest.yml/badge.svg)](https://github.com/Daz-Mac/ocean-fishing-assistant/actions/workflows/hassfest.yml)

## Features

- **Fishing score** — 0-100 rating of current conditions based on 9 weighted factors
- **Species targeting** — score reflects how well conditions match specific fish species or general regional profiles
- **Per-factor breakdown** — see exactly which factors (wind, tide, waves, moon, temperature, etc.) boost or penalize the score
- **Period forecasts** — scores for each time period today and the next 5 days
- **Safety limits** — configurable thresholds for wind, waves, gust, visibility, swell, and precipitation
- **Multiple habitat presets** — rocky point, open beach, harbour, offshore reef
- **Tide tracking** — next high/low tide, tide phase per timestamp, tide strength
- **Moon phase** — current moon phase and phase-based scoring aligned with species preferences
- **Unit conversion** — metric or imperial display throughout
- **Configurable factor weights** — tune how much each factor matters to you
- **Wind direction scoring** — optionally set preferred wind directions

## Scoring Factors

| Factor | Default weight | Description |
|--------|---------------|-------------|
| Tide | 25% | How well the tide phase (rising/falling/high) matches species preferences |
| Waves | 15% | Wave height relative to species tolerance and safety limits |
| Time of day | 15% | How well the time matches species preferred feeding times (dawn/dusk periods, specific hours) |
| Wind | 10% | Wind speed relative to species preferences |
| Pressure | 10% | Pressure trend — rising pressure (>2 hPa/3h) is excellent, dropping is poor |
| Season | 10% | How well the current month matches species preferred season |
| Wind direction | 5% | Angular distance to preferred wind directions (optional) |
| Moon | 5% | Moon phase match against species preference (full/new/quarter) |
| Temperature | 5% | Water temperature alignment with species preferred range |

Each factor scores 0-10. The overall score is a weighted average mapped to 0-100.

## Installation

### Via HACS (custom repository)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?category=Integration&repository=https%3A%2F%2Fgithub.com%2FDaz-Mac%2Focean-fishing-assistant)

1. Click the badge above or go to **HACS > Integrations**
2. Click the three dots in the top-right corner and select **Custom repositories**
3. Add the repository URL: `https://github.com/Daz-Mac/ocean-fishing-assistant`
4. Category: **Integration**
5. Click **Add**
6. The Ocean Fishing Assistant should now appear in HACS. Click **Download**
7. Restart Home Assistant

### Manual installation

1. Copy the `ocean-fishing-assistant/custom_components/ocean_fishing_assistant/` directory into your Home Assistant `custom_components/` directory
2. Restart Home Assistant

```bash
# Example: copy from download directory to HA config
cp -r ocean_fishing_assistant /path/to/config/custom_components/
```

### After installation

1. Go to **Settings > Devices & Services > Add Integration**
2. Search for **Ocean Fishing Assistant**
3. Follow the configuration flow

## Configuration

The integration uses Home Assistant's config flow (UI-based setup). There are two modes:

### Normal mode (recommended)

Steps you through the essential settings:

1. **Location** — Set a name, latitude, longitude, and your [World Tides API key](https://www.worldtides.info)
2. **Profile type** — Choose a general regional profile or target a specific species
3. **Region** (species mode only) — Select your fishing region
4. **Species** (species mode only) — Choose your target species from region-filtered list
5. **Habitat preset** — Select your fishing environment (Rocky Point, Open Beach, Harbour, Offshore Reef)
6. **Time periods** — Full day (4 periods) or Dawn & Dusk only
7. **Wind direction** — Optionally enable wind direction as a scoring factor
8. **Display units** — Metric (m, km/h, °C) or Imperial (ft, mph, °F)
9. **Safety thresholds** — Configure limits for wind, waves, gust, visibility, swell, and precipitation

### Advanced mode

All normal mode steps plus:

- **Update interval** — How often to refresh data (default: 30 min)
- **Cache TTLs** — Separate TTLs for fetch cache, tide cache, and weather cache
- **Tide phase offset** — Local offset in minutes for tide predictions (±180 min)
- **Factor weights** — Adjust the relative importance of each scoring factor (must sum to 100%)

### Options (post-setup)

After setup, you can change settings via **Settings > Devices & Services > Ocean Fishing Assistant > Configure**.

Editable options include:
- Time periods mode
- Safety thresholds (wind, waves, gust, visibility, swell, precipitation)
- Factor weights
- Tide phase offset
- Wind direction preferences
- Raw data exposure toggle

## Sensor Entity

The integration creates one sensor entity per config entry:

```
sensor.ocean_fishing_assistant_{entry_id}_score
```

**State**: The current fishing score as an integer (0-100).

| Range | Icon | Meaning |
|-------|------|---------|
| 70-100 | 🐟 mdi:fish | Excellent conditions |
| 50-69 | 🐟 mdi:fish-off | Fair conditions |
| 0-49 | ⚠️ mdi:alert-circle-outline | Poor conditions |

### Attributes

The sensor exposes detailed attributes for dashboarding and automations:

| Attribute | Type | Description |
|-----------|------|-------------|
| `current_forecast` | dict | Current timestamp's full forecast with score, components, safety, and breaches |
| `remainder_of_today_periods` | dict | Today's remaining time periods with scores |
| `next_5_day_periods` | dict | Forecast periods for the next 5 days, grouped by date |
| `next_high_tide` | dict or null | Next high tide timestamp |
| `next_low_tide` | dict or null | Next low tide timestamp |
| `moon_phase_name` | string | Current moon phase name (New Moon, Waxing Crescent, etc.) |
| `moon_phase` | float | Raw moon phase value (0-1, only with expose_raw) |
| `current_temperature` | string | Current temperature with unit |
| `current_wind_speed` | string | Current wind speed with unit |
| `current_wind_gust` | string | Current wind gust with unit |
| `current_pressure` | string | Current pressure with unit |
| `current_wave_height` | string | Current wave height with unit |
| `current_swell_period_s` | float | Current swell period in seconds |
| `profile_used` | dict | The species profile used for scoring (includes scientific name and info link) |
| `units` | string | Display units (metric/imperial) |
| `raw_output_enabled` | boolean | Whether raw data exposure is enabled |
| `attribution` | string | Data source attribution |
| `per_timestamp_forecasts` | list | Full per-timestamp forecast data (only with expose_raw) |

### current_forecast structure

```yaml
current_forecast:
  timestamp: "2026-05-26T14:00:00Z"
  index: 14
  score_10: 7.5
  score_100: 75
  components:
    tide:
      score_10: 10.0
      score_100: 100
      tide_phase: "rising"
    wind:
      score_10: 8.5
      wind_speed: "18.5 km/h"
    waves:
      score_10: 7.3
      wave_height: "1.2 m"
    time:
      score_10: 10.0
    pressure:
      score_10: 6.2
      pressure_delta: "0.8 hPa"
    season:
      score_10: 10.0
    moon:
      score_10: 10.0
      moon_phase_name: "Waxing Gibbous"
    temperature:
      score_10: 9.1
      temperature: "22.0 °C"
    wind_direction:
      score_10: 10.0
      wind_direction_deg: 135.0
  safety_values:
    wind_gust: "28.5 km/h"
    visibility: "15.0 km"
  breaches:
    - variable: "wind"
      value: "7.2 m/s"
      severity: "caution"
      reason: "wind_near_preferred_max"
      advice: "Sea Bass prefers lighter winds"
  tide_phase: "rising"
```

## Species Reference

### Regions

| Region ID | Name | Species count |
|-----------|------|---------------|
| `gibraltar_strait` | Gibraltar Strait | 3 |
| `south_africa_kzn` | South Africa KZN | 9 |
| `usa_florida_keys` | USA Florida Keys | 4 |
| `australia_gold_coast` | Australia Gold Coast | 4 |

### Species by Region

#### Gibraltar Strait

| Species | Common name | Scientific name | Key preferences |
|---------|-------------|-----------------|-----------------|
| `sea_bass` | Sea Bass | *Dicentrarchus labrax* | Temp 10-20°C, wind 0-6 m/s, waves <2m, tide rising/high, full moon |
| `mackerel` | Mackerel | *Scomber scombrus* | Temp 12-22°C, wind 0-8 m/s, waves <1.5m, tide falling |
| `bonito` | Atlantic Bonito | *Sarda sarda* | Temp 16-24°C, wind 0-8 m/s, waves <1.8m, any tide |

#### South Africa KZN

| Species | Common name | Scientific name | Key preferences |
|---------|-------------|-----------------|-----------------|
| `shad` | Shad / Elf | *Pomatomus saltatrix* | Temp 16-24°C, wind 0-6 m/s, waves <2m, tide high |
| `kob` | Kob / Kabeljou | *Argyrosomus japonicus* | Temp 13-16°C, wind 0-6 m/s, waves <3m, tide rising |
| `natal_stumpnose` | Natal Stumpnose | *Rhabdosargus sarba* | Temp 18-24°C, wind 0-6 m/s, waves <1.5m, tide rising/high |
| `blacktail` | Blacktail / Rock Bream | *Diplodus capensis* | Temp 18-24°C, wind 0-8 m/s, waves <1.5m, any tide |
| `galjoen` | Galjoen | *Dichistius capensis* | Temp 16-22°C, wind 0-6 m/s, waves <2m, tide high |
| `mahi_mahi` | Mahi Mahi / Dorado | *Coryphaena hippurus* | Temp 22-29°C, wind 0-10 m/s, waves <2.5m |
| `yellowfin_tuna` | Yellowfin Tuna | *Thunnus albacares* | Temp 20-28°C, wind 0-12 m/s, waves <3m |
| `king_mackerel` | King Mackerel | *Scomberomorus cavalla* | Temp 21-27°C, wind 0-10 m/s, waves <2m |
| `wahoo` | Wahoo | *Acanthocybium solandri* | Temp 22-29°C, wind 0-12 m/s, waves <3m |
| `white_musselcracker` | White Musselcracker | *Sparodon durbanensis* | Temp 18-25°C, wind 0-8 m/s, waves <1.5m |

#### USA Florida Keys

| Species | Common name | Scientific name | Key preferences |
|---------|-------------|-----------------|-----------------|
| `tarpon` | Tarpon | *Megalops atlanticus* | Temp 22-30°C, wind 0-8 m/s, waves <1.5m, full/new moon |
| `snook` | Snook | *Centropomus undecimalis* | Temp 22-30°C, wind 0-8 m/s, waves <1.5m, tide rising/high |
| `bonefish` | Bonefish | *Albula vulpes* | Temp 20-28°C, wind 0-7 m/s, waves <1m, tide rising/high |
| `mangrove_snapper` | Mangrove Snapper | *Lutjanus griseus* | Temp 22-29°C, wind 0-9 m/s, waves <2m, tide rising/high |

#### Australia Gold Coast

| Species | Common name | Scientific name | Key preferences |
|---------|-------------|-----------------|-----------------|
| `tailor` | Tailor | *Pomatomus saltatrix* | Temp 17-22°C, wind 0-10 m/s, waves <2.5m, tide rising/high |
| `yellowfin_bream` | Yellowfin Bream | *Acanthopagrus australis* | Temp 18-24°C, wind 0-8 m/s, waves <2m, tide rising/high |
| `sand_whiting` | Sand Whiting | *Sillago ciliata* | Temp 21-26°C, wind 0-6 m/s, waves <1.5m, tide rising/high |
| `mulloway` | Mulloway / Jewfish | *Argyrosomus japonicus* | Temp 16-21°C, wind 0-8 m/s, waves <2.5m, tide high/falling |

### General Profiles

| Profile ID | Name | Description |
|------------|------|-------------|
| `general_mixed_global` | General Mixed Global | Broad mixed-species profile with wave preference and dawn/dusk focus |

## Adding Species Profiles

Species are defined in `custom_components/ocean_fishing_assistant/species_profiles.json`.

### Schema

```json
{
  "species": {
    "species_id": {
      "common_name": "Display name",
      "scientific_name": "Genus species",
      "info": "URL to species reference",
      "emoji": "🐟",
      "regions": ["region_id"],
      "preferred_months": [1, 2, 3],
      "preferred_temp_c": [min, max],
      "preferred_wind_m_s": [min, max],
      "preferred_swell_period_s": [min, max],
      "max_wave_height_m": 2.0,
      "preferred_tide_phase": ["rising", "falling", "high", "low"],
      "preferred_times": [
        {"start_hour": 5, "end_hour": 9},
        "dawn",
        "dusk"
      ],
      "moon_preference": ["full", "new", "first_quarter", "last_quarter"]
    }
  }
}
```

### Field reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `common_name` | string | yes | User-facing species name |
| `scientific_name` | string | yes | Latin binomial name |
| `info` | string | no | URL to species reference page (Wikipedia, FishBase, etc.) |
| `emoji` | string | no | Emoji for display in config flow |
| `regions` | string[] | yes | Region IDs this species is found in (match keys in `regions` object) |
| `preferred_months` | int[] | no | Months (1-12) when this species is active; empty = all months |
| `preferred_temp_c` | float[2] | no | Preferred water temperature range [min, max] in °C |
| `preferred_wind_m_s` | float[2] | no | Preferred wind speed range [min, max] in m/s |
| `preferred_swell_period_s` | float[2] | no | Preferred swell period range [min, max] in seconds |
| `max_wave_height_m` | float | no | Maximum wave height in meters |
| `preferred_tide_phase` | string[] | no | Preferred tide phases: `rising`, `falling`, `high`, `low`, or `any` |
| `preferred_times` | mixed[] | no | Preferred times: hour objects `{start_hour, end_hour}` or tokens `"dawn"`, `"dusk"` |
| `moon_preference` | string[] | no | Preferred moon phases: `full`, `new`, `first_quarter`, `last_quarter` |

### Adding a new species

1. Add the species entry under `species` in `species_profiles.json`
2. Ensure the `regions` field references existing region IDs (or add a new region under `regions`)
3. Restart Home Assistant

### Adding a new region

Add an entry to the `regions` object:

```json
"my_new_region": {
  "id": "my_new_region",
  "name": "My New Region"
}
```

Then reference it in species profile `regions` arrays.

## Example Lovelace Dashboard

The integration works with HA's native entities and any card that can display sensor attributes. Here's a simple dashboard layout using the built-in entities card and [apexcharts-card](https://github.com/RomRider/apexcharts-card) for tide graphs:

See [examples/lovelace-dashboard.yaml](examples/lovelace-dashboard.yaml) for a complete dashboard template.

## Troubleshooting

### Sensor shows "unavailable"

1. Check that your **World Tides API key** is valid and has not expired
2. Verify that the **latitude/longitude** coordinates are correct (must be valid ocean locations)
3. Check the Home Assistant logs for error messages:
   - `WorldTides authentication failed (401)` → invalid or expired API key
   - `Timezone resolution failed` → location may be in a region where timezone lookup fails
   - `No matching forecast found` → data freshness issue, coordinator will retry

### Score shows unexpected values

1. Check which **species profile** is selected via the sensor's `profile_used` attribute
2. Review the **component scores** in `current_forecast.components` — each factor shows its individual score
3. Check **safety_values** and **breaches** — safety limits may be capping the score
4. Verify **habitat preset** matches your actual fishing environment (affects default safety limits)

### Data not updating

- Default update interval is **30 minutes**. You can change this in advanced config
- Open-Meteo and World Tides both have rate limits; the integration respects cached data within configured TTLs
- Check if the `_async_update_data` method completed successfully in logs

### Configuration flow errors

| Error | Cause |
|-------|-------|
| `invalid_coordinates` | Latitude must be -90 to 90, longitude -180 to 180 |
| `missing_world_tides_api_key` | A World Tides API key is required |
| `title_exists` | An entry with that name already exists |
| `update_interval_too_small` | Minimum update interval is 30 seconds |
| `sum_not_100` | Factor weights must add up to 100% |

## Data Sources

- **Weather**: [Open-Meteo](https://open-meteo.com/) — free weather forecast API (no API key required)
- **Marine**: [Open-Meteo Marine](https://open-meteo.com/en/docs/marine-weather-api) — free marine forecast API (no API key required)
- **Tide**: [World Tides](https://www.worldtides.info/) — tide predictions API (free API key available)
- **Astronomy**: [Skyfield](https://rhodesmill.org/skyfield/) — moon phase and sun position calculations
- **Timezones**: [timezonefinder](https://github.com/mattalbertson/timezonefinder) — IANA timezone resolution from coordinates

## Requirements

- **Home Assistant** 2025.12.5 or later
- **World Tides API key** (free at https://www.worldtides.info)
- Python dependencies (installed automatically): skyfield, jplephem, numpy, timezonefinder

## Technical Architecture

```
WeatherFetcher ──►  OFACoordinator  ──► DataFormatter  ──► OceanScoring  ──► Sensor
(Open-Meteo)        (DataUpdate        (validate &        (compute_score)       (OFASensor
  Marine API          Coordinator)       canonicalize)                            entity)
  TideProxy               ▲
  (World Tides +          │
   Skyfield moon)   ──────┘
```

The coordinator orchestrates fetching weather (Open-Meteo), marine data (Open-Meteo Marine), and tidal/moon data (World Tides API + Skyfield). The formatter converts raw API responses into a canonical structure. The scoring engine evaluates each timestamp against the selected species profile. The sensor entity exposes the current score and rich forecast attributes.

## Contributing

Contributions are welcome! Areas to contribute:

- **Species profiles** — Add profiles for fish species and regions not yet covered
- **New data sources** — Additional weather or ocean data providers
- **Dashboard cards** — Custom Lovelace cards for visual tide graphs and score displays
- **Bug fixes and improvements** — Open an issue or pull request on GitHub

## Release Process

This project uses automated releases via GitHub Actions. To publish a new version:

1. Ensure all changes are merged to the default branch
2. Push a tag matching `v*.*.*` (e.g., `v0.2.0`):
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```
3. The release workflow automatically creates a GitHub Release with auto-generated release notes
4. HACS will detect the new release and prompt users to update

Version numbering follows [semantic versioning](https://semver.org/):
- **Patch** (v0.1.x) — Bug fixes and minor changes
- **Minor** (v0.x.0) — New features, backward-compatible
- **Major** (x.0.0) — Breaking changes

## License

This project is distributed under the terms of the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
