# Sensor Attribute Reference

The Ocean Fishing Assistant creates one sensor entity per configured location:

```
sensor.ocean_fishing_assistant_<entry_id>_score
```

## State

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `state` | integer | 0-100 | Current fishing score. Higher is better. |

## Attributes

### Current forecast

The main attribute containing the current period's full score breakdown.

| Key | Type | Description |
|-----|------|-------------|
| `current_forecast.timestamp` | string (ISO 8601) | The timestamp this forecast applies to |
| `current_forecast.index` | integer | Hour index of the forecast (0-23) |
| `current_forecast.score_10` | float | Raw score on a 0-10 scale |
| `current_forecast.score_100` | integer | Score mapped to 0-100 scale |
| `current_forecast.safety_capped` | boolean | Whether safety limits reduced the score |
| `current_forecast.tide_phase` | string | Current tide phase: `rising`, `falling`, or `flat` |

#### Components

Each scoring factor under `current_forecast.components`:

| Path | Type | Values |
|------|------|--------|
| `components.tide.score_100` | integer | 0-100 |
| `components.tide.tide_phase` | string | `rising`, `falling`, or `flat` |
| `components.wind.score_100` | integer | 0-100 |
| `components.wind.wind_speed` | string | e.g., `"18.5 km/h"` or `"11.5 mph"` |
| `components.wind_direction.score_100` | integer | 0-100 |
| `components.wind_direction.wind_direction_deg` | float | Wind direction in degrees |
| `components.waves.score_100` | integer | 0-100 |
| `components.waves.wave_height` | string | e.g., `"1.2 m"` or `"3.9 ft"` |
| `components.time.score_100` | integer | 0-100 |
| `components.pressure.score_100` | integer | 0-100 |
| `components.pressure.pressure_delta` | string | e.g., `"0.8 hPa"` |
| `components.season.score_100` | integer | 0-100 |
| `components.moon.score_100` | integer | 0-100 |
| `components.moon.moon_phase_name` | string | e.g., `"Waxing Gibbous"` |
| `components.temperature.score_100` | integer | 0-100 |
| `components.temperature.temperature` | string | e.g., `"22.0 °C"` or `"71.6 °F"` |

#### Safety data

| Path | Type | Description |
|------|------|-------------|
| `current_forecast.safety_values.wind_gust` | string | Current wind gust with unit |
| `current_forecast.safety_values.visibility` | string | Current visibility |
| `current_forecast.safety_values.swell_period_s` | float | Swell period in seconds |
| `current_forecast.safety_values.precipitation_probability` | integer | Precipitation chance percentage |
| `current_forecast.breaches` | list | Active safety breaches (empty if none) |

Each breach entry:

```yaml
- variable: "wind"
  value: "7.2 m/s"
  severity: "caution"            # or "warning"
  reason: "wind_near_preferred_max"
  advice: "Sea Bass prefers lighter winds"
```

### Period forecasts

**Remainder of today:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `remainder_of_today_periods` | dict | Scores for remaining time periods today |

Each key is a period name (`period_00_06`, `period_06_12`, `period_12_18`, `period_18_24`) with the same structure as `current_forecast` (score, components, safety) plus two period-specific fields:

| Field | Type | Description |
|-------|------|-------------|
| `spring_tide_bonus` | int | Bonus points added (0 or 10) on full/new moon days |
| `safety.unsafe` | bool | Whether any safety limit is breached in this period |
| `safety.caution` | bool | Whether any safety limit is near-breach |
| `safety.reasons` | string[] | Reason codes for breaches (e.g. `["wind>15"]`) |

**5-day forecast:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `next_5_day_periods` | dict | Forecast grouped by date |

```yaml
next_5_day_periods:
  2026-05-28:
    period_06_12:
      score_100: 83
      tide_phase: "rising"
      components:
        wind:
          score_100: 90
          wind_speed: "12.3 km/h"
        ...
    period_12_18:
      score_100: 67
      ...
```

### Tide data

| Attribute | Type | Description |
|-----------|------|-------------|
| `next_high_tide.timestamp` | string or null | ISO 8601 timestamp of next high tide |
| `next_high_tide.height` | string | Height with unit |
| `next_low_tide.timestamp` | string or null | ISO 8601 timestamp of next low tide |
| `next_low_tide.height` | string | Height with unit |

Returns `null` if tide data is unavailable.

### Moon phase

| Attribute | Type | Description |
|-----------|------|-------------|
| `moon_phase_name` | string | Current moon phase name |
| `moon_phase` | float | Raw moon phase 0-1 (only with `expose_raw` enabled) |

Moon phase names: `New Moon`, `Waxing Crescent`, `First Quarter`, `Waxing Gibbous`, `Full Moon`, `Waning Gibbous`, `Last Quarter`, `Waning Crescent`

### Current conditions

Summary attributes for quick dashboard access:

| Attribute | Type | Description |
|-----------|------|-------------|
| `current_temperature` | string | Current temperature with unit |
| `current_wind_speed` | string | Current wind speed with unit |
| `current_wind_gust` | string | Current wind gust with unit |
| `current_pressure` | string | Current pressure with unit |
| `current_wave_height` | string | Current wave height with unit |
| `current_swell_period_s` | float | Current swell period in seconds |

### Profile info

| Attribute | Type | Description |
|-----------|------|-------------|
| `profile_used.common_name` | string | Species or profile display name |
| `profile_used.scientific_name` | string | Scientific name (species mode only) |
| `profile_used.info` | string | Reference URL (if available) |

### Runtime info

| Attribute | Type | Description |
|-----------|------|-------------|
| `units` | string | `metric` or `imperial` |
| `raw_output_enabled` | boolean | Whether raw data exposure is on |
| `attribution` | string | Data source attribution notice |

### Raw data (optional)

If `expose_raw` is enabled in options, the sensor also includes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `per_timestamp_forecasts` | list | Full per-timestamp forecast data with raw values |

This is disabled by default because it adds significant attribute size.

## Using attributes in automations

**Fire when score drops below 30:**

```yaml
trigger:
  - platform: numeric_state
    entity_id: sensor.ocean_fishing_assistant_123_score
    below: 30
action:
  - service: notify.mobile_app_phone
    data:
      message: "Fishing conditions poor — wind exceeds limits"
```

**Check if a specific breach is active:**

```yaml
condition:
  - condition: template
    value_template: >
      {{ state_attr('sensor.ocean_fishing_assistant_123_score',
         'current_forecast').breaches | selectattr('variable', 'equalto', 'wind') | list | length > 0 }}
```

**Read the best upcoming period:**

```yaml
template:
  - sensor:
      - name: "Best fishing period tomorrow"
        state: >
          {% set periods = state_attr('sensor.ocean_fishing_assistant_123_score',
             'next_5_day_periods') %}
          {% if periods and periods[0] %}
            {% set tomorrow = periods.keys() | first %}
            {% set best = periods[tomorrow].values() | max(attribute='score_100') %}
            {{ best.score_100 }}
          {% else %}
            unavailable
          {% endif %}
```

## Related

- [Understanding scores](understanding-scores.md) — how scores are calculated
- [Configuring safety limits](configuring-safety.md) — adjust safety thresholds
- [Getting started](getting-started.md) — install and configure the integration
