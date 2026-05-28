# Understanding Fishing Scores

The Ocean Fishing Assistant produces a single 0-100 score that tells you how good conditions are for fishing right now. This document explains how that number is calculated so you can trust it, tune it, and understand why it says what it says.

## The short version

The score is a weighted average of 9 independent factors. Each factor scores 0-10 based on how well current conditions match your selected species profile. The weighted scores are summed, mapped to 0-100, and then **capped** if any safety limit is exceeded.

```
score = weighted_average(9 factors) × 10
if safety_breach → score = min(score, safety_cap)
```

## The 9 scoring factors

Each factor contributes to the final score based on its default weight:

| Factor | Weight | What it measures |
|--------|--------|-----------------|
| Tide | 25% | How well the tide phase (rising, falling, high, low) matches species preferences |
| Waves | 15% | Wave height relative to the species' tolerance |
| Time of day | 15% | How well the current time matches the species' preferred feeding hours |
| Wind | 10% | Wind speed relative to the species' preference |
| Pressure | 10% | Pressure trend — rising is good, dropping is poor |
| Season | 10% | Whether the current month falls in the species' active season |
| Wind direction | 5% | Angular distance to your preferred wind direction (only if enabled) |
| Moon | 5% | Moon phase match against species preference |
| Temperature | 5% | Water temperature alignment with species' preferred range |

### How each factor scores

Factors that match a range (temperature, wind, waves) use a linear scoring curve:

```
If value is within preferred range     → score = 10
If value is outside tolerable range    → score = 0
If value is in between                 → score = 10 × distance_from_tolerable / preferred_span
```

Factors that match a category (tide phase, moon phase) score as:

```
If current phase matches any preferred phase  → score = 10
If no match                                   → score = 0
```

### Customizing weights

You can adjust factor weights in the integration's **Configure** settings. The weights must always add up to 100%. Increasing a factor's weight makes it pull the score more when conditions are bad for that factor.

## Safety capping

Safety limits are separate from scoring. They act as a **hard ceiling** on the final score. If any safety limit is breached (e.g., wind gust exceeds your max), the score is capped proportionally to the severity.

Safety limits are configured per-habitat preset but can be customized:

| Limit | Default (Rocky Point) | Effect |
|-------|----------------------|--------|
| Max wind speed | 10.7 m/s (24 mph) | Score capped at 30 if exceeded |
| Max gust speed | 15 m/s (34 mph) | Score capped at 20 if exceeded |
| Max wave height | 2.0 m (6.6 ft) | Score capped at 30 if exceeded |
| Min visibility | 5.0 km (3.1 mi) | Score capped at 40 if below |
| Min swell period | 6.0 s | Score capped at 40 if below |
| Max precipitation | 60% | Score capped at 30 if exceeded |

The safety cap has a **band** — the score doesn't jump to zero immediately. It scales down as the severity increases, giving you a gradual warning rather than a sudden cutoff.

## Reading the sensor

The sensor entity shows the current score as its state. The attributes show the full breakdown:

```yaml
current_forecast:
  score_100: 75                       # Overall score
  components:
    tide:
      score_100: 100                  # Individual factor score
      tide_phase: "rising"            # Current value
    wind:
      score_100: 85
      wind_speed: "18.5 km/h"
    waves:
      score_100: 73
      wave_height: "1.2 m"
  safety_capped: false                # Whether safety limits reduced the score
  breaches: []                        # Active safety breaches, if any
```

### Interpreting the numbers

- **Score 70-100**: Most factors are favorable. Good fishing conditions.
- **Score 50-69**: Mixed conditions. Some factors are good, some are marginal.
- **Score 30-49**: Several factors are poor. Fishable but not ideal.
- **Score 0-29**: Multiple factors are working against you. Consider waiting.

### Why is my score low?

Check the component scores in the sensor attributes. The factor with the lowest score is the main drag. Common reasons:

| Low factor | Likely cause |
|-----------|-------------|
| Tide | Tide phase doesn't match your species' preference |
| Wind | Too windy for the species, or wind exceeds safety limit |
| Waves | Too rough for the species or your safety setting |
| Season | The species isn't active this month |
| Time | Not in a preferred feeding window |
| Pressure | Pressure is dropping quickly |
| Temperature | Water is too warm or too cold for the species |

## Forecast scores

The sensor also provides forecast scores for the rest of today and the next 5 days. Each period (e.g., "Today 18-24h") gets its own score calculated the same way, using forecasted weather and tide data instead of current readings. This lets you plan ahead: "Thursday morning looks excellent, but the wind picks up by noon."

## Related

- [Getting started](getting-started.md) — install and configure the integration
- [Configuring safety limits](configuring-safety.md) — tune safety thresholds for your fishing style
- [Sensor attribute reference](sensor-attributes.md) — complete list of available sensor data
