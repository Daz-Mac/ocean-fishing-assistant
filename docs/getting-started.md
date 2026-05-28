# Getting Started with Ocean Fishing Assistant

This tutorial walks you through installing the integration, configuring your first fishing location, and reading your first fishing score. You'll go from zero to a working sensor in about 10 minutes.

## What you'll need

- **Home Assistant** 2025.12.5 or later running and accessible via the web UI
- **A World Tides API key** (free at https://www.worldtides.info) — sign up and copy your key
- **Coordinates** for a fishing spot near you (latitude/longitude)

## Step 1: Install the integration

### If you use HACS (recommended)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?category=Integration&repository=https%3A%2F%2Fgithub.com%2FDaz-Mac%2Focean-fishing-assistant)

Or install manually:

1. Go to **HACS > Integrations**
2. Click the three dots in the top-right corner and select **Custom repositories**
3. Enter `https://github.com/Daz-Mac/ocean-fishing-assistant` with category **Integration**
4. Click **Add**, then **Download** on the integration card
5. **Restart Home Assistant**

### Without HACS (manual install)

Copy the `custom_components/ocean_fishing_assistant/` directory into your Home Assistant `custom_components/` directory and restart.

## Step 2: Add the integration

1. Go to **Settings > Devices & Services**
2. Click **Add Integration** (bottom-right)
3. Search for **Ocean Fishing Assistant** and select it

## Step 3: Configure your fishing spot

You'll walk through a multi-step configuration wizard:

### Location & API key

Enter:
- **Name** — a label for this location (e.g., "Rosia Bay")
- **Latitude** — e.g., `36.14` (Gibraltar)
- **Longitude** — e.g., `-5.36`
- **World Tides API Key** — paste the key from step 1

Click **Submit**.

### Profile type

Choose how to score conditions:

- **General region profile (mixed)** — uses a broad mixed-species profile. Best if you fish for whatever is biting.
- **Target a specific species** — scores conditions against a specific fish's preferences. Best if you're after a particular species.

If you choose **General**, it skips ahead to habitat. If you choose **Specific**, you'll pick a region and then a species.

### Region & species (species mode only)

If you selected species targeting:

1. Select your **region** (e.g., "Gibraltar Strait", "South Africa KZN")
2. Select a **species** from the filtered list (e.g., "Sea Bass", "Mackerel")

### Habitat preset

Select the environment you'll fish in:

| Habitat | Best for |
|---------|----------|
| Open Sandy Beach | Surf fishing, wide beaches |
| Rocky Point/Jetty | Rocks, breakwaters, piers |
| Harbour/Pier | Sheltered harbors, marina walls |
| Offshore Reef | Boat fishing, deeper water |

Your habitat preset sets sensible default safety limits for that environment.

### Time periods

Choose how the integration slices the day:

- **Full Day (4 periods)** — Morning (00-06), Afternoon (06-12), Evening (12-18), Night (18-24)
- **Dawn & Dusk Only** — Focuses on the two best fishing windows around sunrise and sunset

### Wind direction (optional)

Toggle whether wind direction affects the score. If enabled, you'll select favorable wind directions (e.g., Easterly, Westerly) for your spot.

### Display units

Choose **Metric** (m, km/h, °C) or **Imperial** (ft, mph, °F).

### Safety thresholds

Set limits for conditions you consider unsafe:

| Setting | Default | What it does |
|---------|---------|-------------|
| Max wind speed | 10.7 m/s (24 mph) | Scores fall if wind exceeds this |
| Max gust speed | 15 m/s (34 mph) | Scores capped if gusts exceed this |
| Max wave height | 2.0 m (6.6 ft) | Scores fall if waves exceed this |
| Min visibility | 5.0 km (3.1 mi) | Scores fall if visibility drops below |
| Min swell period | 6.0 s | Shorter swell = worse conditions |
| Max precipitation | 60% | Heavy rain triggers safety cap |

### Advanced options (optional)

In advanced mode you can also set:
- **Update interval** — how often data refreshes (default: 30 min, minimum: 30 s)
- **Cache TTLs** — how long fetched data stays cached
- **Tide phase offset** — adjust for local tide timing differences

## Step 4: Read your first fishing score

After configuration completes, Home Assistant creates a sensor entity:

```
sensor.ocean_fishing_assistant_<entry_id>_score
```

You can find it in **Settings > Devices & Services > Ocean Fishing Assistant**. The sensor shows:

- **State**: Current fishing score (0-100)
- **Attributes**: Breakdown of all 9 scoring factors, tide phase, moon phase, and 5-day forecast

The score works like this:
| Range | Meaning |
|-------|---------|
| 70-100 | Excellent conditions — good fishing |
| 50-69 | Fair conditions — fishable |
| 0-49 | Poor conditions — one or more factors are working against you |

## Step 5: Add it to your dashboard

1. Go to your dashboard, click **Edit Dashboard** (pencil icon)
2. Click **Add Card** and search for **Ocean Fishing Assistant**
3. Select the sensor entity you just configured
4. Optional: set a custom title and forecast display options, then **Save**

You should see the card with your current score, conditions grid, and forecast periods.

## What you built

You now have:
- A working fishing conditions sensor that updates automatically
- A dashboard card showing the current score, factor breakdown, and forecast
- Configurable safety limits tuned to your habitat
- 5-day forecast of fishing conditions

Next steps:
- [Understanding scores](understanding-scores.md) — learn how the 0-100 score is calculated
- [Configuring safety limits](configuring-safety.md) — fine-tune your safety thresholds
- [Sensor attribute reference](sensor-attributes.md) — full list of available sensor data
