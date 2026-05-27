# How to Configure Safety Limits

Safety limits prevent the integration from recommending fishing when conditions are dangerous. When a limit is breached, the score is capped — even if all other factors are perfect.

## Prerequisites

- Ocean Fishing Assistant is installed and configured
- You know your general fishing environment (beach, rocks, harbour, reef)

## Step 1: Open the options

1. Go to **Settings > Devices & Services**
2. Find **Ocean Fishing Assistant** and click **Configure**
3. The options dialog opens

## Step 2: Adjust safety thresholds

The options form shows all safety settings with your current values. The defaults depend on which habitat preset you chose during setup.

### Wind speed

**What it controls:** Sustained wind speed. High wind makes casting difficult, affects boat handling, and fish often stop feeding.

- **Default (Rocky Point):** 10.7 m/s (24 mph / 5 Beaufort)
- **Default (Open Beach):** 8.0 m/s (18 mph)
- **Default (Harbour):** 12.0 m/s (27 mph)
- **Default (Offshore Reef):** 10.0 m/s (22 mph)

**Try this:** If you fish from a sheltered spot, you can raise this. If you're on an exposed beach, lower it.

### Gust speed

**What it controls:** Peak wind gusts. More important than sustained wind for safety — a sudden gust can knock you off balance or capsize a small boat.

- **Default:** 15 m/s (34 mph)

**Try this:** Lower to 12 m/s (27 mph) if you fish from a kayak or small boat. Raise to 18 m/s (40 mph) if you're on a large boat or sheltered pier.

### Wave height

**What it controls:** Significant wave height. Affects boat safety and shore fishing accessibility.

- **Default (Rocky Point):** 2.0 m (6.6 ft)
- **Default (Open Beach):** 1.5 m (5 ft)
- **Default (Harbour):** 1.0 m (3.3 ft)
- **Default (Offshore Reef):** 2.5 m (8.2 ft)

**Try this:** Lower if you're wade fishing or have a small boat. Raise if you're on a larger vessel.

### Visibility

**What it controls:** Horizontal visibility. Poor visibility can mean fog, heavy rain, or murky water.

- **Default:** 5.0 km (3.1 miles)

**Try this:** Set higher (10 km) if you need clear conditions for spotting fish. Set lower (2 km) if visibility isn't important to you.

### Swell period

**What it controls:** Time between wave crests. Shorter periods mean steeper, more uncomfortable seas. Longer periods mean gentler, rolling swell.

- **Default:** 6.0 seconds

**Try this:** If you're prone to seasickness, raise to 8 seconds. If you're fishing from shore, swell period matters less — consider lowering to 4 seconds.

### Precipitation chance

**What it controls:** Probability of rain. Heavy rain affects visibility, fish behavior, and your comfort.

- **Default:** 60%

**Try this:** Set to 80% if you don't mind fishing in the rain. Set to 30% if rain ruins your trip.

## Step 3: Save and observe

Click **Submit**. The integration recalculates the score with your new limits. The change takes effect immediately — you don't need to restart Home Assistant.

Check the sensor's `current_forecast.breaches` attribute to see if any safety limits are actively capping your score:

```yaml
breaches:
  - variable: "wind"
    value: "12.5 m/s"
    severity: "caution"
    reason: "wind_near_preferred_max"
    advice: "Sea Bass prefers lighter winds"
```

## Fine-tuning tips

- **Start with the habitat defaults** — they're sensible for each environment type
- **If the score is always capped** — one or more safety limits are too strict. Check `breaches` to see which one.
- **If the score never drops below 80** — your safety limits may be too loose, or your species profile is a great match for current conditions. Both are fine.
- **Change one limit at a time** — you'll learn which matters most for your fishing spot.
- **Safety limits are not scoring factors** — they don't gradually reduce the score like wind speed does. They cap the score when conditions cross a threshold. If you want a factor to have gradual influence, adjust its factor weight instead.

## Related

- [Understanding scores](understanding-scores.md) — how factor weights and safety interact
- [Getting started](getting-started.md) — install and configure the integration
- [Sensor attribute reference](sensor-attributes.md) — where to find breach data in attributes
