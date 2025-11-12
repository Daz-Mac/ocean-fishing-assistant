# Strict sensor entity - surfaces errors loudly so callers see misconfiguration
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import math

from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.const import ATTR_ATTRIBUTION
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN, CONF_NAME

ATTRIBUTION = "Data provided by Open-Meteo"


def _parse_dt(v: Any) -> Optional[datetime]:
    """Parse a timestamp value into an aware UTC datetime, or None."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc) if v.tzinfo else v.replace(tzinfo=timezone.utc)
    try:
        # numeric epoch (secs or ms)
        if isinstance(v, (int, float)):
            val = float(v)
            if val > 1e12:  # probably milliseconds
                val = val / 1000.0
            return datetime.fromtimestamp(val, tz=timezone.utc)
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        parsed = datetime.fromisoformat(s)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _iso_z(dt: datetime) -> str:
    """Return ISO string ending with Z for an aware UTC datetime."""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _moon_phase_name(p: Optional[float]) -> Optional[str]:
    """Return human-friendly moon phase name for p in [0,1], or None."""
    if p is None:
        return None
    try:
        # normalize to 0..1
        val = float(p) % 1.0
    except Exception:
        return None
    # thresholds
    if val < 0.0625 or val >= 0.9375:
        return "New Moon"
    if val < 0.1875:
        return "Waxing Crescent"
    if val < 0.3125:
        return "First Quarter"
    if val < 0.4375:
        return "Waxing Gibbous"
    if val < 0.5625:
        return "Full Moon"
    if val < 0.6875:
        return "Waning Gibbous"
    if val < 0.8125:
        return "Last Quarter"
    return "Waning Crescent"


def _find_height_for_timestamp(target_iso: str, tide_ts: List[Any], tide_heights: List[Any]) -> Optional[float]:
    """Find tide height aligned with a target ISO timestamp. Returns float or None."""
    if not target_iso or not tide_ts or not tide_heights:
        return None
    try:
        target_dt = _parse_dt(target_iso)
        if target_dt is None:
            return None
    except Exception:
        return None
    # try exact match first
    for i, t in enumerate(tide_ts):
        dt = _parse_dt(t)
        if dt is None:
            continue
        if dt == target_dt:
            try:
                return float(tide_heights[i]) if tide_heights[i] is not None else None
            except Exception:
                return None
    # fallback: nearest by absolute time difference
    best_idx = None
    best_dt_diff = None
    for i, t in enumerate(tide_ts):
        dt = _parse_dt(t)
        if dt is None:
            continue
        diff = abs((dt - target_dt).total_seconds())
        if best_dt_diff is None or diff < best_dt_diff:
            best_dt_diff = diff
            best_idx = i
    if best_idx is not None:
        try:
            return float(tide_heights[best_idx]) if tide_heights[best_idx] is not None else None
        except Exception:
            return None
    return None


class OFASensor(CoordinatorEntity):
    def __init__(self, coordinator, name: str):
        """
        Strict CoordinatorEntity wrapper.

        - Fails loudly if coordinator lacks expected data.
        - Requires explicit `name` (no defaults or fallbacks).
        - The integration expects the configured name to be provided via
          the config entry data under the key CONF_NAME. The runtime
          name used for the entity will be prefixed with
          "ocean_fishing_assistant_" (unless the provided name already
          starts with that prefix).
        """
        if not name:
            raise RuntimeError("Sensor name must be provided (strict)")

        super().__init__(coordinator)

        # Enforce prefix deterministically; do not silently choose a name if missing.
        prefix = "ocean_fishing_assistant"
        if not name.startswith(prefix):
            name = f"{prefix}_{name}"

        self._attr_name = name

        # Provide a deterministic unique id so HA entities remain stable across restarts.
        # Include the entry/coordinator id and the configured name to avoid collisions
        # if multiple entries are created.
        try:
            safe_name = name.replace(" ", "_")
            self._attr_unique_id = f"{prefix}_{getattr(coordinator, 'entry_id', 'noentry')}_{safe_name}"
        except Exception:
            # If coordinator doesn't provide entry_id, still permit creation but flag absence via unique id being None.
            self._attr_unique_id = None

    @property
    def available(self) -> bool:
        return bool(self.coordinator.last_update_success and self.coordinator.data is not None)

    def _get_current_forecast(self) -> Optional[Dict[str, Any]]:
        data = self.coordinator.data or {}
        forecasts = data.get("per_timestamp_forecasts") or []
        if isinstance(forecasts, (list, tuple)) and len(forecasts) > 0:
            return forecasts[0]
        return None

    @property
    def state(self) -> Optional[int]:
        """Return state as integer 0..100 or raise on strict failures."""
        if not self.coordinator.data:
            raise RuntimeError("Coordinator data missing for OFA sensor (strict)")

        forecast = self._get_current_forecast()
        if forecast:
            sc = forecast.get("score_100")
            if sc is None:
                raise RuntimeError("Precomputed forecast present but score_100 is missing (strict)")
            return int(sc)

        raise RuntimeError("No precomputed per_timestamp_forecasts found in coordinator.data (strict)")

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """
        Strict attributes:

        - Fail loudly if coordinator.data is absent (consistent with strict policy).
        - Expose canonical fields required for debugging and integrations.
        - Add several current_* top-level convenience fields drawn from either
          forecast_raw.formatted_weather or the raw current snapshot.
        - Prefer tide.tide_phase and tide.next_high/next_low produced by TideProxy;
          normalize next_high/next_low to objects with timestamp + height_m when possible.
        """
        if not self.coordinator.data:
            raise RuntimeError("Coordinator data missing when building attributes (strict)")

        attrs: Dict[str, Any] = {}
        forecasts = self.coordinator.data.get("per_timestamp_forecasts")
        if forecasts is not None:
            attrs["per_timestamp_forecasts"] = forecasts

        current = self._get_current_forecast()
        if current is not None:
            attrs["current_forecast"] = current

        raw = self.coordinator.data.get("raw_payload") or self.coordinator.data
        attrs["raw_payload"] = raw

        # Add profile and safety details coming from per-timestamp/current forecast where present
        attrs["profile_used"] = (current.get("profile_used") if current else None)
        # Coordinator holds chosen units and safety limits in strict coordinator implementation
        attrs["units"] = getattr(self.coordinator, "units", None)
        attrs["safety_limits"] = getattr(self.coordinator, "safety_limits", None)
        attrs["safety"] = (current.get("safety") if current else None)

        # Expose both reason codes and human-readable strings for safety reasons (if present)
        if current and isinstance(current.get("safety"), dict):
            safety_obj = current.get("safety") or {}
            # codes
            attrs["safety_reason_codes"] = list(safety_obj.get("reasons", []) or [])
            # human-friendly strings (compute_forecast supplies 'reason_strings')
            attrs["safety_reason_strings"] = list(safety_obj.get("reason_strings", []) or [])

        # Extract formatted-weather (preferred) and raw current snapshot (fallback)
        formatted = None
        try:
            if current and isinstance(current.get("forecast_raw"), dict):
                formatted = current.get("forecast_raw", {}).get("formatted_weather") or {}
        except Exception:
            formatted = {}

        raw_current = None
        try:
            if isinstance(raw, dict):
                raw_current = raw.get("current")
        except Exception:
            raw_current = None

        # Helper to read a key from formatted then raw_current (return None if missing)
        def _pick(*keys):
            for k in keys:
                try:
                    if formatted and k in formatted:
                        return formatted.get(k)
                except Exception:
                    pass
                try:
                    if raw_current and k in raw_current:
                        return raw_current.get(k)
                except Exception:
                    pass
            return None

        # Provide convenient top-level current_* attributes:
        attrs["current_temperature"] = _pick("temperature", "temp", "temperature_c")
        # wind speed: formatted uses "wind" (m/s canonical) while raw_current likely uses "wind_speed"
        attrs["current_wind_speed"] = _pick("wind", "wind_speed", "wind_m_s")
        # wind gust (canonical)
        attrs["current_wind_gust"] = _pick("wind_gust", "wind_max_m_s", "windgusts_10m")
        # wind unit (from current snapshot)
        attrs["current_wind_unit"] = _pick("wind_unit", "wind_unit")
        # pressure: formatted uses 'pressure_hpa', raw_current may use 'pressure'
        attrs["current_pressure_hpa"] = _pick("pressure_hpa", "pressure", "pressure_hpa")
        # cloud cover
        attrs["current_cloud_cover"] = _pick("cloud_cover", "cloud", "cloudcover")
        # precipitation probability
        attrs["current_precipitation_probability"] = _pick("precipitation_probability", "precipitation_probability", "pop")
        # visibility (if present)
        attrs["current_visibility_km"] = _pick("visibility", "visibility_km")
        # Wave / swell metrics
        attrs["current_wave_height_m"] = _pick("wave_height_m", "wave_height", "wave_height")
        attrs["current_wave_period_s"] = _pick("wave_period_s", "wave_period", "wave_period_s")
        attrs["current_swell_height_m"] = _pick("swell_height_m", "swell_wave_height", "swell_height")
        attrs["current_swell_period_s"] = _pick("swell_period_s", "swell_period_s")

        # Keep legacy-named compact fields where applicable for backwards compat
        if attrs.get("current_wind_gust") is None:
            attrs["current_wind_gust"] = _pick("wind_max_m_s", "wind_max_m_s")

        # --- Prefer tide dict produced by TideProxy for moon phase & next high/low tide ---
        next_high_obj = None
        next_low_obj = None
        moon_numeric = None
        try:
            tide_obj = None
            # coordinator may expose canonical 'tide' at top-level
            if isinstance(self.coordinator.data, dict) and "tide" in self.coordinator.data:
                tide_obj = self.coordinator.data.get("tide")
            # fallback to raw payload tide block
            if (not tide_obj or not isinstance(tide_obj, dict)) and isinstance(raw, dict):
                tide_obj = raw.get("tide") or tide_obj

            if isinstance(tide_obj, dict):
                # canonical naming from TideProxy: 'timestamps', 'tide_height_m', 'tide_phase', 'next_high', 'next_low'
                tide_ts = tide_obj.get("timestamps") or tide_obj.get("time")
                tide_heights = tide_obj.get("tide_height_m") or tide_obj.get("height_m") or tide_obj.get("height")
                # next_high/next_low are ISO strings produced by TideProxy (may be empty strings)
                raw_next_high = tide_obj.get("next_high")
                raw_next_low = tide_obj.get("next_low")

                # Normalize next_high
                if raw_next_high:
                    # ensure it's parseable ISO; keep as ISO Z string for consumers
                    parsed = _parse_dt(raw_next_high)
                    iso = _iso_z(parsed) if parsed else str(raw_next_high)
                    height = None
                    if parsed and tide_ts and tide_heights and isinstance(tide_ts, (list, tuple)) and isinstance(tide_heights, (list, tuple)):
                        height = _find_height_for_timestamp(iso, list(tide_ts), list(tide_heights))
                    # round height if numeric
                    if height is not None:
                        try:
                            height = round(float(height), 3)
                        except Exception:
                            pass
                    next_high_obj = {"timestamp": iso, "height_m": height}

                # Normalize next_low
                if raw_next_low:
                    parsed = _parse_dt(raw_next_low)
                    iso = _iso_z(parsed) if parsed else str(raw_next_low)
                    height = None
                    if parsed and tide_ts and tide_heights and isinstance(tide_ts, (list, tuple)) and isinstance(tide_heights, (list, tuple)):
                        height = _find_height_for_timestamp(iso, list(tide_ts), list(tide_heights))
                    if height is not None:
                        try:
                            height = round(float(height), 3)
                        except Exception:
                            pass
                    next_low_obj = {"timestamp": iso, "height_m": height}

                # Moon phase numeric
                if "tide_phase" in tide_obj:
                    try:
                        moon_numeric = float(tide_obj.get("tide_phase")) if tide_obj.get("tide_phase") is not None else None
                    except Exception:
                        moon_numeric = None
        except Exception:
            next_high_obj = None
            next_low_obj = None
            moon_numeric = None

        # If not found via tide dict, fall back to coordinator.data["moon_phase"] or raw['moon_phase']
        if moon_numeric is None:
            try:
                mp = self.coordinator.data.get("moon_phase") if isinstance(self.coordinator.data, dict) else None
                if isinstance(mp, (list, tuple)):
                    moon_numeric = float(mp[0]) if len(mp) > 0 else None
                else:
                    moon_numeric = float(mp) if mp is not None else None
            except Exception:
                moon_numeric = None

        if moon_numeric is None:
            try:
                rp_mp = raw.get("moon_phase") if isinstance(raw, dict) else None
                if isinstance(rp_mp, (list, tuple)):
                    moon_numeric = float(rp_mp[0]) if len(rp_mp) > 0 else None
                else:
                    moon_numeric = float(rp_mp) if rp_mp is not None else None
            except Exception:
                moon_numeric = None

        attrs["moon_phase"] = moon_numeric
        attrs["moon_phase_name"] = _moon_phase_name(moon_numeric)

        # Expose normalized next_high/next_low objects (may be None)
        attrs["next_high_tide"] = next_high_obj
        attrs["next_low_tide"] = next_low_obj

        # Attribution to appear both as explicit key and HA-standard ATTR_ATTRIBUTION for older integrations
        attrs["attribution"] = ATTRIBUTION
        attrs[ATTR_ATTRIBUTION] = ATTRIBUTION
        return attrs


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities):
    """
    Entry setup for the single OFA sensor.

    Expects coordinator instance to be stored in hass.data[DOMAIN][entry.entry_id].

    Strict behavior:
      - Requires entry.data[CONF_NAME] to be set by the user during config.
      - No default or fallback names are used.
      - The configured name will be deterministically prefixed with "ocean_fishing_assistant_" if not already.
      - If the required key is missing, setup fails loudly by raising RuntimeError.
    """
    coordinator = hass.data[DOMAIN].get(entry.entry_id) if hass.data.get(DOMAIN) else None
    if coordinator is None:
        raise RuntimeError("Coordinator not found in hass.data for this config entry (strict)")

    try:
        # Read name from entry.data (strict — the config flow must have stored this)
        sensor_name = entry.data.get(CONF_NAME)
    except Exception:
        raise RuntimeError("Failed to read config entry data; 'name' (CONF_NAME) is required (strict)")

    if not sensor_name:
        raise RuntimeError(
            "Missing required name in config entry data; the Ocean Fishing Assistant integration requires the user to set a name during configuration (strict)."
        )

    # Ensure prefixing per policy (deterministic transformation).
    prefix = "ocean_fishing_assistant"
    final_name = sensor_name if sensor_name.startswith(prefix) else f"{prefix}_{sensor_name}"

    async_add_entities([OFASensor(coordinator, name=final_name)], update_before_add=True)