# Strict sensor entity - surfaces errors loudly so callers see misconfiguration
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import math

from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.const import ATTR_ATTRIBUTION
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN, CONF_NAME
from . import unit_helpers

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


def _iso_z(dt: Optional[datetime]) -> str:
    """Return ISO string ending with Z for an aware UTC datetime, or empty string for None."""
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
        val = float(p) % 1.0
    except Exception:
        return None
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


def _round_opt(v: Any, ndigits: int = 3) -> Any:
    """Round numeric v to ndigits or return None if v is None."""
    try:
        if v is None:
            return None
        return round(float(v), ndigits)
    except Exception:
        return v


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
    # Try exact match first
    for i, t in enumerate(tide_ts):
        dt = _parse_dt(t)
        if dt is None:
            continue
        if dt == target_dt:
            try:
                return float(tide_heights[i]) if tide_heights[i] is not None else None
            except Exception:
                return None
    # Fallback to nearest by time
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
        """
        if not name:
            raise RuntimeError("Sensor name must be provided (strict)")

        super().__init__(coordinator)

        prefix = "ocean_fishing_assistant"
        if not name.startswith(prefix):
            name = f"{prefix}_{name}"

        self._attr_name = name

        try:
            safe_name = name.replace(" ", "_")
            self._attr_unique_id = f"{prefix}_{getattr(coordinator, 'entry_id', 'noentry')}_{safe_name}"
        except Exception:
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

    def _extract_wind_ms_and_hint(self, formatted: Optional[Dict[str, Any]], raw_current: Optional[Dict[str, Any]]) -> (Optional[float], Optional[str]):
        """
        Return a tuple (wind_m_s_or_None, src_unit_hint_or_None).
        - If formatted provides 'wind', assume canonical m/s (DataFormatter canonicalization).
        - Else look at raw_current and attempt to interpret wind/wind_speed and wind_unit.
        """
        # 1) formatted (canonical m/s)
        try:
            if formatted and isinstance(formatted.get("wind"), (int, float)):
                return float(formatted.get("wind")), "m/s"
            if formatted and formatted.get("wind") is not None:
                # if it's a string numeric
                try:
                    return float(formatted.get("wind")), "m/s"
                except Exception:
                    pass
        except Exception:
            pass

        # 2) raw_current: try common keys and unit hints
        try:
            if raw_current and isinstance(raw_current, dict):
                for k in ("wind", "wind_speed", "wind_speed_10m", "wind_m_s", "wind_kmh", "wind_mph"):
                    if k in raw_current and raw_current.get(k) is not None:
                        val = raw_current.get(k)
                        # unit hint detection
                        unit_hint = None
                        try:
                            # prefer explicit unit label key
                            if "wind_unit" in raw_current and raw_current.get("wind_unit"):
                                unit_hint = str(raw_current.get("wind_unit")).strip().lower()
                            elif k.endswith("kmh") or k.endswith("km/h") or k.endswith("kmh"):
                                unit_hint = "km/h"
                            elif k.endswith("mph"):
                                unit_hint = "mph"
                            elif k.endswith("m_s") or k.endswith("m/s"):
                                unit_hint = "m/s"
                        except Exception:
                            unit_hint = None
                        try:
                            return float(val), unit_hint
                        except Exception:
                            return None, unit_hint
        except Exception:
            pass
        return None, None

    def _to_display_wind(self, raw_val: Optional[float], src_unit_hint: Optional[str], entry_units: str) -> (Optional[float], Optional[str]):
        """
        Convert a wind value (raw_val with optional src_unit_hint) to display units.
        entry_units: "metric" => km/h, "imperial" => mph, else leave as m/s.
        Returns (value_converted_or_None, display_unit_label_or_None).
        """
        if raw_val is None:
            return None, None

        # First, coerce to m/s (canonical) from source hint if needed.
        ms_val = None
        try:
            if src_unit_hint:
                uh = str(src_unit_hint).strip().lower()
                if uh in ("m/s", "m_s", "mps"):
                    ms_val = float(raw_val)
                elif uh in ("km/h", "kmh", "kph", "km per h"):
                    ms_val = unit_helpers.kmh_to_m_s(raw_val)
                elif uh in ("mph", "mi/h", "miles/h"):
                    ms_val = unit_helpers.mph_to_m_s(raw_val)
                else:
                    # unknown hint: try numeric interpretation and assume m/s
                    ms_val = float(raw_val)
            else:
                # No hint: treat incoming as m/s (DataFormatter typically gives canonical m/s)
                ms_val = float(raw_val)
        except Exception:
            return None, None

        # Convert to requested display units
        if entry_units == "metric":
            # display km/h
            out_val = unit_helpers.m_s_to_kmh(ms_val)
            return _round_opt(out_val, 2), "km/h"
        if entry_units == "imperial":
            out_val = unit_helpers.m_s_to_mph(ms_val)
            return _round_opt(out_val, 2), "mph"

        # fallback: show m/s
        return _round_opt(ms_val, 2), "m/s"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """
        Strict attributes with display conversions applied for final outputs (wind units converted
        per user selection). Raw payload is preserved under 'raw_payload'.
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

        # Profile and safety detail pointers (raw canonical values)
        attrs["profile_used"] = (current.get("profile_used") if current else None)

        entry_units = getattr(self.coordinator, "units", "metric") or "metric"
        attrs["units"] = entry_units

        # Report canonical safety_limits on coordinator but present a display-friendly copy
        raw_limits = getattr(self.coordinator, "safety_limits", {}) or {}
        # Build a display-friendly safety_limits dict with wind/gust converted to display units
        display_limits: Dict[str, Any] = {}
        for k, v in raw_limits.items():
            if v is None:
                display_limits[k] = None
                continue
            try:
                if k in ("max_wind_m_s", "max_gust_m_s"):
                    # convert m/s -> km/h or mph
                    if entry_units == "metric":
                        display_limits[k] = _round_opt(unit_helpers.m_s_to_kmh(v), 2)
                    elif entry_units == "imperial":
                        display_limits[k] = _round_opt(unit_helpers.m_s_to_mph(v), 2)
                    else:
                        display_limits[k] = _round_opt(v, 2)
                else:
                    # Keep canonical metric for other values, but round
                    display_limits[k] = _round_opt(v, 3)
            except Exception:
                display_limits[k] = v
        attrs["safety_limits"] = display_limits

        attrs["safety"] = (current.get("safety") if current else None)

        # Expose both reason codes and human-readable strings for safety reasons (if present)
        if current and isinstance(current.get("safety"), dict):
            safety_obj = current.get("safety") or {}
            attrs["safety_reason_codes"] = list(safety_obj.get("reasons", []) or [])
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
        def _pick_raw(*keys):
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

        # Provide top-level attributes with conversions/rounding applied for display
        # Temperature (keep °C; rounding to 1 dp)
        temp = _pick_raw("temperature", "temp", "temperature_c")
        try:
            attrs["current_temperature"] = _round_opt(float(temp), 1) if temp is not None else None
        except Exception:
            attrs["current_temperature"] = temp

        # Wind speed & gust: convert to display units (km/h or mph) using unit_helpers
        raw_wind_val, raw_wind_hint = self._extract_wind_ms_and_hint(formatted, raw_current)
        wind_disp_val, wind_unit_label = self._to_display_wind(raw_wind_val, raw_wind_hint, entry_units)
        attrs["current_wind_speed"] = wind_disp_val
        attrs["current_wind_unit"] = wind_unit_label

        # Gust
        # Try formatted wind_gust first (canonical m/s), else raw_current keys
        try:
            raw_gust_val = None
            raw_gust_hint = None
            if formatted and formatted.get("wind_gust") is not None:
                raw_gust_val = formatted.get("wind_gust")
                raw_gust_hint = "m/s"
            elif raw_current:
                for k in ("wind_gust", "wind_max_m_s", "windgusts_10m", "wind_max"):
                    if k in raw_current and raw_current.get(k) is not None:
                        raw_gust_val = raw_current.get(k)
                        # unit hint if available
                        raw_gust_hint = raw_current.get("wind_unit") or None
                        break
            gust_disp_val, gust_unit_label = self._to_display_wind(raw_gust_val, raw_gust_hint, entry_units)
            attrs["current_wind_gust"] = gust_disp_val
        except Exception:
            attrs["current_wind_gust"] = None

        # Pressure (hPa) round 1 decimal
        try:
            p = _pick_raw("pressure_hpa", "pressure")
            attrs["current_pressure_hpa"] = _round_opt(float(p), 1) if p is not None else None
        except Exception:
            attrs["current_pressure_hpa"] = p

        # Cloud cover (int)
        try:
            cc = _pick_raw("cloud_cover", "cloud", "cloudcover")
            attrs["current_cloud_cover"] = int(round(float(cc))) if cc is not None else None
        except Exception:
            attrs["current_cloud_cover"] = None

        # Precip probability (int)
        try:
            pop = _pick_raw("precipitation_probability", "pop")
            attrs["current_precipitation_probability"] = int(round(float(pop))) if pop is not None else None
        except Exception:
            attrs["current_precipitation_probability"] = None

        # Visibility (km) round to 2 dp
        try:
            vis = _pick_raw("visibility", "visibility_km")
            attrs["current_visibility_km"] = _round_opt(vis, 2) if vis is not None else None
        except Exception:
            attrs["current_visibility_km"] = None

        # Wave / swell metrics (keep meters/seconds canonical; round)
        try:
            wh = _pick_raw("wave_height_m", "wave_height", "wave_height")
            attrs["current_wave_height_m"] = _round_opt(wh, 3) if wh is not None else None
        except Exception:
            attrs["current_wave_height_m"] = None
        try:
            wp = _pick_raw("wave_period_s", "wave_period", "wave_period_s")
            attrs["current_wave_period_s"] = _round_opt(wp, 2) if wp is not None else None
        except Exception:
            attrs["current_wave_period_s"] = None
        try:
            sh = _pick_raw("swell_height_m", "swell_wave_height", "swell_height")
            attrs["current_swell_height_m"] = _round_opt(sh, 3) if sh is not None else None
        except Exception:
            attrs["current_swell_height_m"] = None
        try:
            sp = _pick_raw("swell_period_s", "swell_period_s")
            attrs["current_swell_period_s"] = _round_opt(sp, 1) if sp is not None else None
        except Exception:
            attrs["current_swell_period_s"] = None

        # --- Tide & Moon: prefer canonical tide dict produced by TideProxy ---
        next_high_obj = None
        next_low_obj = None
        moon_numeric = None
        try:
            tide_obj = None
            if isinstance(self.coordinator.data, dict) and "tide" in self.coordinator.data:
                tide_obj = self.coordinator.data.get("tide")
            if (not tide_obj or not isinstance(tide_obj, dict)) and isinstance(raw, dict):
                tide_obj = raw.get("tide") or tide_obj

            if isinstance(tide_obj, dict):
                tide_ts = tide_obj.get("timestamps") or tide_obj.get("time")
                tide_heights = tide_obj.get("tide_height_m") or tide_obj.get("height_m") or tide_obj.get("height")
                raw_next_high = tide_obj.get("next_high")
                raw_next_low = tide_obj.get("next_low")

                if raw_next_high:
                    parsed = _parse_dt(raw_next_high)
                    iso = _iso_z(parsed) if parsed else str(raw_next_high)
                    height = None
                    if parsed and tide_ts and tide_heights and isinstance(tide_ts, (list, tuple)) and isinstance(tide_heights, (list, tuple)):
                        height = _find_height_for_timestamp(iso, list(tide_ts), list(tide_heights))
                    if height is not None:
                        try:
                            height = round(float(height), 3)
                        except Exception:
                            pass
                    next_high_obj = {"timestamp": iso, "height_m": height}

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

                if "tide_phase" in tide_obj:
                    try:
                        moon_numeric = float(tide_obj.get("tide_phase")) if tide_obj.get("tide_phase") is not None else None
                    except Exception:
                        moon_numeric = None
        except Exception:
            next_high_obj = None
            next_low_obj = None
            moon_numeric = None

        # Fallback moon_phase from other locations if not provided by tide dict
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

        attrs["moon_phase"] = _round_opt(moon_numeric, 6) if moon_numeric is not None else None
        attrs["moon_phase_name"] = _moon_phase_name(moon_numeric)

        attrs["next_high_tide"] = next_high_obj
        attrs["next_low_tide"] = next_low_obj

        # Attribution to appear both as explicit key and HA-standard ATTR_ATTRIBUTION for older integrations
        attrs["attribution"] = ATTRIBUTION
        attrs[ATTR_ATTRIBUTION] = ATTRIBUTION
        return attrs


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities):
    """
    Entry setup for the single OFA sensor.
    """
    coordinator = hass.data[DOMAIN].get(entry.entry_id) if hass.data.get(DOMAIN) else None
    if coordinator is None:
        raise RuntimeError("Coordinator not found in hass.data for this config entry (strict)")

    try:
        sensor_name = entry.data.get(CONF_NAME)
    except Exception:
        raise RuntimeError("Failed to read config entry data; 'name' (CONF_NAME) is required (strict)")

    if not sensor_name:
        raise RuntimeError(
            "Missing required name in config entry data; the Ocean Fishing Assistant integration requires the user to set a name during configuration (strict)."
        )

    prefix = "ocean_fishing_assistant"
    final_name = sensor_name if sensor_name.startswith(prefix) else f"{prefix}_{sensor_name}"

    async_add_entities([OFASensor(coordinator, name=final_name)], update_before_add=True)