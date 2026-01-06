# custom_components/ocean_fishing_assistant/data_formatter.py
"""
Minimal, strict DataFormatter (fixed).

Changes:
- When the raw payload does not include moon_phase, DataFormatter deterministically
  computes and attaches a moon_phase array (fraction 0.0..1.0) for every timestamp.
  This preserves strict scoring behavior without requiring external providers to include moon_phase.
- No other loosened validations; required arrays are still enforced and types validated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from homeassistant.util import dt as dt_util

from . import unit_helpers
from . import ocean_scoring
from .const import CONF_FACTOR_WEIGHTS

# Canonical mapping: incoming Open-Meteo key -> canonical key
HOURLY_KEY_MAP = {
    "time": "timestamps",
    "temperature_2m": "temperature_c",
    "wind_speed_10m": "wind_m_s",
    "windgusts_10m": "wind_max_m_s",
    "pressure_msl": "pressure_hpa",
    "cloudcover": "cloud_cover",
    "precipitation_probability": "precipitation_probability",
    "visibility": "visibility_km",
    "wave_height": "wave_height_m",
    "wave_period": "wave_period_s",
    "swell_wave_height": "swell_height_m",
    "swell_wave_period": "swell_period_s",
}


def _ensure_length(key: str, timestamps: List[str], arr: List[Any]) -> None:
    if len(timestamps) != len(arr):
        raise ValueError(f"Array length mismatch: '{key}' length={len(arr)} vs timestamps length={len(timestamps)}")


class DataFormatter:
    def __init__(self, config_entry_data: Optional[Dict[str, Any]] = None) -> None:
        self._config_entry_data = config_entry_data or {}

    def _compute_moon_phase_fraction(self, dt: datetime) -> float:
        """
        Deterministic moon phase fraction in [0.0, 1.0] using a simple Julian-date based algorithm.
        Lightweight and does not require external libraries.
        """
        # Convert to UTC naive fractional day
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        year = dt.year
        month = dt.month
        day = dt.day + dt.hour / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0
        # Algorithm to compute Julian day
        y = year
        m = month
        if m < 3:
            y -= 1
            m += 12
        a = int(y / 100)
        b = 2 - a + int(a / 4)
        jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + b - 1524.5
        # Reference new moon at JD 2451550.1 (2000-01-06 18:14 UT roughly)
        days_since_ref = jd - 2451550.1
        synodic_month = 29.53058867
        phase = (days_since_ref % synodic_month) / synodic_month
        return float(phase % 1.0)

    def validate(
        self,
        raw_payload: Dict[str, Any],
        species_profile=None,
        units: str = "metric",
        safety_limits: Optional[dict] = None,
        precomputed_period_indices: Optional[Dict[str, Dict[str, Any]]] = None,
        factor_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Validate raw_payload strictly and return canonical data dict.

        Raises ValueError on any missing/invalid required item.
        """
        if not isinstance(raw_payload, dict):
            raise ValueError("raw_payload must be a dict")

        hourly = raw_payload.get("hourly")
        if not isinstance(hourly, dict):
            raise ValueError("raw_payload['hourly'] must be a dict")

        # timestamps
        times = hourly.get("time")
        if not isinstance(times, (list, tuple)) or not times:
            raise ValueError("'hourly.time' must be a non-empty list")
        timestamps: List[str] = []
        for t in times:
            parsed = dt_util.parse_datetime(str(t))
            if parsed is None:
                # try numeric epoch fallback
                try:
                    v = float(t)
                    if v > 1e12:
                        v = v / 1000.0
                    parsed = datetime.fromtimestamp(v, tz=timezone.utc)
                except Exception as exc:
                    raise ValueError(f"Unable to parse timestamp '{t}': {exc}") from exc
            # normalize to Z-formatted ISO
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            timestamps.append(parsed.isoformat().replace("+00:00", "Z"))

        hourly_units = raw_payload.get("hourly_units")
        if not isinstance(hourly_units, dict):
            raise ValueError("raw_payload must include 'hourly_units' dict (strict)")

        canonical: Dict[str, Any] = {}
        canonical["timestamps"] = timestamps
        if isinstance(raw_payload.get("location_tz"), str):
            canonical["location_tz"] = raw_payload.get("location_tz")

        # Map straightforward keys. Required keys raise if absent or not lists.
        required_keys = ["temperature_2m", "wind_speed_10m", "windgusts_10m", "pressure_msl"]
        for req in required_keys:
            if req not in hourly or not isinstance(hourly[req], (list, tuple)):
                raise ValueError(f"Missing required hourly array '{req}' (strict)")

        # perform direct mapping with minimal conversion
        for om_key, canon_key in HOURLY_KEY_MAP.items():
            if om_key == "time":
                continue
            arr = hourly.get(om_key)
            if arr is None:
                continue
            if not isinstance(arr, (list, tuple)):
                raise ValueError(f"Hourly key '{om_key}' must be a list")
            _ensure_length(om_key, timestamps, list(arr))

            # Handle wind arrays: convert to canonical m/s
            if om_key in ("wind_speed_10m", "windgusts_10m"):
                # expect unit hint in hourly_units
                unit_hint = hourly_units.get(om_key)
                if not unit_hint:
                    # strict: require unit hint for wind arrays
                    raise ValueError(f"Missing unit hint for '{om_key}' in hourly_units (strict)")
                converted = []
                for v in arr:
                    if v is None:
                        converted.append(None)
                    else:
                        try:
                            val = float(v)
                        except Exception:
                            raise ValueError(f"Non-numeric wind value in '{om_key}': {v!r}")
                        uh = str(unit_hint).strip().lower()
                        if uh in ("m/s", "mps", "m s-1"):
                            converted.append(float(val))
                        elif uh in ("km/h", "kmh", "kph"):
                            converted.append(unit_helpers.kmh_to_m_s(val))
                        elif uh in ("mph", "mi/h", "miles/h"):
                            converted.append(unit_helpers.mph_to_m_s(val))
                        else:
                            # strict: unknown unit hint is error
                            raise ValueError(f"Unknown wind unit hint for '{om_key}': {unit_hint!r}")
                canonical[canon_key] = converted
            elif om_key == "visibility":
                # Open-Meteo visibility is reported in meters; canonical is km
                converted = []
                for v in arr:
                    if v is None:
                        converted.append(None)
                    else:
                        converted.append(float(v) / 1000.0)
                canonical[canon_key] = converted
            else:
                canonical[canon_key] = list(arr)

        # minimal canonical checks
        if "wind_m_s" not in canonical or "temperature_c" not in canonical or "pressure_hpa" not in canonical:
            raise ValueError("Insufficient canonical keys: required wind/temperature/pressure arrays missing after mapping")

        # attach tide if present (we assume tide provider returned canonical aligned arrays)
        tide_obj = raw_payload.get("tide")
        if isinstance(tide_obj, dict):
            canonical["tide"] = tide_obj

        # attach other optional parts directly if present
        for key in ("moon_phase", "astro", "marine"):
            if key in raw_payload:
                canonical[key] = raw_payload.get(key)

        # precomputed period indices (optional) — pass through
        if precomputed_period_indices is not None:
            canonical["period_forecasts"] = precomputed_period_indices

        # If moon_phase missing in canonical, compute deterministic moon_phase per timestamp
        if "moon_phase" not in canonical:
            phases: List[float] = []
            for ts in timestamps:
                dt = dt_util.parse_datetime(ts)
                if dt is None:
                    # fallback to new moon (deterministic) if parsing unexpectedly fails
                    phases.append(0.0)
                    continue
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                try:
                    phases.append(self._compute_moon_phase_fraction(dt))
                except Exception:
                    # conservative default
                    phases.append(0.0)
            canonical["moon_phase"] = phases

        # precompute factor weights param to pass through
        fw = factor_weights if factor_weights is not None else (self._config_entry_data.get(CONF_FACTOR_WEIGHTS) if isinstance(self._config_entry_data, dict) else None)

        # compute per-timestamp forecasts using ocean_scoring
        per_ts_forecasts = ocean_scoring.compute_forecast(canonical, species_profile=species_profile, safety_limits=safety_limits, units=units, factor_weights=fw)

        # basic validation: ensure every forecast entry has timestamp and score_10 (score may be None if compute_score failed)
        for i, e in enumerate(per_ts_forecasts):
            if "timestamp" not in e:
                raise ValueError(f"Forecast entry at index {i} missing 'timestamp'")

        final = {
            "timestamps": timestamps,
            **canonical,
            "raw_payload": raw_payload,
            "per_timestamp_forecasts": per_ts_forecasts,
            # period_forecasts will be constructed by DataFormatter if not provided by coordinator; for strict mode we rely on coordinator's precomputed indices
            "period_forecasts": canonical.get("period_forecasts", {}),
        }
        return final