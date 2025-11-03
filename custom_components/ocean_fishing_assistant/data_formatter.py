"""Data formatter: validate incoming raw payloads, canonicalize units, attach tide data,
and precompute per-timestamp forecasts using the scoring engine.

This module provides a single DataFormatter class with:
- validate(raw, species_profile=None, units='metric', safety_limits=None) -> canonical dict with 'forecasts' array
- format_tide_data(raw_tide) -> normalized tide dict

The formatter accepts either the normalized output from the internal OpenMeteo client or
more raw structures and will attempt to canonicalize keys.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

from .ocean_scoring import compute_score, MissingDataError, compute_forecast
from . import unit_helpers

_LOGGER = logging.getLogger(__name__)


class DataFormatter:
    """Validate and canonicalize payloads and precompute forecasts."""

    @staticmethod
    def format_tide_data(raw_tide: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize tide proxy output to canonical keys.

        Accepts the structure produced by TideProxy and returns a dict containing:
        - timestamps (list of ISO strings)
        - tide_height_m (list of floats/None)
        - tide_phase (float|None)
        - tide_strength (float)
        - next_high/next_low (ISO strings)
        - confidence/source
        - forecast (optional per-day forecast dict)
        """
        if not isinstance(raw_tide, dict):
            return raw_tide or {}

        out: Dict[str, Any] = {}
        out["timestamps"] = raw_tide.get("timestamps") or raw_tide.get("time") or []

        # Accept either meter or feet keys
        if "tide_height_m" in raw_tide:
            out["tide_height_m"] = raw_tide.get("tide_height_m")
        elif "tide_height_ft" in raw_tide:
            out["tide_height_m"] = [unit_helpers.ft_to_m(x) for x in (raw_tide.get("tide_height_ft") or [])]
        else:
            out["tide_height_m"] = [None] * len(out["timestamps"])

        out["tide_phase"] = raw_tide.get("tide_phase")
        # ensure strength is a float 0..1
        try:
            out["tide_strength"] = float(raw_tide.get("tide_strength", 0.5))
        except Exception:
            out["tide_strength"] = 0.5

        out["next_high"] = raw_tide.get("next_high")
        out["next_low"] = raw_tide.get("next_low")
        out["confidence"] = raw_tide.get("confidence")
        out["source"] = raw_tide.get("source")
        if "forecast" in raw_tide:
            out["forecast"] = raw_tide.get("forecast")

        return out

    @staticmethod
    def _convert_imperial_to_metric(payload: Dict[str, Any]) -> None:
        """In-place convert common imperial keys to canonical metric keys expected by scoring.

        Common conversions handled:
        - temperature_f -> temperature_c
        - wind_mph / wind_kmh -> wind_m_s
        - wave_height_ft -> wave_height_m
        - tide_height_ft -> tide_height_m
        - pressure_inhg -> pressure_hpa
        - visibility_mi -> visibility_km (if present)
        - swell_period_s / swell_period -> swell_period_s (try to canonicalize)
        """
        # Temperature
        if "temperature_f" in payload and "temperature_c" not in payload:
            payload["temperature_c"] = [unit_helpers.f_to_c(x) for x in payload.get("temperature_f") or []]

        # Wind
        if "wind_mph" in payload and "wind_m_s" not in payload:
            payload["wind_m_s"] = [unit_helpers.mph_to_m_s(x) for x in payload.get("wind_mph") or []]
        if "wind_kmh" in payload and "wind_m_s" not in payload:
            payload["wind_m_s"] = [unit_helpers.kmh_to_m_s(x) for x in payload.get("wind_kmh") or []]

        # Wave
        if "wave_height_ft" in payload and "wave_height_m" not in payload:
            payload["wave_height_m"] = [unit_helpers.ft_to_m(x) for x in payload.get("wave_height_ft") or []]

        # Tide
        if "tide_height_ft" in payload and "tide_height_m" not in payload:
            payload["tide_height_m"] = [unit_helpers.ft_to_m(x) for x in payload.get("tide_height_ft") or []]

        # Pressure
        if "pressure_inhg" in payload and "pressure_hpa" not in payload:
            payload["pressure_hpa"] = [unit_helpers.inhg_to_hpa(x) for x in payload.get("pressure_inhg") or []]

        # Visibility (miles -> km)
        if "visibility_mi" in payload and "visibility_km" not in payload:
            vis = payload.get("visibility_mi")
            if isinstance(vis, (list, tuple)):
                payload["visibility_km"] = [unit_helpers.miles_to_km(x) for x in vis]
            else:
                payload["visibility_km"] = unit_helpers.miles_to_km(vis)

        # Swell period: try to canonicalize common keys
        if "swell_period_s" not in payload:
            if "swell_period" in payload:
                payload["swell_period_s"] = payload.get("swell_period")
            elif "swell_period_sec" in payload:
                payload["swell_period_s"] = payload.get("swell_period_sec")

    def validate(
        self,
        raw: Dict[str, Any],
        species_profile: Optional[str] = None,
        units: str = "metric",
        safety_limits: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Take raw payloads (normalized by fetcher) and produce canonical output.

        Returns a dict with keys:
        - timestamps
        - forecasts (list of per-index forecast dicts)
        - raw_payload (canonicalized input used for scoring)
        - units ("metric")
        - species_profile_requested
        - safety_limits (the canonical limits used)
        """
        if not raw:
            return {}

        # Make a defensive copy
        payload = copy.deepcopy(raw)

        # Some callers (e.g. TideProxy) may return a nested structured tide dict; merge if present
        if "tide" in payload and isinstance(payload.get("tide"), dict):
            t = payload.get("tide") or {}
            if "tide_height_m" in t and "tide_height_m" not in payload:
                payload["tide_height_m"] = t.get("tide_height_m")

        # If the fetcher returned a strict Open-Meteo payload (hourly arrays under 'hourly'),
        # translate those arrays into the canonical keys expected by the scoring/forecast
        # engine. This keeps the translator behavior explicit and avoids heuristics.
        if "hourly" in payload and isinstance(payload.get("hourly"), dict):
            hourly = payload.get("hourly") or {}
            # timestamps
            timestamps = list(hourly.get("time") or [])
            payload["timestamps"] = timestamps

            # Map Open-Meteo keys to canonical arrays (do not mutate original hourly dict)
            def _as_list_from_hourly(key: str) -> List[Any]:
                v = hourly.get(key)
                return list(v) if isinstance(v, (list, tuple)) else []

            # Temperature (C)
            if "temperature_2m" in hourly and "temperature_c" not in payload:
                payload["temperature_c"] = _as_list_from_hourly("temperature_2m")

            # Wind (Open-Meteo provides m/s for wind_speed_10m)
            if "wind_speed_10m" in hourly and "wind_m_s" not in payload:
                payload["wind_m_s"] = _as_list_from_hourly("wind_speed_10m")
            if "windgusts_10m" in hourly and "wind_gust_m_s" not in payload:
                payload["wind_gust_m_s"] = _as_list_from_hourly("windgusts_10m")

            # Pressure (hPa)
            if "pressure_msl" in hourly and "pressure_hpa" not in payload:
                payload["pressure_hpa"] = _as_list_from_hourly("pressure_msl")

            # Cloud cover (percent)
            if "cloudcover" in hourly and "cloud_cover" not in payload:
                payload["cloud_cover"] = _as_list_from_hourly("cloudcover")

            # Precipitation probability (percent)
            if "precipitation_probability" in hourly and "precipitation_probability" not in payload:
                payload["precipitation_probability"] = _as_list_from_hourly("precipitation_probability")

            # Marine/wave fields (map common OM marine keys to our canonical names)
            if "wave_height" in hourly and "wave_height_m" not in payload:
                payload["wave_height_m"] = _as_list_from_hourly("wave_height")
            if "wave_period" in hourly and "wave_period_s" not in payload:
                payload["wave_period_s"] = _as_list_from_hourly("wave_period")
            if "swell_height" in hourly and "swell_height_m" not in payload:
                payload["swell_height_m"] = _as_list_from_hourly("swell_height")
            if "swell_period" in hourly and "swell_period_s" not in payload:
                payload["swell_period_s"] = _as_list_from_hourly("swell_period")

        else:
            # Convert known imperial keys if requested or present
            # Always attempt to canonicalize any imperial keys we find (defensive)
            self._convert_imperial_to_metric(payload)

            # Ensure timestamps exist (some payloads use 'time' or nested structures)
            timestamps = payload.get("timestamps") or payload.get("time") or []
            payload["timestamps"] = timestamps

        # If TideProxy attached a normalized tide structure under 'tide' key, normalize it
        if "tide" in payload and isinstance(payload.get("tide"), dict):
            tide_norm = self.format_tide_data(payload.get("tide"))
            payload["tide_height_m"] = tide_norm.get("tide_height_m")
            payload["tide_phase"] = tide_norm.get("tide_phase")
            payload["tide_strength"] = tide_norm.get("tide_strength")

        # Prefer compute_forecast to build rich per-timestamp forecasts (includes astro/tide/marine snippets)
        try:
            forecasts = compute_forecast(payload, species_profile=species_profile, safety_limits=safety_limits)
        except Exception:
            _LOGGER.debug("compute_forecast failed, falling back to per-index compute_score", exc_info=True)
            forecasts = []
            for idx, ts in enumerate(timestamps):
                try:
                    res = compute_score(
                        payload,
                        species_profile=species_profile,
                        use_index=idx,
                        safety_limits=safety_limits,
                    )
                    # Attach index and timestamp to the forecast entry (backward-compatible shape)
                    entry: Dict[str, Any] = {
                        "timestamp": ts,
                        "index": idx,
                        "score_10": res.get("score_10"),
                        "score_100": res.get("score_100"),
                        "components": res.get("components"),
                        "raw": res.get("raw"),
                        "profile_used": res.get("profile_used"),
                        "safety": res.get("safety"),
                    }
                except MissingDataError:
                    entry = {
                        "timestamp": ts,
                        "index": idx,
                        "score_10": None,
                        "score_100": None,
                        "components": None,
                        "raw": None,
                        "profile_used": None,
                        "safety": {"unsafe": False, "caution": False, "reasons": []},
                    }
                except Exception:
                    _LOGGER.debug("Failed to compute score for index %s", idx, exc_info=True)
                    entry = {
                        "timestamp": ts,
                        "index": idx,
                        "score_10": None,
                        "score_100": None,
                        "components": None,
                        "raw": None,
                        "profile_used": None,
                        "safety": {"unsafe": False, "caution": False, "reasons": []},
                    }
                forecasts.append(entry)

        out: Dict[str, Any] = {
            "timestamps": timestamps,
            "forecasts": forecasts,
            "raw_payload": payload,
            "units": "metric",
            "safety_limits": safety_limits or {},
        }
        # If caller passed species_profile name, add to output for visibility
        out["species_profile_requested"] = species_profile
        return out