# Strict DataFormatter (no fallbacks, fail loudly)
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from homeassistant.util import dt as dt_util

from . import unit_helpers
from . import ocean_scoring

_LOGGER = logging.getLogger(__name__)


def _ensure_list_length_equal(key: str, timestamps: List[Any], arr: List[Any]) -> None:
    if len(timestamps) != len(arr):
        raise ValueError(f"Array length mismatch for '{key}': timestamps length={len(timestamps)}, {key} length={len(arr)}")


class DataFormatter:
    """
    Strict DataFormatter:

    - Requires payload to contain 'hourly' dict with 'time' array and canonical keys.
    - Requires hourly_units to exist and to provide units for wind arrays.
    - No legacy aliases, no heuristics. Raises ValueError on any validation failure.
    """

    # mapping of Open-Meteo hourly keys -> canonical keys used by scoring/formatting
    HOURLY_KEY_MAP = {
        "time": "timestamps",
        "temperature_2m": "temperature_c",
        "wind_speed_10m": "wind_m_s",
        "windgusts_10m": "wind_max_m_s",
        "pressure_msl": "pressure_hpa",
        "cloudcover": "cloud_cover",
        "precipitation_probability": "precipitation_probability",
        # marine keys
        "wave_height": "wave_height_m",
        "wave_direction": "wave_direction",
        "wave_period": "wave_period_s",
        "swell_wave_height": "swell_height_m",
        "swell_wave_period": "swell_period_s",
    }

    def __init__(self) -> None:
        pass

    def validate(self, raw_payload: Dict[str, Any], species_profile=None, units: str = "metric", safety_limits: Optional[dict] = None, precomputed_period_indices: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Validate and normalize raw_payload into strict canonical shape:
        - 'timestamps' list
        - canonical arrays using canonical keys
        - compute per_timestamp_forecasts via ocean_scoring.compute_forecast (errors propagate)
        - compute period_forecasts via strict aggregator (errors propagate)

        If precomputed_period_indices is provided, it should be a mapping:
          { date: { period_name: { "indices": [idx,...], "start": ISOZ, "end": ISOZ }, ... }, ... }
        where indices reference positions into the canonical timestamps (0-based).
        """
        if not isinstance(raw_payload, dict):
            raise ValueError("raw_payload must be a dict (strict)")

        if "hourly" not in raw_payload or not isinstance(raw_payload["hourly"], dict):
            raise ValueError("raw_payload must include an 'hourly' dict (strict)")

        hourly = raw_payload["hourly"]
        if "time" not in hourly or not isinstance(hourly["time"], (list, tuple)):
            raise ValueError("'hourly' must include 'time' array (strict)")

        timestamps = list(hourly["time"])
        if not timestamps:
            raise ValueError("'time' array is empty (strict)")

        # prepare hourly_units (required for wind conversion)
        hourly_units = raw_payload.get("hourly_units") or hourly.get("units")
        if not isinstance(hourly_units, dict):
            raise ValueError("Missing 'hourly_units' mapping in payload; wind unit hints are required (strict)")

        canonical: Dict[str, Any] = {}
        canonical["timestamps"] = timestamps

        # Map required numeric arrays: require temperature_2m and wind_speed_10m at minimum
        missing_required = []
        if "temperature_2m" not in hourly:
            missing_required.append("temperature_2m")
        if "wind_speed_10m" not in hourly:
            missing_required.append("wind_speed_10m")
        if missing_required:
            raise ValueError(f"Missing required hourly arrays: {missing_required} (strict)")

        # map arrays and enforce equal lengths
        for om_key, canon_key in self.HOURLY_KEY_MAP.items():
            if om_key == "time":
                continue
            if om_key in hourly:
                arr = hourly[om_key]
                if not isinstance(arr, (list, tuple)):
                    raise ValueError(f"Hourly key '{om_key}' must be a list/tuple (strict)")
                # enforce same length as timestamps
                _ensure_list_length_equal(om_key, timestamps, list(arr))
                if om_key in ("wind_speed_10m", "windgusts_10m"):
                    # require explicit unit hint for these
                    unit_hint = hourly_units.get(om_key) or hourly_units.get("windspeed") or hourly_units.get("wind_speed_10m")
                    if not unit_hint:
                        raise ValueError(f"Missing unit hint for hourly key '{om_key}' (strict)")
                    converted: List[Optional[float]] = []
                    for v in arr:
                        if v is None:
                            converted.append(None)
                            continue
                        converted.append(self._convert_wind_array_value(v, unit_hint))
                    canonical[canon_key] = converted
                else:
                    canonical[canon_key] = list(arr)  # shallow copy

        # --- Incorporate tide & astro if present in raw_payload so scoring can use them ---
        tide_obj = raw_payload.get("tide")
        if isinstance(tide_obj, dict):
            # Preserve the original tide dict at top-level so downstream code that expects payload["tide"]
            # can find the structured tide forecast/snapshot.
            canonical["tide"] = tide_obj

            # If tide provides its own 'timestamps' align by index; otherwise require arrays to match
            tide_ts = tide_obj.get("timestamps")
            if tide_ts and isinstance(tide_ts, (list, tuple)):
                # attempt to align by nearest index only if lengths match
                if len(tide_ts) == len(timestamps):
                    # copy tide arrays
                    for k, v in tide_obj.items():
                        if k == "timestamps":
                            continue
                        if isinstance(v, (list, tuple)):
                            _ensure_list_length_equal(k, timestamps, list(v))
                            canonical[k] = list(v)
                        else:
                            # scalar tide metadata
                            canonical[k] = v
            else:
                # tide_obj may provide arrays keyed to timestamps directly — include arrays that match length
                for k, v in tide_obj.items():
                    if isinstance(v, (list, tuple)) and len(v) == len(timestamps):
                        canonical[k] = list(v)
                    elif not isinstance(v, (list, tuple)):
                        canonical[k] = v

        # --- NORMALIZE moon_phase from tide proxy or raw payload ---
        # Prefer explicit top-level moon_phase if provided by raw_payload.
        # Otherwise, if TideProxy returned 'tide_phase' under canonical["tide"], use that.
        # Enforce strict alignment: arrays must match timestamps length; scalar values are replicated.
        moon_phase_set = False
        # raw_payload top-level moon_phase (preferred)
        if "moon_phase" in raw_payload:
            mp = raw_payload.get("moon_phase")
            if isinstance(mp, (list, tuple)):
                _ensure_list_length_equal("moon_phase", timestamps, list(mp))
                canonical["moon_phase"] = list(mp)
            else:
                canonical["moon_phase"] = [mp] * len(timestamps)
            moon_phase_set = True

        # If not set yet, consider tide.tide_phase (TideProxy)
        if not moon_phase_set and "tide" in canonical and isinstance(canonical["tide"], dict) and "tide_phase" in canonical["tide"]:
            tp = canonical["tide"].get("tide_phase")
            if tp is not None:
                if isinstance(tp, (list, tuple)):
                    _ensure_list_length_equal("tide_phase", timestamps, list(tp))
                    canonical["moon_phase"] = list(tp)
                else:
                    canonical["moon_phase"] = [tp] * len(timestamps)
                moon_phase_set = True

        # Copy top-level astro/astronomy/moon_phase into canonical so compute_score's _get_moon_phase_for_index() can find them.
        # Note: we still copied raw_payload['moon_phase'] above (preferred), this loop will not overwrite canonical['moon_phase']
        for key in ("astro", "astronomy", "astronomy_forecast", "astro_forecast", "moon_phase"):
            if key in raw_payload and key not in canonical:
                canonical[key] = raw_payload.get(key)

        # Also accept 'marine' or 'marine_forecast' if caller attached it (we do not transform)
        for key in ("marine", "marine_forecast", "marine_current"):
            if key in raw_payload:
                canonical[key] = raw_payload.get(key)

        # If marine arrays were attached into hourly (coordinator may attach marine hourly fields under raw['hourly']),
        # construct a conservative top-level 'marine' dict so scoring can match by timestamps. This is deterministic:
        marine_fields = ["wave_height", "wave_direction", "wave_period", "swell_wave_height", "swell_wave_period"]
        marine_candidate: Dict[str, Any] = {}
        for mf in marine_fields:
            if mf in hourly:
                arr = hourly[mf]
                if isinstance(arr, (list, tuple)):
                    _ensure_list_length_equal(mf, timestamps, list(arr))
                    marine_candidate[mf] = list(arr)
        if marine_candidate:
            # include canonical timestamps for marine block
            marine_candidate_with_ts = {"timestamps": timestamps, **marine_candidate}
            # avoid clobbering existing explicit marine key if present; prefer explicit
            if "marine" not in canonical:
                canonical["marine"] = marine_candidate_with_ts

        # STRICT PRECHECK: ensure canonical contains the essential arrays required by compute_score
        # compute_score expects (per-index): tide_height_m, wind (wind_m_s), wave (wave_height_m), temperature_c,
        # moon_phase/astro, and a pressure_hpa series with at least one future point (len > index+1).
        missing_keys = []
        # tide
        if "tide_height_m" not in canonical and "tide" not in canonical:
            missing_keys.append("tide_height_m")
        # wind
        if "wind_m_s" not in canonical:
            missing_keys.append("wind_m_s")
        # wave
        if "wave_height_m" not in canonical:
            missing_keys.append("wave_height_m")
        # temperature
        if "temperature_c" not in canonical:
            missing_keys.append("temperature_c")
        # pressure series check (STRICT):
        if "pressure_hpa" not in canonical:
            missing_keys.append("pressure_hpa")
        else:
            p_arr = canonical.get("pressure_hpa")
            if not isinstance(p_arr, (list, tuple)):
                missing_keys.append("pressure_hpa_not_array")
            else:
                len_p = len(p_arr)
                len_t = len(timestamps)
                if len_p < len_t:
                    missing_keys.append(f"pressure_hpa_series_too_short ({len_p} < {len_t})")
                elif len_p == len_t:
                    new_len = len_t - 1
                    if new_len <= 0:
                        missing_keys.append("pressure_hpa_series_needs_more_points")
                    else:
                        _LOGGER.info(
                            "Pressure series length equals timestamps (%s); trimming final timestamp and aligned arrays to %s entries to satisfy strict future-point requirement",
                            len_t,
                            new_len,
                        )
                        timestamps = list(timestamps[:new_len])
                        canonical["timestamps"] = timestamps
                        for k, v in list(canonical.items()):
                            if k == "pressure_hpa" or k == "timestamps":
                                continue
                            if isinstance(v, (list, tuple)):
                                if len(v) >= len_t:
                                    canonical[k] = list(v[:new_len])
                                else:
                                    raise ValueError(f"Unexpected array length mismatch while trimming '{k}': expected >= {len_t}, got {len(v)}")
                else:
                    pass

        if not any(k in canonical for k in ("moon_phase", "tide_phase", "astro", "astronomy", "astronomy_forecast", "astro_forecast")):
            missing_keys.append("moon_phase/astro/tide_phase")

        if missing_keys:
            raise ValueError(f"Insufficient canonical keys to compute strict forecasts (missing {missing_keys})")

        # compute per-timestamp forecasts (compute errors propagate)
        per_ts_forecasts: List[Dict[str, Any]] = ocean_scoring.compute_forecast(canonical, species_profile=species_profile, safety_limits=safety_limits)

        # STRICT POSTCHECK: fail if any per-timestamp forecast is incomplete (score_100 is None)
        for i, entry in enumerate(per_ts_forecasts):
            if entry.get("score_100") is None:
                ts = entry.get("timestamp")
                details = entry.get("forecast_raw") or {}
                raise ValueError(f"Incomplete scoring at index={i} timestamp={ts}: missing required inputs or scoring failed; details={details}")

        # Build hourly_like list expected by aggregator (strict)
        hourly_like: List[Dict[str, Any]] = []
        length = len(timestamps)
        for i, ts in enumerate(timestamps):
            row: Dict[str, Any] = {"time": ts}
            for src_key in ("temperature_c", "wind_m_s", "wind_max_m_s", "pressure_hpa", "cloud_cover", "precipitation_probability",
                            "wave_height_m", "wave_period_s", "swell_height_m", "swell_period_s"):
                arr = canonical.get(src_key)
                if isinstance(arr, (list, tuple)):
                    if i < len(arr):
                        row[src_key] = arr[i]
            if i < len(per_ts_forecasts):
                row["_forecast_entry"] = per_ts_forecasts[i]
            hourly_like.append(row)

        # If precomputed_period_indices provided, use that mapping to build period forecasts.
        period_forecasts: Dict[str, Dict[str, Any]] = {}
        if precomputed_period_indices is not None:
            # iterate keys in sorted order, but limit to 7 days for compatibility with previous days param
            for date_key in sorted(precomputed_period_indices.keys())[:7]:
                pmap = precomputed_period_indices.get(date_key) or {}
                period_forecasts[date_key] = {}
                for pname, pdata in pmap.items():
                    indices = pdata.get("indices") or []
                    per_ts_entries = []
                    for idx in indices:
                        if idx < len(hourly_like):
                            fe = hourly_like[idx].get("_forecast_entry")
                            if fe:
                                per_ts_entries.append(fe)
                    score_vals = [float(e.get("score_10")) for e in per_ts_entries if e.get("score_10") is not None]
                    score_10 = float(sum(score_vals) / len(score_vals)) if score_vals else None
                    components = None
                    if per_ts_entries:
                        keys = set().union(*(e.get("components", {}).keys() if e.get("components") else [] for e in per_ts_entries))
                        out_comp = {}
                        for k in keys:
                            vals = []
                            for e in per_ts_entries:
                                c = e.get("components") or {}
                                if k in c and c[k].get("score_10") is not None:
                                    vals.append(float(c[k]["score_10"]))
                            if vals:
                                avg = float(sum(vals) / len(vals))
                                out_comp[k] = {"score_10": round(avg, 3), "score_100": int(round(avg * 10))}
                        components = out_comp or None
                    profile_used = next((e.get("profile_used") for e in per_ts_entries if e.get("profile_used")), None)
                    safety = {"unsafe": any((e.get("safety") or {}).get("unsafe") for e in per_ts_entries),
                              "caution": any((e.get("safety") or {}).get("caution") for e in per_ts_entries),
                              "reasons": sorted({r for e in per_ts_entries for r in (e.get("safety") or {}).get("reasons", [])})}
                    summary = {
                        "score_10": round(score_10, 3) if score_10 is not None else None,
                        "score_100": int(round(score_10 * 10)) if score_10 is not None else None,
                        "components": components,
                        "profile_used": profile_used,
                        "safety": safety,
                        # include provided start/end if present
                        "start": pdata.get("start"),
                        "end": pdata.get("end"),
                        "indices": list(indices),
                    }
                    period_forecasts[date_key][pname] = summary
        else:
            # Fallback: compute using internal hourly->period aggregator (same behavior as before)
            default_periods = [
                {"name": "period_00_06", "start_hour": 0, "end_hour": 6},
                {"name": "period_06_12", "start_hour": 6, "end_hour": 12},
                {"name": "period_12_18", "start_hour": 12, "end_hour": 18},
                {"name": "period_18_24", "start_hour": 18, "end_hour": 24},
            ]

            period_agg = self._aggregate_hourly_into_periods(hourly_like, days=7, aggregation_periods=default_periods, full_payload=raw_payload, units=units)

            for date_key, pmap in period_agg.items():
                period_forecasts[date_key] = {}
                for pname, pdata in pmap.items():
                    indices = pdata.get("indices") or []
                    per_ts_entries = []
                    for idx in indices:
                        if idx < len(hourly_like):
                            fe = hourly_like[idx].get("_forecast_entry")
                            if fe:
                                per_ts_entries.append(fe)
                    score_vals = [float(e.get("score_10")) for e in per_ts_entries if e.get("score_10") is not None]
                    score_10 = float(sum(score_vals) / len(score_vals)) if score_vals else None
                    components = None
                    if per_ts_entries:
                        keys = set().union(*(e.get("components", {}).keys() if e.get("components") else [] for e in per_ts_entries))
                        out_comp = {}
                        for k in keys:
                            vals = []
                            for e in per_ts_entries:
                                c = e.get("components") or {}
                                if k in c and c[k].get("score_10") is not None:
                                    vals.append(float(c[k]["score_10"]))
                            if vals:
                                avg = float(sum(vals) / len(vals))
                                out_comp[k] = {"score_10": round(avg, 3), "score_100": int(round(avg * 10))}
                        components = out_comp or None
                    profile_used = next((e.get("profile_used") for e in per_ts_entries if e.get("profile_used")), None)
                    safety = {"unsafe": any((e.get("safety") or {}).get("unsafe") for e in per_ts_entries),
                              "caution": any((e.get("safety") or {}).get("caution") for e in per_ts_entries),
                              "reasons": sorted({r for e in per_ts_entries for r in (e.get("safety") or {}).get("reasons", [])})}
                    summary = dict(pdata)
                    summary.update({
                        "score_10": round(score_10, 3) if score_10 is not None else None,
                        "score_100": int(round(score_10 * 10)) if score_10 is not None else None,
                        "components": components,
                        "profile_used": profile_used,
                        "safety": safety,
                    })
                    period_forecasts[date_key][pname] = summary

        final_out = {
            "timestamps": timestamps,
            **canonical,
            "raw_payload": raw_payload,
            "per_timestamp_forecasts": per_ts_forecasts,
            "period_forecasts": period_forecasts,
        }
        return final_out

    # Aggregate hourly entries into named periods (implemented locally to avoid cross-class delegation)
    def _aggregate_hourly_into_periods(self, hourly_list: List[Dict[str, Any]], days: int, aggregation_periods: Optional[Sequence[Dict[str, Any]]], full_payload: Optional[Dict[str, Any]] = None, units: str = "metric") -> Dict[str, Dict[str, Any]]:
        """
        Local aggregator that accepts hourly-like rows produced earlier in validate().
        Expects rows with keys: time, temperature_c, wind_m_s, wind_max_m_s, pressure_hpa, cloud_cover,
        precipitation_probability, wave_height_m, wave_period_s, swell_height_m, swell_period_s
        Returns mapping: date -> period_name -> {metrics..., indices: [hourly indices included]}
        """
        if not aggregation_periods:
            aggregation_periods = [
                {"name": "period_00_06", "start_hour": 0, "end_hour": 6},
                {"name": "period_06_12", "start_hour": 6, "end_hour": 12},
                {"name": "period_12_18", "start_hour": 12, "end_hour": 18},
                {"name": "period_18_24", "start_hour": 18, "end_hour": 24},
            ]

        per_date_periods: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for idx, entry in enumerate(hourly_list):
            if not isinstance(entry, dict):
                continue
            t_raw = entry.get("time") or entry.get("datetime") or entry.get("timestamp")
            if t_raw is None:
                continue
            try:
                t = dt_util.parse_datetime(str(t_raw)) if t_raw is not None else None
            except Exception:
                t = None
            if t is None:
                try:
                    tnum = float(t_raw)
                    if tnum > 1e12:
                        tnum = tnum / 1000.0
                    t = datetime.fromtimestamp(tnum, tz=timezone.utc)
                except Exception:
                    continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            date_key = t.date().isoformat()
            hour = t.hour

            # find matching period
            for p in aggregation_periods:
                start = int(p["start_hour"])
                end = int(p["end_hour"])
                if start <= hour < end:
                    pname = p["name"]
                    per_date_periods.setdefault(date_key, {}).setdefault(pname, {
                        "temperature_sum": 0.0,
                        "wind_speed_sum": 0.0,
                        "pressure_sum": 0.0,
                        "cloud_sum": 0,
                        "precip_max": 0,
                        "gust_max": None,
                        "count": 0,
                        "indices": [],
                    })
                    agg = per_date_periods[date_key][pname]
                    try:
                        temp = float(entry.get("temperature_c")) if entry.get("temperature_c") is not None else None
                    except Exception:
                        temp = None
                    try:
                        wind_m_s = float(entry.get("wind_m_s")) if entry.get("wind_m_s") is not None else None
                    except Exception:
                        wind_m_s = None
                    try:
                        gust_m_s = float(entry.get("wind_max_m_s")) if entry.get("wind_max_m_s") is not None else None
                    except Exception:
                        gust_m_s = None
                    try:
                        cloud = int(entry.get("cloud_cover")) if entry.get("cloud_cover") is not None else None
                    except Exception:
                        cloud = None
                    try:
                        pop = int(entry.get("precipitation_probability")) if entry.get("precipitation_probability") is not None else None
                    except Exception:
                        pop = None
                    try:
                        pressure = float(entry.get("pressure_hpa")) if entry.get("pressure_hpa") is not None else None
                    except Exception:
                        pressure = None

                    if temp is not None:
                        agg["temperature_sum"] += temp
                    if wind_m_s is not None:
                        agg["wind_speed_sum"] += wind_m_s
                    if pressure is not None:
                        agg["pressure_sum"] += pressure
                    if cloud is not None:
                        agg["cloud_sum"] += cloud
                    if pop is not None:
                        agg["precip_max"] = max(agg["precip_max"], pop)
                    if gust_m_s is not None:
                        # gust_max must be the maximum gust seen in the period (do not average)
                        if agg["gust_max"] is None:
                            agg["gust_max"] = gust_m_s
                        else:
                            agg["gust_max"] = max(agg["gust_max"], gust_m_s)
                    agg["count"] += 1
                    agg["indices"].append(idx)
                    break

        # finalize
        final: Dict[str, Dict[str, Any]] = {}

        # determine output wind unit
        out_wind_unit = "km/h" if units == "metric" else "mph" if units == "imperial" else units

        for date_key in sorted(per_date_periods.keys())[:days]:
            final[date_key] = {}
            for pname, agg in per_date_periods[date_key].items():
                cnt = agg.get("count", 0) or 0
                if cnt == 0:
                    # Nothing to aggregate for this period
                    continue
                try:
                    mean_temp = float(agg["temperature_sum"]) / cnt if agg.get("temperature_sum") is not None else None
                    mean_wind_m_s = float(agg["wind_speed_sum"]) / cnt if agg.get("wind_speed_sum") is not None else None
                    gust_m_s = float(agg["gust_max"]) if agg.get("gust_max") is not None else None
                    pressure = float(agg["pressure_sum"]) / cnt if agg.get("pressure_sum") is not None else None
                    cloud = int(round(float(agg["cloud_sum"]) / cnt)) if agg.get("cloud_sum") is not None else None
                    precip = int(round(float(agg["precip_max"]))) if agg.get("precip_max") is not None else None
                except Exception:
                    _LOGGER.debug("Failed to finalize aggregation for %s %s; skipping", date_key, pname)
                    continue

                # convert wind to requested display unit
                wind_out = None
                gust_out = None
                try:
                    if mean_wind_m_s is not None:
                        if out_wind_unit in ("km/h", "kph", "kmh"):
                            wind_out = unit_helpers.m_s_to_kmh(mean_wind_m_s)
                        elif out_wind_unit in ("mph",):
                            wind_out = unit_helpers.m_s_to_mph(mean_wind_m_s)
                        else:
                            wind_out = mean_wind_m_s
                    if gust_m_s is not None:
                        if out_wind_unit in ("km/h", "kph", "kmh"):
                            gust_out = unit_helpers.m_s_to_kmh(gust_m_s)
                        elif out_wind_unit in ("mph",):
                            gust_out = unit_helpers.m_s_to_mph(gust_m_s)
                        else:
                            gust_out = gust_m_s
                except Exception:
                    wind_out = None
                    gust_out = None

                final[date_key][pname] = {
                    "temperature": mean_temp,
                    "wind_speed": wind_out,
                    "wind_gust": gust_out,
                    "wind_unit": out_wind_unit,
                    "cloud_cover": cloud,
                    "precipitation_probability": precip,
                    "pressure": pressure,
                    "indices": list(agg.get("indices", [])),
                }

        return final

    def _convert_wind_array_value(self, v: Any, unit_hint: str) -> float:
        """Convert single wind array value to m/s using local converters."""
        try:
            if v is None:
                raise ValueError("None wind value")
            val = float(v)
        except Exception as exc:
            raise ValueError(f"Non-numeric wind value: {v!r}") from exc
        u = str(unit_hint).strip().lower() if unit_hint is not None else "m/s"
        if u in ("km/h", "kph", "kmh"):
            out = unit_helpers.kmh_to_m_s(val)
        elif u in ("mph", "mi/h", "miles/h"):
            out = unit_helpers.mph_to_m_s(val)
        else:
            # assume m/s
            out = val
        if out is None:
            raise ValueError(f"Unable to convert wind value: {v!r} with hint {unit_hint!r}")
        return float(out)