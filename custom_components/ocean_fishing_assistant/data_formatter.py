# custom_components/ocean_fishing_assistant/data_formatter.py
"""
Strict DataFormatter (no fallbacks, fail loudly)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from homeassistant.util import dt as dt_util

from . import unit_helpers
from . import ocean_scoring
from .const import CONF_FACTOR_WEIGHTS

_LOGGER = logging.getLogger(__name__)


class DataFormatter:
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
        "wave_direction": "wave_direction",
        "wave_period": "wave_period_s",
        "swell_wave_height": "swell_height_m",
        "swell_wave_period": "swell_period_s",
    }

    def __init__(self, config_entry_data: Optional[Dict[str, Any]] = None) -> None:
        self._config_entry_data = config_entry_data or {}

    def _extract_factor_weights_from_self(self) -> Optional[Dict[str, float]]:
        if isinstance(self._config_entry_data, dict) and self._config_entry_data.get(CONF_FACTOR_WEIGHTS):
            return self._config_entry_data.get(CONF_FACTOR_WEIGHTS)
        return None

    def validate(
        self,
        raw_payload: Dict[str, Any],
        species_profile=None,
        units: str = "metric",
        safety_limits: Optional[dict] = None,
        precomputed_period_indices: Optional[Dict[str, Dict[str, Any]]] = None,
        factor_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(raw_payload, dict):
            raise ValueError("raw_payload must be a dict (strict)")

        if "hourly" not in raw_payload or not isinstance(raw_payload["hourly"], dict):
            raise ValueError("raw_payload must include an 'hourly' dict (strict)")

        hourly = raw_payload["hourly"]
        if "time" not in hourly or not isinstance(hourly["time"], (list, tuple)):
            raise ValueError("'hourly' must include 'time' array (strict)")

        raw_timestamps = list(hourly["time"])
        if not raw_timestamps:
            raise ValueError("'time' array is empty (strict)")

        timestamps: List[str] = []
        for t in raw_timestamps:
            parsed = dt_util.parse_datetime(str(t)) if t is not None else None
            if parsed is None:
                v = float(t)
                if v > 1e12:
                    v = v / 1000.0
                parsed = datetime.fromtimestamp(v, tz=timezone.utc)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            timestamps.append(parsed.isoformat().replace("+00:00", "Z"))

        # hourly_units may be absent / non-dict; do not require it to be present as dict.
        hourly_units_raw = raw_payload.get("hourly_units") or hourly.get("units") or {}
        hourly_units = hourly_units_raw if isinstance(hourly_units_raw, dict) else {}

        canonical: Dict[str, Any] = {}
        canonical["timestamps"] = timestamps

        if isinstance(raw_payload.get("location_tz"), str):
            canonical["location_tz"] = raw_payload.get("location_tz")

        if precomputed_period_indices is not None:
            canonical["period_forecasts"] = precomputed_period_indices

        missing_required = []
        if "temperature_2m" not in hourly:
            missing_required.append("temperature_2m")
        if "wind_speed_10m" not in hourly:
            missing_required.append("wind_speed_10m")
        if missing_required:
            raise ValueError(f"Missing required hourly arrays: {missing_required} (strict)")

        for om_key, canon_key in self.HOURLY_KEY_MAP.items():
            if om_key == "time":
                continue
            if om_key in hourly:
                arr = hourly[om_key]
                if not isinstance(arr, (list, tuple)):
                    raise ValueError(f"Hourly key '{om_key}' must be a list/tuple (strict)")
                # Trust upstream for alignment — do not re-check lengths here.
                if om_key in ("wind_speed_10m", "windgusts_10m"):
                    unit_hint = hourly_units.get(om_key) or hourly_units.get("windspeed") or hourly_units.get("wind_speed_10m") or "m/s"
                    converted: List[Optional[float]] = []
                    for v in arr:
                        if v is None:
                            converted.append(None)
                            continue
                        converted.append(self._convert_wind_array_value(v, unit_hint))
                    canonical[canon_key] = converted
                elif om_key == "visibility":
                    unit_hint = hourly_units.get(om_key) or hourly_units.get("visibility") or "km"
                    converted: List[Optional[float]] = []
                    uh = str(unit_hint).strip().lower()
                    for v in arr:
                        if v is None:
                            converted.append(None)
                            continue
                        fv = float(v)
                        if "km" in uh:
                            converted.append(float(fv))
                        elif "m" in uh:
                            converted.append(float(fv) / 1000.0)
                        else:
                            # assume kilometers for unknown hints
                            converted.append(float(fv))
                    canonical[canon_key] = converted
                else:
                    canonical[canon_key] = list(arr)

        tide_obj = raw_payload.get("tide")
        if isinstance(tide_obj, dict):
            # Remove human-friendly duplicated fields and heights; keep structured tide block.
            tide_copy = {k: v for k, v in tide_obj.items() if k not in ("tide_height_m", "next_high_height_m", "next_low_height_m", "tide_phase_name")}
            canonical["tide"] = tide_copy
            # Promote machine-friendly tide_phase to top-level canonical['tide_phase']
            for k, v in tide_obj.items():
                if k in ("tide_height_m", "next_high_height_m", "next_low_height_m", "tide_phase_name"):
                    continue
                if k == "tide_phase":
                    if isinstance(v, (list, tuple)):
                        canonical["tide_phase"] = list(v)
                    else:
                        raise ValueError("tide_phase must be an array aligned to timestamps (strict)")
                else:
                    # keep other tide keys under canonical['tide']
                    tide_copy[k] = list(v) if isinstance(v, (list, tuple)) else v

        # Moon phase handling: trust TideProxy for moon_phase. Prefer canonical['tide']['moon_phase'] if present,
        # otherwise fall back to raw_payload['moon_phase']. Conversion to float is performed once and will fail-fast
        # if upstream provided non-numeric values.
        mp = None
        if isinstance(canonical.get("tide"), dict) and "moon_phase" in canonical["tide"]:
            mp = canonical["tide"]["moon_phase"]
        elif "moon_phase" in raw_payload:
            mp = raw_payload["moon_phase"]

        if mp is not None:
            if not isinstance(mp, (list, tuple)):
                raise ValueError("moon_phase must be an array aligned to timestamps (strict)")
            canonical["moon_phase"] = [float(x) for x in list(mp)]

        for key in ("astro", "astronomy", "astronomy_forecast", "astro_forecast", "moon_phase"):
            if key in raw_payload and key not in canonical:
                canonical[key] = raw_payload.get(key)

        for key in ("marine", "marine_forecast", "marine_current"):
            if key in raw_payload:
                canonical[key] = raw_payload.get(key)

        marine_fields = ["wave_height", "wave_direction", "wave_period", "swell_wave_height", "swell_wave_period"]
        marine_candidate: Dict[str, Any] = {}
        for mf in marine_fields:
            if mf in hourly:
                arr = hourly[mf]
                if isinstance(arr, (list, tuple)):
                    # Trust upstream alignment; simply include arrays
                    marine_candidate[mf] = list(arr)
        if marine_candidate:
            marine_candidate_with_ts = {"timestamps": timestamps, **marine_candidate}
            if "marine" not in canonical:
                canonical["marine"] = marine_candidate_with_ts

        missing_keys = []
        if "wind_m_s" not in canonical:
            missing_keys.append("wind_m_s")
        if "wave_height_m" not in canonical:
            missing_keys.append("wave_height_m")
        if "temperature_c" not in canonical:
            missing_keys.append("temperature_c")
        if "pressure_hpa" not in canonical:
            missing_keys.append("pressure_hpa")

        if not any(k in canonical for k in ("moon_phase", "tide_phase", "astro", "astronomy", "astronomy_forecast", "astro_forecast")):
            missing_keys.append("moon_phase/astro/tide_phase")

        if missing_keys:
            raise ValueError(f"Insufficient canonical keys to compute strict forecasts (missing {missing_keys})")

        # Determine factor weights to use: explicit param -> self._config_entry_data -> None
        fw = factor_weights if factor_weights is not None else self._extract_factor_weights_from_self()

        per_ts_forecasts: List[Dict[str, Any]] = ocean_scoring.compute_forecast(
            canonical, species_profile=species_profile, safety_limits=safety_limits, units=units, factor_weights=fw
        )

        expose_raw = bool(self._config_entry_data.get("expose_raw", False))
        max_breach_examples = 4 if expose_raw else 1

        for i, entry in enumerate(per_ts_forecasts):
            if entry.get("score_100") is None:
                ts = entry.get("timestamp")
                details = entry.get("forecast_raw") or {}
                raise ValueError(f"Incomplete scoring at index={i} timestamp={ts}: missing required inputs or scoring failed; details={details}")

        def _merge_breach_example(ex: Dict[str, Any], units_local: str) -> Dict[str, Any]:
            u = ex.pop("unit", None)
            v = ex.get("value")
            if u is None or v is None:
                return ex
            if isinstance(v, str) and isinstance(u, str) and u in v:
                ex["value"] = v
                return ex
            try:
                num = float(v)
            except Exception:
                ex["value"] = f"{v} {u}"
                return ex

            if u == "m/s":
                if units_local == "metric":
                    conv = unit_helpers.m_s_to_kmh(num)
                    label = "km/h"
                elif units_local == "imperial":
                    conv = unit_helpers.m_s_to_mph(num)
                    label = "mph"
                else:
                    conv = num
                    label = "m/s"
                ex["value"] = f"{round(conv, 2)} {label}"
                return ex

            nd = 3
            if isinstance(u, str) and ("hour" in u):
                nd = 0
            elif isinstance(u, str) and ("°C" in u or "hPa" in u):
                nd = 1
            ex["value"] = f"{round(num, nd)} {u}"
            return ex

        hourly_like: List[Dict[str, Any]] = []
        for i, ts in enumerate(timestamps):
            row: Dict[str, Any] = {"time": ts}
            for src_key in ("temperature_c", "wind_m_s", "wind_max_m_s", "pressure_hpa", "cloud_cover", "precipitation_probability",
                            "wave_height_m", "wave_period_s", "swell_height_m", "swell_period_s"):
                arr = canonical.get(src_key)
                if isinstance(arr, (list, tuple)) and i < len(arr):
                    row[src_key] = arr[i]
            if i < len(per_ts_forecasts):
                row["_forecast_entry"] = per_ts_forecasts[i]
            hourly_like.append(row)

        period_forecasts: Dict[str, Dict[str, Any]] = {}
        if precomputed_period_indices is not None:
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

                    breach_counts: Dict[str, Dict[str, Any]] = {}
                    breach_examples: List[Dict[str, Any]] = []
                    for e in per_ts_entries:
                        for b in (e.get("breaches") or []):
                            var = b.get("variable")
                            if not var:
                                continue
                            entry_bc = breach_counts.setdefault(var, {"count": 0, "severity": "caution"})
                            entry_bc["count"] += 1
                            if entry_bc["severity"] != "unsafe" and b.get("severity") == "unsafe":
                                entry_bc["severity"] = "unsafe"
                            if len(breach_examples) < max_breach_examples:
                                ex = dict(b)
                                ex["timestamp"] = e.get("timestamp")
                                ex = _merge_breach_example(ex, units)
                                breach_examples.append(ex)

                    breaches_summary = {"by_variable": breach_counts, "examples": breach_examples} if breach_counts else {}

                    # determine tide_phase for the period using canonical top-level tide_phase array (first index representative)
                    tide_phase = None
                    if isinstance(canonical.get("tide_phase"), (list, tuple)):
                        if indices and isinstance(indices[0], int):
                            first_idx = indices[0]
                            tp_arr = canonical.get("tide_phase")
                            if isinstance(tp_arr, (list, tuple)) and first_idx < len(tp_arr):
                                tide_phase = tp_arr[first_idx]

                    summary = {
                        "score_10": round(score_10, 3) if score_10 is not None else None,
                        "score_100": int(round(score_10 * 10)) if score_10 is not None else None,
                        "components": components,
                        "profile_used": profile_used,
                        "safety": safety,
                        "tide_phase": tide_phase,
                        "start": pdata.get("start"),
                        "end": pdata.get("end"),
                        "indices": list(indices),
                        "breaches": breaches_summary,
                    }
                    period_forecasts[date_key][pname] = summary
        else:
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

                    breach_counts: Dict[str, Dict[str, Any]] = {}
                    breach_examples: List[Dict[str, Any]] = []
                    for e in per_ts_entries:
                        for b in (e.get("breaches") or []):
                            var = b.get("variable")
                            if not var:
                                continue
                            entry_bc = breach_counts.setdefault(var, {"count": 0, "severity": "caution"})
                            entry_bc["count"] += 1
                            if entry_bc["severity"] != "unsafe" and b.get("severity") == "unsafe":
                                entry_bc["severity"] = "unsafe"
                            if len(breach_examples) < max_breach_examples:
                                ex = dict(b)
                                ex["timestamp"] = e.get("timestamp")
                                ex = _merge_breach_example(ex, units)
                                breach_examples.append(ex)

                    breaches_summary = {"by_variable": breach_counts, "examples": breach_examples} if breach_counts else {}

                    # determine tide_phase for the period using canonical top-level tide_phase array (first index representative)
                    tide_phase = None
                    if isinstance(canonical.get("tide_phase"), (list, tuple)):
                        if indices and isinstance(indices[0], int):
                            first_idx = indices[0]
                            tp_arr = canonical.get("tide_phase")
                            if isinstance(tp_arr, (list, tuple)) and first_idx < len(tp_arr):
                                tide_phase = tp_arr[first_idx]

                    summary = dict(pdata)
                    summary.update({
                        "score_10": round(score_10, 3) if score_10 is not None else None,
                        "score_100": int(round(score_10 * 10)) if score_10 is not None else None,
                        "components": components,
                        "profile_used": profile_used,
                        "safety": safety,
                        "tide_phase": tide_phase,
                        "breaches": breaches_summary,
                    })
                    period_forecasts[date_key][pname] = summary

        # enforce breach examples limit per expose_raw setting (post-process safety)
        for date_key, pmap in period_forecasts.items():
            for pname, pdata in pmap.items():
                breaches = pdata.get("breaches") or {}
                ex = breaches.get("examples") if isinstance(breaches, dict) else None
                if ex and isinstance(ex, list):
                    breaches["examples"] = ex[:max_breach_examples]
                    pdata["breaches"] = breaches

        final_out = {
            "timestamps": timestamps,
            **canonical,
            "raw_payload": raw_payload,
            "per_timestamp_forecasts": per_ts_forecasts,
            "period_forecasts": period_forecasts,
        }
        return final_out

    def _aggregate_hourly_into_periods(self, hourly_list: List[Dict[str, Any]], days: int, aggregation_periods: Optional[Sequence[Dict[str, Any]]], full_payload: Optional[Dict[str, Any]] = None, units: str = "metric") -> Dict[str, Dict[str, Any]]:
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
            t = dt_util.parse_datetime(str(t_raw)) if t_raw is not None else None
            if t is None:
                tnum = float(t_raw)
                if tnum > 1e12:
                    tnum = tnum / 1000.0
                t = datetime.fromtimestamp(tnum, tz=timezone.utc)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            date_key = t.date().isoformat()
            hour = t.hour

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

                    temp = float(entry.get("temperature_c")) if entry.get("temperature_c") is not None else None
                    wind_m_s = float(entry.get("wind_m_s")) if entry.get("wind_m_s") is not None else None
                    gust_m_s = float(entry.get("wind_max_m_s")) if entry.get("wind_max_m_s") is not None else None
                    cloud = int(entry.get("cloud_cover")) if entry.get("cloud_cover") is not None else None
                    pop = int(entry.get("precipitation_probability")) if entry.get("precipitation_probability") is not None else None
                    pressure = float(entry.get("pressure_hpa")) if entry.get("pressure_hpa") is not None else None

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
                        if agg["gust_max"] is None:
                            agg["gust_max"] = gust_m_s
                        else:
                            agg["gust_max"] = max(agg["gust_max"], gust_m_s)
                    agg["count"] += 1
                    agg["indices"].append(idx)
                    break

        final: Dict[str, Dict[str, Any]] = {}

        out_wind_unit = "km/h" if units == "metric" else "mph" if units == "imperial" else units

        for date_key in sorted(per_date_periods.keys())[:days]:
            final[date_key] = {}
            for pname, agg in per_date_periods[date_key].items():
                cnt = agg.get("count", 0) or 0
                if cnt == 0:
                    continue

                mean_temp = float(agg["temperature_sum"]) / cnt if agg.get("temperature_sum") is not None else None
                mean_wind_m_s = float(agg["wind_speed_sum"]) / cnt if agg.get("wind_speed_sum") is not None else None
                gust_m_s = float(agg["gust_max"]) if agg.get("gust_max") is not None else None
                pressure = float(agg["pressure_sum"]) / cnt if agg.get("pressure_sum") is not None else None
                cloud = int(round(float(agg["cloud_sum"]) / cnt)) if agg.get("cloud_sum") is not None else None
                precip = int(round(float(agg["precip_max"]))) if agg.get("precip_max") is not None else None

                wind_out = None
                gust_out = None
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
        if v is None:
            raise ValueError("None wind value")
        val = float(v)
        u = str(unit_hint).strip().lower() if unit_hint is not None else "m/s"
        if u in ("km/h", "kph", "kmh"):
            out = unit_helpers.kmh_to_m_s(val)
        elif u in ("mph", "mi/h", "miles/h"):
            out = unit_helpers.mph_to_m_s(val)
        else:
            out = val
        if out is None:
            raise ValueError(f"Unable to convert wind value: {v!r} with hint {unit_hint!r}")
        return float(out)