# custom_components/ocean_fishing_assistant/sensor.py
"""
This sensor assumes strict canonical payloads produced by DataFormatter and
Skyfield. It removes historical aliases and many fallback heuristics so that
missing or malformed upstream data surfaces as clear errors.
"""
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import logging
import copy
import os
import json
import re

from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.const import ATTR_ATTRIBUTION
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import DOMAIN, CONF_NAME
from . import unit_helpers
from .moon_utils import coerce_phase as moon_coerce_phase, fraction_to_name

_LOGGER = logging.getLogger(__name__)

ATTRIBUTION = "Data provided by Open-Meteo"


def _parse_dt_isoz(isoz: str) -> Optional[datetime]:
    if isoz is None:
        return None
    if isinstance(isoz, datetime):
        return isoz.astimezone(timezone.utc) if isoz.tzinfo else isoz.replace(tzinfo=timezone.utc)
    s = str(isoz)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    parsed = datetime.fromisoformat(s)
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso_z(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _round_opt(v: Optional[float], ndigits: int = 3) -> Optional[float]:
    if v is None:
        return None
    return round(float(v), ndigits)


def _m_s_to_display(w_m_s: Optional[float], entry_units: str) -> Tuple[Optional[float], Optional[str]]:
    if w_m_s is None:
        return None, None
    val, unit = unit_helpers.wind_m_s_to_display(w_m_s, entry_units)
    if val is None:
        return None, None
    return _round_opt(val, 2), unit


def _moon_phase_name(phase: Optional[float]) -> Optional[str]:
    if phase is None:
        return None
    p = moon_coerce_phase(phase)
    p = float(p) % 1.0
    return fraction_to_name(p)


def _format_with_unit(val: Optional[float], unit: Optional[str], ndigits: int = 3) -> Optional[Any]:
    if val is None:
        return None
    rv = _round_opt(val, ndigits)
    if unit:
        return f"{rv} {unit}"
    return rv


def _augment_components_with_values_simple(
    components: Optional[Dict[str, Any]],
    score_calc_raw: Optional[Dict[str, Any]],
    entry_units: str,
    expose_raw: bool = False,
) -> Optional[Dict[str, Any]]:
    if components is None:
        return None
    if not isinstance(components, dict):
        return components

    comps_copy = copy.deepcopy(components)

    def _merge_sibling_value_unit(cc: Dict[str, Any]) -> None:
        for unit_key in list(cc.keys()):
            if not unit_key.endswith("_unit"):
                continue
            unit = cc.get(unit_key)
            base = unit_key[:-5]
            candidate = None

            for k in cc.keys():
                if k == unit_key or k.endswith("_unit"):
                    continue
                if k.startswith(base):
                    candidate = k
                    break

            if candidate is None:
                for k in cc.keys():
                    if k == unit_key or k.endswith("_unit"):
                        continue
                    if base in k:
                        candidate = k
                        break

            if candidate is None:
                for k in cc.keys():
                    if k == unit_key or k.endswith("_unit"):
                        continue
                    v = cc.get(k)
                    if isinstance(v, (int, float)):
                        candidate = k
                        break
                    if isinstance(v, str):
                        s = v.strip()
                        if s.replace(".", "", 1).lstrip("-").isdigit():
                            candidate = k
                            break

            if candidate is None:
                cc.pop(unit_key, None)
                continue

            val = cc.get(candidate)
            if isinstance(val, str) and isinstance(unit, str) and unit in val:
                cc.pop(unit_key, None)
                continue

            try:
                num = float(val)
            except Exception:
                cc.pop(unit_key, None)
                continue

            nd = 3
            if "wind" in candidate or "gust" in candidate:
                nd = 2
            elif "temp" in candidate or "temperature" in candidate:
                nd = 1
            elif "height" in candidate or "wave" in candidate or "tide" in candidate or "delta" in candidate or "pressure" in candidate:
                nd = 3

            cc[candidate] = _format_with_unit(num, unit, ndigits=nd)
            cc.pop(unit_key, None)

    out: Dict[str, Any] = {}
    for cname, cobj in comps_copy.items():
        if not isinstance(cobj, dict):
            out[cname] = cobj
            continue
        cc = copy.deepcopy(cobj)
        cc.pop("score_10", None)
        raw = score_calc_raw or {}
        if cname == "wind":
            if raw.get("wind") is not None:
                val, unit = unit_helpers.wind_m_s_to_display(raw.get("wind"), entry_units)
                cc["wind_speed"] = _format_with_unit(val, unit, ndigits=2) if val is not None else None
        elif cname == "waves":
            if raw.get("wave") is not None:
                val, unit = unit_helpers.length_m_to_display(raw.get("wave"), entry_units)
                cc["wave_height"] = _format_with_unit(val, unit, ndigits=3) if val is not None else None
        elif cname == "pressure":
            if raw.get("pressure_delta") is not None:
                val, unit = unit_helpers.pressure_hpa_to_display(raw.get("pressure_delta"), entry_units)
                cc["pressure_delta"] = _format_with_unit(val, unit, ndigits=3) if val is not None else None
        elif cname == "moon":
            if raw.get("moon_phase") is not None:
                mp = raw.get("moon_phase")
                mpn = moon_coerce_phase(mp)
                if expose_raw:
                    cc["moon_phase"] = _round_opt(mpn, 6)
                mname = _moon_phase_name(mpn)
                if mname:
                    cc["moon_phase_name"] = mname
        elif cname == "temperature":
            if raw.get("temperature") is not None:
                temp_c = raw.get("temperature")
                val, unit = unit_helpers.temp_c_to_display(temp_c, entry_units)
                cc["temperature"] = _format_with_unit(val, unit, ndigits=1) if val is not None else None

        _merge_sibling_value_unit(cc)

        out[cname] = cc
    return out


def _collect_safety_values(score_calc_raw: Optional[Dict[str, Any]], entry_units: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not score_calc_raw or not isinstance(score_calc_raw, dict):
        return out
    raw = score_calc_raw
    if raw.get("wind_gust") is not None:
        val = raw.get("wind_gust")
        w_val, w_unit = unit_helpers.wind_m_s_to_display(val, entry_units)
        if w_val is not None:
            out["wind_gust"] = _format_with_unit(w_val, w_unit, ndigits=2)
    if raw.get("visibility_km") is not None:
        vis_km = raw.get("visibility_km")
        v_val, v_unit = unit_helpers.visibility_km_to_display(vis_km, entry_units)
        if v_val is not None:
            out["visibility"] = _format_with_unit(v_val, v_unit, ndigits=2)
    if raw.get("swell_period_s") is not None:
        out["swell_period_s"] = _round_opt(raw.get("swell_period_s"), 1)
    if raw.get("precipitation_probability") is not None:
        out["precipitation_probability"] = int(round(float(raw.get("precipitation_probability"))))
    return out


async def _async_options_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    _LOGGER.debug("ocean_fishing_assistant: options updated for %s -> %s", entry.entry_id, entry.options)
    coordinator = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if coordinator is None:
        _LOGGER.debug("ocean_fishing_assistant: coordinator not found for entry %s", entry.entry_id)
        return
    await coordinator.async_request_refresh()


class OFASensor(CoordinatorEntity):
    def __init__(
        self,
        coordinator,
        name: str,
        entry: Optional[ConfigEntry] = None,
    ):
        if not name:
            raise RuntimeError("Sensor name must be provided")

        super().__init__(coordinator)

        if entry is None or not entry.data.get(CONF_NAME):
            raise RuntimeError("ConfigEntry with CONF_NAME required for strict sensor initialization")

        friendly_name = entry.data[CONF_NAME]
        self._attr_name = friendly_name
        self._entry: Optional[ConfigEntry] = entry

        entry_id = entry.entry_id
        slug = re.sub(r"\W+", "_", entry_id).strip("_").lower()
        self._attr_unique_id = f"ocean_fishing_assistant_{slug}_score"

        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry_id)},
            "name": friendly_name,
            "manufacturer": "Ocean Fishing Assistant",
            "model": "Fishing Score Sensor",
        }

    def _is_raw_enabled(self) -> bool:
        if self._entry is None:
            return False
        opts = getattr(self._entry, "options", None) or {}
        return bool(opts.get("expose_raw", False))

    @property
    def available(self) -> bool:
        return bool(self.coordinator.last_update_success and self.coordinator.data)

    def _get_current_forecast(self) -> Optional[Dict[str, Any]]:
        data = self.coordinator.data
        if not data or "per_timestamp_forecasts" not in data:
            raise RuntimeError("Coordinator data missing 'per_timestamp_forecasts' (strict)")

        forecasts = data["per_timestamp_forecasts"]
        if not isinstance(forecasts, list) or not forecasts:
            raise RuntimeError("per_timestamp_forecasts is empty or not a list (strict)")

        now_utc = dt_util.utcnow()
        floored = now_utc.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

        for entry in forecasts:
            ts = entry.get("timestamp")
            if ts is None:
                continue
            dt = _parse_dt_isoz(ts)
            if dt == floored:
                return entry

        for entry in forecasts:
            dt = _parse_dt_isoz(entry.get("timestamp"))
            if dt and dt >= floored:
                return entry

        raise RuntimeError("No matching forecast found for current time (strict)")

    @property
    def state(self) -> Optional[int]:
        data = self.coordinator.data
        if not data:
            raise RuntimeError("Coordinator data missing (strict)")

        forecast = self._get_current_forecast()
        sc = forecast.get("score_100")
        if sc is None:
            raise RuntimeError("Current forecast missing score_100 (strict)")
        return int(sc)

    @property
    def icon(self) -> str:
        forecast = None
        try:
            forecast = self._get_current_forecast()
        except Exception:
            data = getattr(self.coordinator, "data", {}) or {}
            per_ts = data.get("per_timestamp_forecasts") or []
            if isinstance(per_ts, list) and per_ts:
                forecast = per_ts[0]

        if not forecast or not isinstance(forecast, dict):
            return "mdi:fish"

        sc = forecast.get("score_100")
        if sc is None:
            return "mdi:fish"
        sc_int = int(sc)
        if sc_int >= 70:
            return "mdi:fish"
        if sc_int >= 50:
            return "mdi:fish-off"
        return "mdi:alert-circle-outline"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        data = self.coordinator.data
        if not data:
            raise RuntimeError("Coordinator data missing when building attributes (strict)")

        attrs: Dict[str, Any] = {}
        entry_units = getattr(self.coordinator, "units", "metric") or "metric"

        tide_phases = None
        tide_container = data.get("tide")
        if isinstance(tide_container, dict):
            tide_phases = tide_container.get("tide_phase")

        current = self._get_current_forecast()

        score_calc = (current.get("forecast_raw") or {}).get("score_calc") or {}
        score_calc_raw = score_calc.get("raw") if isinstance(score_calc, dict) else None

        current_copy = dict(current)
        if not self._is_raw_enabled():
            current_copy.pop("forecast_raw", None)
            current_copy.pop("index", None)
            current_copy.pop("score_10", None)
            current_copy.pop("profile_used", None)

        comps = current_copy.get("components")
        current_copy["components"] = _augment_components_with_values_simple(comps, score_calc_raw, entry_units, self._is_raw_enabled())

        safety_vals = _collect_safety_values(score_calc_raw, entry_units)
        if safety_vals:
            current_copy["safety_values"] = safety_vals

        if isinstance(current_copy.get("breaches"), list):
            sanitized = []
            for b in current_copy.get("breaches") or []:
                ex = dict(b)
                u = ex.pop("unit", None)
                v = ex.get("value")
                if u and v is not None:
                    try:
                        num = float(v)
                        if u == "m/s":
                            disp, disp_unit = unit_helpers.wind_m_s_to_display(num, entry_units)
                            if disp is not None:
                                ex["value"] = f"{round(disp,2)} {disp_unit}"
                            else:
                                ex["value"] = f"{round(num,2)} {u}"
                        else:
                            nd = 0 if "hour" in u else (1 if "°C" in u or "hPa" in u else 3)
                            ex["value"] = f"{round(num, nd)} {u}"
                    except Exception:
                        ex["value"] = f"{v} {u}"
                sanitized.append(ex)
            current_copy["breaches"] = sanitized

        if tide_phases is None:
            raise RuntimeError("Coordinator tide_phase missing (strict)")
        idx = current.get("index")
        if not isinstance(idx, int):
            raise RuntimeError("Current forecast missing index (strict)")
        if idx < 0 or idx >= len(tide_phases):
            raise RuntimeError("Current forecast index out of range for tide_phase (strict)")
        current_copy.pop("tide_phase_name", None)
        current_copy["tide_phase"] = tide_phases[idx]

        ccomps = current_copy["components"]
        tcomp = ccomps["tide"]
        tcomp.pop("tide_phase_name", None)
        tcomp["tide_phase"] = tide_phases[idx]

        attrs["current_forecast"] = current_copy

        period_forecasts = data.get("period_forecasts", {}) or {}
        timestamps = data.get("timestamps", []) or []
        per_ts_forecasts = data.get("per_timestamp_forecasts", []) or []

        now_local = dt_util.now()
        today_local = now_local.date()

        remainder_of_today: Dict[str, Any] = {}
        next_5_days: Dict[str, Any] = {}

        for date_key, pmap in period_forecasts.items():
            for pname, pdata in (pmap or {}).items():
                indices = pdata.get("indices") or []
                if not isinstance(indices, list) or not indices:
                    continue
                include_today = False
                include_next_days = False
                touched_dates = set()
                for idx in indices:
                    if not isinstance(idx, int):
                        continue
                    if idx < 0 or idx >= len(timestamps):
                        continue
                    ts = timestamps[idx]
                    dt_utc = _parse_dt_isoz(ts)
                    if dt_utc is None:
                        continue
                    dt_local = dt_util.as_local(dt_utc)
                    if dt_local.date() == today_local and dt_local >= now_local:
                        include_today = True
                    if today_local < dt_local.date() <= (today_local + timedelta(days=5)):
                        include_next_days = True
                        touched_dates.add(dt_local.date().isoformat())

                sanitized = dict(pdata)
                sanitized.pop("indices", None)
                if not self._is_raw_enabled():
                    sanitized.pop("score_10", None)
                sanitized.pop("profile_used", None)

                raw_agg: Dict[str, float] = {}
                counts: Dict[str, int] = {}
                for idx in indices:
                    if not isinstance(idx, int):
                        continue
                    if idx < 0 or idx >= len(per_ts_forecasts):
                        continue
                    fe = per_ts_forecasts[idx] or {}
                    score_calc = (fe.get("forecast_raw") or {}).get("score_calc") or {}
                    raw = score_calc.get("raw") if isinstance(score_calc, dict) else None
                    if not raw:
                        continue
                    for k, keyname in (("wind", "wind"), ("tide", "tide"), ("wave", "wave"),
                                      ("pressure_delta", "pressure_delta"), ("moon_phase", "moon_phase"),
                                      ("temperature", "temperature"), ("wind_gust", "wind_gust"),
                                      ("visibility_km", "visibility_km"), ("swell_period_s", "swell_period_s"),
                                      ("precipitation_probability", "precipitation_probability")):
                        if raw.get(keyname) is None:
                            continue
                        v = float(raw.get(keyname))
                        if k not in raw_agg:
                            raw_agg[k] = v
                            counts[k] = 1
                        else:
                            raw_agg[k] += v
                            counts[k] += 1
                for k in list(raw_agg.keys()):
                    raw_agg[k] = raw_agg[k] / (counts.get(k, 1) or 1)

                if tide_phases is None:
                    raise RuntimeError("Coordinator tide_phase missing (strict)")
                first_idx = indices[0]
                if not isinstance(first_idx, int) or first_idx < 0 or first_idx >= len(tide_phases):
                    raise RuntimeError("Period indices out of range for tide_phase (strict)")
                sanitized.pop("tide_phase_name", None)
                sanitized["tide_phase"] = tide_phases[first_idx]

                pcomps = sanitized["components"]
                tcomp = pcomps["tide"]
                tcomp.pop("tide_phase_name", None)
                tcomp["tide_phase"] = tide_phases[first_idx]

                comps = sanitized.get("components")
                sanitized["components"] = _augment_components_with_values_simple(comps, raw_agg or None, entry_units, self._is_raw_enabled())

                period_safety = _collect_safety_values(raw_agg or None, entry_units) or {}
                if period_safety:
                    sanitized["safety_values"] = period_safety

                if include_today:
                    remainder_of_today[pname] = sanitized
                if include_next_days:
                    for d in touched_dates:
                        next_5_days.setdefault(d, {})[pname] = sanitized

        attrs["remainder_of_today_periods"] = remainder_of_today
        attrs["next_5_day_periods"] = next_5_days

        if self._is_raw_enabled():
            raw_per_ts = data.get("per_timestamp_forecasts")
            if isinstance(raw_per_ts, list):
                sanitized_list = []
                for entry in raw_per_ts:
                    e_copy = copy.deepcopy(entry)
                    e_copy.pop("profile_used", None)
                    e_copy.pop("safety", None)

                    if tide_phases is None:
                        raise RuntimeError("Coordinator tide_phase missing (strict)")
                    idx = e_copy.get("index")
                    if isinstance(idx, int):
                        if idx < 0 or idx >= len(tide_phases):
                            raise RuntimeError("Per-timestamp entry index out of range for tide_phase (strict)")
                        e_copy.pop("tide_phase_name", None)
                        e_copy["tide_phase"] = tide_phases[idx]

                        ccomps = e_copy["components"]
                        tcomp = ccomps["tide"]
                        tcomp.pop("tide_phase_name", None)
                        tcomp["tide_phase"] = tide_phases[idx]

                    sanitized_list.append(e_copy)
                attrs["per_timestamp_forecasts"] = sanitized_list
            else:
                attrs["per_timestamp_forecasts"] = raw_per_ts

            attrs["period_forecasts"] = period_forecasts
            attrs["raw_payload"] = data.get("raw_payload") or data
        attrs["raw_output_enabled"] = bool(self._is_raw_enabled())

        pu = current.get("profile_used")
        if pu is None:
            attrs["profile_used"] = None
        else:
            pu_copy = dict(pu) if isinstance(pu, dict) else pu
            selected = getattr(self.coordinator, "species", None)

            def _resolve_species_profile(sel: Any) -> Optional[Dict[str, Any]]:
                if sel is None:
                    return None
                if isinstance(sel, dict):
                    return sel
                if isinstance(sel, str):
                    base_dir = os.path.dirname(__file__)
                    spath = os.path.join(base_dir, "species_profiles.json")
                    with open(spath, "r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    species_map = payload.get("species", {}) if isinstance(payload, dict) else {}
                    prof = species_map.get(sel)
                    return prof
                return None

            resolved = _resolve_species_profile(selected)

            if isinstance(pu_copy, dict) and isinstance(resolved, dict) and resolved.get("scientific_name"):
                pu_copy["scientific_name"] = resolved.get("scientific_name", "")
                pu_copy["info"] = resolved.get("info", "")
            attrs["profile_used"] = pu_copy

        attrs["units"] = entry_units

        tide_obj = data.get("tide")
        if not tide_obj or not isinstance(tide_obj, dict):
            _LOGGER.error("Coordinator tide block missing or invalid (strict): %r", tide_obj)
            attrs["next_high_tide"] = None
            attrs["next_low_tide"] = None
        else:
            def _build_tide_attr(tobj: dict) -> Optional[dict]:
                if not isinstance(tobj, dict):
                    _LOGGER.error("Tide entry malformed (expected dict): %r", tobj)
                    return None
                ts = tobj.get("timestamp")
                if ts is None:
                    _LOGGER.error("Tide entry missing required key 'timestamp': %r", tobj)
                    return None
                dt = _parse_dt_isoz(ts)
                if dt is None:
                    _LOGGER.error("Tide entry timestamp not ISOZ: %r", ts)
                    return None
                return {"timestamp": dt.isoformat().replace("+00:00", "Z")}

            nh_attr = _build_tide_attr(tide_obj.get("next_high"))
            nl_attr = _build_tide_attr(tide_obj.get("next_low"))

            attrs["next_high_tide"] = nh_attr
            attrs["next_low_tide"] = nl_attr

        moon_numeric = None
        if score_calc_raw and isinstance(score_calc_raw, dict) and score_calc_raw.get("moon_phase") is not None:
            moon_numeric = score_calc_raw.get("moon_phase")
        elif "moon_phase" in data:
            mp = data.get("moon_phase")
            if isinstance(mp, (list, tuple)):
                idx = current.get("index")
                if isinstance(idx, int) and idx < len(mp):
                    moon_numeric = mp[idx]
                else:
                    moon_numeric = mp[0] if mp else None
            else:
                moon_numeric = mp

        if self._is_raw_enabled():
            attrs["moon_phase"] = _round_opt(moon_numeric, 6) if moon_numeric is not None else None

        attrs["moon_phase_name"] = _moon_phase_name(moon_numeric) if moon_numeric is not None else None

        formatted = (current.get("forecast_raw") or {}).get("formatted_weather") or {}
        temp_c = formatted.get("temperature")
        if temp_c is not None:
            t_val, t_unit = unit_helpers.temp_c_to_display(temp_c, entry_units)
            attrs["current_temperature"] = _format_with_unit(t_val, t_unit, ndigits=1) if t_val is not None else None
        else:
            attrs["current_temperature"] = None

        wind_m_s = formatted.get("wind")
        w_val, w_unit = _m_s_to_display(wind_m_s, entry_units)
        attrs["current_wind_speed"] = _format_with_unit(w_val, w_unit, ndigits=2) if w_val is not None else None

        gust_m_s = formatted.get("wind_gust")
        g_val, g_unit = _m_s_to_display(gust_m_s, entry_units)
        attrs["current_wind_gust"] = _format_with_unit(g_val, g_unit, ndigits=2) if g_val is not None else None

        p_hpa = formatted.get("pressure_hpa")
        if p_hpa is not None:
            p_val, p_unit = unit_helpers.pressure_hpa_to_display(p_hpa, entry_units)
            attrs["current_pressure"] = _format_with_unit(p_val, p_unit, ndigits=3) if p_val is not None else None
        else:
            attrs["current_pressure"] = None

        wave_m = formatted.get("wave_height_m")
        if wave_m is not None:
            wave_val, wave_unit = unit_helpers.length_m_to_display(wave_m, entry_units)
            attrs["current_wave_height"] = _format_with_unit(wave_val, wave_unit, ndigits=3) if wave_val is not None else None
        else:
            attrs["current_wave_height"] = None

        attrs["current_swell_period_s"] = _round_opt(formatted.get("swell_period_s"), 1) if formatted.get("swell_period_s") is not None else None

        attrs["attribution"] = ATTRIBUTION
        attrs[ATTR_ATTRIBUTION] = ATTRIBUTION
        return attrs


# Platform setup / unload
async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    _LOGGER.debug("sensor.async_setup_entry called for entry %s", entry.entry_id)
    coordinator = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if coordinator is None:
        _LOGGER.error("Coordinator not found for entry %s; aborting platform setup", entry.entry_id)
        return

    name = entry.data.get(CONF_NAME)
    if not name:
        _LOGGER.error(
            "Config entry %s missing required '%s' in entry.data (strict); aborting sensor setup",
            entry.entry_id,
            CONF_NAME,
        )
        raise ValueError("Entry data missing 'name' (strict)")

    sensor = OFASensor(coordinator, name, entry=entry)
    async_add_entities([sensor], True)
    entry.add_update_listener(_async_options_updated)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    _LOGGER.debug("sensor.async_unload_entry called for entry %s", entry.entry_id)
    return True