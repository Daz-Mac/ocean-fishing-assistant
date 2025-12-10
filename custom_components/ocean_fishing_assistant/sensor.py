# custom_components/ocean_fishing_assistant/sensor.py
"""
Simplified OFA sensor (aggressive mode).

This sensor assumes strict canonical payloads produced by DataFormatter and
Skyfield. It removes historical aliases and many fallback heuristics so that
missing or malformed upstream data surfaces as clear errors.

Canonical assumptions (from DataFormatter / ocean_scoring):
 - coordinator.data is a dict with keys:
    - "timestamps" : list[str ISOZ]
    - "per_timestamp_forecasts": list[dict] (entries produced by ocean_scoring.compute_forecast)
    - "period_forecasts": dict (optional; strict shape)
    - "raw_payload": original raw payload (optional)
    - "tide": tide dict (optional, canonical fields if present)
    - "moon_phase": list or scalar (optional)
 - Each per_timestamp_forecasts entry contains:
    - "timestamp" (ISOZ), "index" (int), "score_100" (int or None), "components" (dict or None),
      "forecast_raw": { "formatted_weather": {...}, "score_calc": {...} }
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
from . import tide_proxy
from .moon_utils import coerce_phase as moon_coerce_phase, fraction_to_name

_LOGGER = logging.getLogger(__name__)

ATTRIBUTION = "Data provided by Open-Meteo"


def _parse_dt_isoz(isoz: str) -> Optional[datetime]:
    """Parse ISOZ timestamp (expects ISO with trailing 'Z') into aware UTC datetime."""
    if isoz is None:
        return None
    if isinstance(isoz, datetime):
        return isoz.astimezone(timezone.utc) if isoz.tzinfo else isoz.replace(tzinfo=timezone.utc)
    try:
        # handle trailing Z -> +00:00 for fromisoformat
        s = str(isoz)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        parsed = datetime.fromisoformat(s)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


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
    """Convert canonical m/s wind value to display units using unit_helpers."""
    if w_m_s is None:
        return None, None
    try:
        val, unit = unit_helpers.wind_m_s_to_display(w_m_s, entry_units)
        if val is None:
            return None, None
        return _round_opt(val, 2), unit
    except Exception:
        return None, None


def _moon_phase_name(phase: Optional[float]) -> Optional[str]:
    """
    Map normalized moon phase numeric (0..1) to friendly name.
    Delegates numeric coercion to moon_utils.coerce_phase when available.
    """
    if phase is None:
        return None

    try:
        p = moon_coerce_phase(phase)
    except Exception:
        try:
            p = float(phase) % 1.0
        except Exception:
            return None

    try:
        p = float(p) % 1.0
    except Exception:
        return None

    return fraction_to_name(p)


def _format_with_unit(val: Optional[float], unit: Optional[str], ndigits: int = 3) -> Optional[Any]:
    """
    Helper: return combined display either as string "val unit" when unit is present,
    or numeric rounded value when unit is None. Return None if val is None.
    """
    if val is None:
        return None
    try:
        rv = _round_opt(val, ndigits)
        if unit:
            return f"{rv} {unit}"
        return rv
    except Exception:
        return None


def _augment_components_with_values_simple(
    components: Optional[Dict[str, Any]],
    score_calc_raw: Optional[Dict[str, Any]],
    entry_units: str,
) -> Optional[Dict[str, Any]]:
    """
    Simplified augmentation (display-only output, strict canonical):
      - Remove per-component 'score_10'
      - Inject the numeric display value used by scoring when available as merged display strings:
         wind -> wind_speed (e.g. "11.6 km/h")
         waves -> wave_height (e.g. "1.23 m")
         pressure -> pressure_delta (e.g. "0.5 hPa")
         moon -> moon_phase (numeric) and moon_phase_name
         temperature -> temperature (e.g. "21.0 °C")
    This function strictly reads canonical numeric values from score_calc_raw and does not
    attempt to read legacy fields or pre-merged strings.

    Additionally: if a component contains explicit sibling unit keys (e.g. 'tide_height' and 'tide_unit'),
    merge them into a single display string and remove the separate unit key to ensure consistent merged output.
    """
    if components is None:
        return None
    if not isinstance(components, dict):
        return components

    comps_copy = copy.deepcopy(components)

    def _merge_sibling_value_unit(cc: Dict[str, Any]) -> None:
        """
        Detect and merge any measurement + unit sibling pairs inside a component dict.
        Heuristic rounding:
          - wind-related -> 2 dp
          - temperature -> 1 dp
          - tide / wave / pressure / delta / height -> 3 dp
          - fallback -> 3 dp
        After merging, remove the corresponding '*_unit' key.
        """
        try:
            for unit_key in list(cc.keys()):
                if not unit_key.endswith("_unit"):
                    continue
                unit = cc.get(unit_key)
                # base name is the part before _unit
                base = unit_key[:-5]
                candidate = None

                # Prefer candidate keys that start with base
                for k in cc.keys():
                    if k == unit_key or k.endswith("_unit"):
                        continue
                    if k.startswith(base):
                        candidate = k
                        break

                # If no prefix match, try contains base
                if candidate is None:
                    for k in cc.keys():
                        if k == unit_key or k.endswith("_unit"):
                            continue
                        if base in k:
                            candidate = k
                            break

                # Last-resort: pick first numeric-like key that isn't a unit
                if candidate is None:
                    for k in cc.keys():
                        if k == unit_key or k.endswith("_unit"):
                            continue
                        v = cc.get(k)
                        if isinstance(v, (int, float)):
                            candidate = k
                            break
                        if isinstance(v, str):
                            # numeric-ish string?
                            s = v.strip()
                            if s.replace(".", "", 1).lstrip("-").isdigit():
                                candidate = k
                                break

                if candidate is None:
                    # nothing to merge for this unit_key; remove leftover unit to avoid duplication
                    cc.pop(unit_key, None)
                    continue

                val = cc.get(candidate)
                # If value already looks like a merged string containing the unit, just drop the unit key
                if isinstance(val, str) and isinstance(unit, str) and unit in val:
                    cc.pop(unit_key, None)
                    continue

                # Try to coerce numeric value
                try:
                    num = float(val)
                except Exception:
                    # can't coerce; drop the unit_key to avoid leaving extra keys
                    cc.pop(unit_key, None)
                    continue

                # Decide rounding
                nd = 3
                if "wind" in candidate or "gust" in candidate:
                    nd = 2
                elif "temp" in candidate or "temperature" in candidate:
                    nd = 1
                elif "height" in candidate or "wave" in candidate or "tide" in candidate or "delta" in candidate or "pressure" in candidate:
                    nd = 3

                cc[candidate] = _format_with_unit(num, unit, ndigits=nd)
                cc.pop(unit_key, None)
        except Exception:
            # If anything goes wrong, leave component as-is (best-effort merging only)
            pass

    out: Dict[str, Any] = {}
    for cname, cobj in comps_copy.items():
        if not isinstance(cobj, dict):
            out[cname] = cobj
            continue
        cc = copy.deepcopy(cobj)
        cc.pop("score_10", None)
        try:
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
                    try:
                        mpn = moon_coerce_phase(mp)
                    except Exception:
                        mpn = mp
                    cc["moon_phase"] = _round_opt(mpn, 6)
                    mname = _moon_phase_name(mpn)
                    if mname:
                        cc["moon_phase_name"] = mname
            elif cname == "temperature":
                if raw.get("temperature") is not None:
                    temp_c = raw.get("temperature")
                    val, unit = unit_helpers.temp_c_to_display(temp_c, entry_units)
                    cc["temperature"] = _format_with_unit(val, unit, ndigits=1) if val is not None else None
        except Exception:
            # On any unexpected error we skip augmentation for that component.
            pass

        # After injecting canonical merged strings from raw (when present), also merge any
        # existing sibling value+unit keys into the merged form to ensure consistency.
        _merge_sibling_value_unit(cc)

        out[cname] = cc
    return out


def _collect_safety_values(score_calc_raw: Optional[Dict[str, Any]], entry_units: str) -> Dict[str, Any]:
    """
    Collect minimal safety-related values strictly from canonical score_calc_raw (display-only).
    Only numeric canonical fields are considered; no fallbacks to legacy keys or strings.
    Returned values are merged value+unit strings where applicable.
    """
    out: Dict[str, Any] = {}
    if not score_calc_raw or not isinstance(score_calc_raw, dict):
        return out
    raw = score_calc_raw
    try:
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
            try:
                out["precipitation_probability"] = int(round(float(raw.get("precipitation_probability"))))
            except Exception:
                pass
    except Exception:
        pass
    return out


async def _async_options_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Called when the config entry options are updated via the UI."""
    _LOGGER.debug("ocean_fishing_assistant: options updated for %s -> %s", entry.entry_id, entry.options)
    coordinator = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if coordinator is None:
        _LOGGER.debug("ocean_fishing_assistant: coordinator not found for entry %s", entry.entry_id)
        return
    try:
        await coordinator.async_request_refresh()
    except Exception as exc:
        _LOGGER.exception("ocean_fishing_assistant: error requesting coordinator refresh: %s", exc)


class OFASensor(CoordinatorEntity):
    """
    Simplified CoordinatorEntity for Ocean Fishing Assistant.
    Reads `expose_raw` exclusively from entry.options when available.
    """

    def __init__(
        self,
        coordinator,
        name: str,
        expose_raw: bool = False,
        entry: Optional[ConfigEntry] = None,
    ):
        # Strict: name must be present (async_setup_entry already validates CONF_NAME)
        if not name:
            raise RuntimeError("Sensor name must be provided")

        super().__init__(coordinator)

        # Require a ConfigEntry with CONF_NAME for strict mode
        if entry is None or not entry.data.get(CONF_NAME):
            raise RuntimeError("ConfigEntry with CONF_NAME required for strict sensor initialization")

        friendly_name = entry.data[CONF_NAME]
        self._attr_name = friendly_name
        self._entry: Optional[ConfigEntry] = entry

        # Stable unique id: one sensor per config entry (slugged entry_id)
        entry_id = entry.entry_id
        import re
        slug = re.sub(r"\W+", "_", entry_id).strip("_").lower()
        self._attr_unique_id = f"ocean_fishing_assistant_{slug}_score"

        # Single device per config entry
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry_id)},
            "name": friendly_name,
            "manufacturer": "Ocean Fishing Assistant",
            "model": "Fishing Score Sensor",
        }

    def _is_raw_enabled(self) -> bool:
        """
        Read the ConfigEntry options to determine whether raw output is enabled.
        No constructor fallback is used — if options are not present, raw is treated as False.
        """
        try:
            if self._entry is not None:
                opts = getattr(self._entry, "options", None) or {}
                try:
                    return bool(opts.get("expose_raw", False))
                except Exception:
                    try:
                        return bool(getattr(opts, "expose_raw", False))
                    except Exception:
                        pass
        except Exception:
            pass
        # No fallback: treat as False when entry/options are absent or unreadable
        return False

    @property
    def available(self) -> bool:
        return bool(self.coordinator.last_update_success and self.coordinator.data)

    def _get_current_forecast(self) -> Optional[Dict[str, Any]]:
        """
        Return the per-timestamp forecast for the current UTC hour (floored).
        Simple canonical behavior: use per_timestamp_forecasts list and match 'timestamp'.
        Raises RuntimeError if no suitable forecast can be located (strict).
        """
        data = self.coordinator.data
        if not data or "per_timestamp_forecasts" not in data:
            raise RuntimeError("Coordinator data missing 'per_timestamp_forecasts' (strict)")

        forecasts = data["per_timestamp_forecasts"]
        if not isinstance(forecasts, list) or not forecasts:
            raise RuntimeError("per_timestamp_forecasts is empty or not a list (strict)")

        now_utc = dt_util.utcnow()
        floored = now_utc.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

        # find exact match first
        for entry in forecasts:
            ts = entry.get("timestamp")
            if ts is None:
                continue
            dt = _parse_dt_isoz(ts)
            if dt == floored:
                return entry

        # choose the first future entry (timestamp >= floored)
        for entry in forecasts:
            dt = _parse_dt_isoz(entry.get("timestamp"))
            if dt and dt >= floored:
                return entry

        # No match — strict mode: raise instead of falling back
        raise RuntimeError("No matching forecast found for current time (strict)")

    @property
    def state(self) -> Optional[int]:
        data = self.coordinator.data
        if not data:
            raise RuntimeError("Coordinator data missing (strict)")

        forecast = self._get_current_forecast()
        if not forecast:
            raise RuntimeError("No per_timestamp_forecasts available (strict)")

        sc = forecast.get("score_100")
        if sc is None:
            raise RuntimeError("Current forecast missing score_100 (strict)")
        return int(sc)

    @property
    def icon(self) -> str:
        """
        Dynamic icon selection based on the current score_100.
        - >= 70: good -> mdi:fish
        - >= 50: medium -> mdi:fish-off
        - < 50 or missing: poor -> mdi:alert-circle-outline
        This mirrors the state logic but is best-effort (exceptions return a default icon).
        """
        try:
            forecast = None
            try:
                forecast = self._get_current_forecast()
            except Exception:
                # if strict lookup fails, try a forgiving lookup from coordinator data
                data = getattr(self.coordinator, "data", {}) or {}
                per_ts = data.get("per_timestamp_forecasts") or []
                if isinstance(per_ts, list) and per_ts:
                    forecast = per_ts[0]

            if not forecast or not isinstance(forecast, dict):
                return "mdi:fish"

            sc = forecast.get("score_100")
            if sc is None:
                return "mdi:fish"
            try:
                sc_int = int(sc)
            except Exception:
                return "mdi:fish"

            if sc_int >= 70:
                return "mdi:fish"
            if sc_int >= 50:
                return "mdi:fish-off"
            return "mdi:alert-circle-outline"
        except Exception:
            return "mdi:fish"

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """
        Build a compact attribute set assuming canonical inputs.
        - current_forecast: sanitized view of current per-timestamp forecast (components augmented)
        - remainder_of_today_periods & next_5_day_periods: sanitized period entries built from canonical period_forecasts
        - per_timestamp_forecasts & period_forecasts only when expose_raw True
        - raw_payload only when expose_raw True

        This implementation is strict: it does not attempt to read legacy fields or provide
        fallback/backwards-compatibility behavior.
        """
        data = self.coordinator.data
        if not data:
            raise RuntimeError("Coordinator data missing when building attributes (strict)")

        attrs: Dict[str, Any] = {}
        entry_units = getattr(self.coordinator, "units", "metric") or "metric"

        # current forecast
        current = self._get_current_forecast()
        if current is None:
            raise RuntimeError("Unable to locate current per-timestamp forecast (strict)")

        # Use forecast_raw.score_calc.raw as canonical numeric raw values used for scoring
        score_calc = (current.get("forecast_raw") or {}).get("score_calc") or {}
        score_calc_raw = score_calc.get("raw") if isinstance(score_calc, dict) else None

        # Sanitize current forecast: remove heavy raw blocks unless raw is enabled
        current_copy = dict(current)
        if not self._is_raw_enabled():
            current_copy.pop("forecast_raw", None)
            # hide index and score_10 when raw is disabled
            current_copy.pop("index", None)
            current_copy.pop("score_10", None)
            # hide nested profile_used when raw disabled so profile appears only once at top-level
            current_copy.pop("profile_used", None)

        # augment components: remove per-component score_10 and inject numeric values from score_calc_raw when present
        comps = current_copy.get("components")
        current_copy["components"] = _augment_components_with_values_simple(comps, score_calc_raw, entry_units)

        # attach grouped safety values derived strictly from canonical score_calc_raw.raw
        safety_vals = _collect_safety_values(score_calc_raw, entry_units)
        if safety_vals:
            current_copy["safety_values"] = safety_vals

        # sanitize any breaches present in the current forecast by merging value+unit for display
        try:
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
                                # convert into display based on entry_units
                                try:
                                    disp, disp_unit = unit_helpers.wind_m_s_to_display(num, entry_units)
                                    if disp is not None:
                                        ex["value"] = f"{round(disp,2)} {disp_unit}"
                                    else:
                                        ex["value"] = f"{round(num,2)} {u}"
                                except Exception:
                                    ex["value"] = f"{round(num,2)} {u}"
                            else:
                                nd = 0 if "hour" in u else (1 if "°C" in u or "hPa" in u else 3)
                                ex["value"] = f"{round(num, nd)} {u}"
                        except Exception:
                            ex["value"] = f"{v} {u}"
                    sanitized.append(ex)
                current_copy["breaches"] = sanitized
        except Exception:
            # best-effort only
            pass

        attrs["current_forecast"] = current_copy

        # --- Grouped period views (remainder_of_today_periods, next_5_day_periods) ---
        period_forecasts = data.get("period_forecasts", {}) or {}
        timestamps = data.get("timestamps", []) or []
        per_ts_forecasts = data.get("per_timestamp_forecasts", []) or []

        # compute local now and today
        now_local = dt_util.now()
        today_local = now_local.date()

        remainder_of_today: Dict[str, Any] = {}
        next_5_days: Dict[str, Any] = {}

        # period_forecasts is canonical mapping: date -> period_name -> { "indices": [..], ... }
        for date_key, pmap in period_forecasts.items():
            for pname, pdata in (pmap or {}).items():
                indices = pdata.get("indices") or []
                if not isinstance(indices, list) or not indices:
                    continue
                # determine whether any index is later today (local)
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
                    # next 5 days: pick dates > today and <= today+5
                    if today_local < dt_local.date() <= (today_local + timedelta(days=5)):
                        include_next_days = True
                        touched_dates.add(dt_local.date().isoformat())

                # sanitized period summary (remove heavy arrays)
                sanitized = dict(pdata)
                sanitized.pop("indices", None)
                # hide top-level score_10 when raw not enabled
                if not self._is_raw_enabled():
                    sanitized.pop("score_10", None)
                # Remove nested profile_used (we keep top-level profile_used only)
                sanitized.pop("profile_used", None)

                # Prepare aggregated raw values for this period by averaging canonical raw values
                # collected from per_timestamp_forecasts[indices]. This enables injecting numeric
                # component values similar to current_forecast.
                raw_agg: Dict[str, float] = {}
                counts: Dict[str, int] = {}
                try:
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
                        # keys of interest: wind (m/s), tide (m), wave (m), pressure_delta (hPa),
                        # moon_phase, temperature, plus safety keys (wind_gust, visibility_km, swell_period_s, precipitation_probability)
                        for k, keyname in (("wind", "wind"), ("tide", "tide"), ("wave", "wave"),
                                          ("pressure_delta", "pressure_delta"), ("moon_phase", "moon_phase"),
                                          ("temperature", "temperature"), ("wind_gust", "wind_gust"),
                                          ("visibility_km", "visibility_km"), ("swell_period_s", "swell_period_s"),
                                          ("precipitation_probability", "precipitation_probability")):
                            if raw.get(keyname) is None:
                                continue
                            try:
                                v = float(raw.get(keyname))
                            except Exception:
                                continue
                            if k not in raw_agg:
                                raw_agg[k] = v
                                counts[k] = 1
                            else:
                                raw_agg[k] += v
                                counts[k] += 1
                    # finalize averages
                    for k in list(raw_agg.keys()):
                        c = counts.get(k, 1) or 1
                        raw_agg[k] = raw_agg[k] / c
                except Exception:
                    # on any failure, leave raw_agg empty so augmentation just won't inject values
                    raw_agg = {}

                # augment components similarly (period-level components are aggregated; provide aggregated raw)
                comps = sanitized.get("components")
                sanitized["components"] = _augment_components_with_values_simple(comps, raw_agg or None, entry_units)

                # add safety_values strictly from aggregated canonical raw only
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

        # expose per_timestamp_forecasts and period_forecasts only when explicitly requested
        if self._is_raw_enabled():
            # Sanitize per-timestamp raw list to avoid repeated/duplicated profile_used and safety blocks
            raw_per_ts = data.get("per_timestamp_forecasts")
            if isinstance(raw_per_ts, list):
                sanitized_list = []
                for entry in raw_per_ts:
                    try:
                        e_copy = copy.deepcopy(entry)
                        # Remove per-entry profile_used and safety to ensure single top-level source.
                        e_copy.pop("profile_used", None)
                        e_copy.pop("safety", None)
                        sanitized_list.append(e_copy)
                    except Exception:
                        # If deep-copy fails for any entry, fall back to original
                        sanitized_list.append(entry)
                attrs["per_timestamp_forecasts"] = sanitized_list
            else:
                attrs["per_timestamp_forecasts"] = raw_per_ts

            attrs["period_forecasts"] = period_forecasts
            attrs["raw_payload"] = data.get("raw_payload") or data
        attrs["raw_output_enabled"] = bool(self._is_raw_enabled())

        # Top-level summary fields (simple, canonical)
        # Build profile_used and augment with scientific_name/info when a specific species was selected
        pu = current.get("profile_used")
        try:
            if pu is None:
                attrs["profile_used"] = None
            else:
                # Do not mutate original; work on a shallow copy for display augmentation
                pu_copy = dict(pu) if isinstance(pu, dict) else pu
                selected = getattr(self.coordinator, "species", None)

                # Helper: resolve species profile if coordinator.species is a string id
                def _resolve_species_profile(sel: Any) -> Optional[Dict[str, Any]]:
                    """
                    Accept either a dict or a species id string. If string, attempt to load
                    the bundled species_profiles.json and return the species dict.
                    Returns None if resolution fails.
                    """
                    if sel is None:
                        return None
                    if isinstance(sel, dict):
                        return sel
                    if isinstance(sel, str):
                        try:
                            base_dir = os.path.dirname(__file__)
                            spath = os.path.join(base_dir, "species_profiles.json")
                            with open(spath, "r", encoding="utf-8") as fh:
                                payload = json.load(fh)
                            species_map = payload.get("species", {}) if isinstance(payload, dict) else {}
                            prof = species_map.get(sel)
                            if prof and isinstance(prof, dict):
                                return prof
                            _LOGGER.debug("species resolution: species id %r not found in species_profiles.json", sel)
                            return None
                        except Exception as exc:
                            _LOGGER.exception("Failed to resolve species profile for %r: %s", sel, exc)
                            return None
                    # unsupported type
                    return None

                resolved = _resolve_species_profile(selected)

                # Only augment when the coordinator has a selected species profile that contains a scientific_name.
                # General profiles are expected not to contain scientific_name; in that case we leave profile_used untouched.
                if isinstance(pu_copy, dict) and isinstance(resolved, dict) and resolved.get("scientific_name"):
                    try:
                        pu_copy["scientific_name"] = resolved.get("scientific_name", "")
                        pu_copy["info"] = resolved.get("info", "")
                    except Exception:
                        # Best-effort — if augmentation fails, fall back to original pu_copy without blocking.
                        _LOGGER.debug("Failed to inject scientific_name/info into profile_used for display", exc_info=True)
                attrs["profile_used"] = pu_copy
        except Exception:
            # If anything unexpected happens during profile augmentation, fallback to raw value (do not break attr construction)
            attrs["profile_used"] = pu

        attrs["units"] = entry_units

        # Strict tide handling (NO backwards-compat fallbacks)
        tide_obj = data.get("tide")
        if not tide_obj or not isinstance(tide_obj, dict):
            _LOGGER.error("Coordinator tide block missing or invalid (strict): %r", tide_obj)
            attrs["next_high_tide"] = None
            attrs["next_low_tide"] = None
        else:
            def _build_tide_attr(tobj: dict) -> Optional[dict]:
                """
                Expect tobj to be {'timestamp': ISOZ} (height purposely omitted).
                Return dict with timestamp only; if malformed timestamp present, return None (strict).
                """
                if not isinstance(tobj, dict):
                    _LOGGER.error("Tide entry malformed (expected dict): %r", tobj)
                    return None
                ts = tobj.get("timestamp")
                if ts is None:
                    _LOGGER.error("Tide entry missing required key 'timestamp': %r", tobj)
                    return None
                # Ensure timestamp parses to ISOZ
                dt = _parse_dt_isoz(ts)
                if dt is None:
                    _LOGGER.error("Tide entry timestamp not ISOZ: %r", ts)
                    return None

                return {"timestamp": dt.isoformat().replace("+00:00", "Z")}

            nh_attr = _build_tide_attr(tide_obj.get("next_high"))
            nl_attr = _build_tide_attr(tide_obj.get("next_low"))

            attrs["next_high_tide"] = nh_attr
            attrs["next_low_tide"] = nl_attr

        # moon_phase: prefer score_calc_raw.raw.moon_phase then canonical top-level moon_phase array
        moon_numeric = None
        if score_calc_raw and isinstance(score_calc_raw, dict) and score_calc_raw.get("moon_phase") is not None:
            moon_numeric = score_calc_raw.get("moon_phase")
        elif "moon_phase" in data:
            mp = data.get("moon_phase")
            if isinstance(mp, (list, tuple)):
                # try to align using current index
                idx = current.get("index")
                if isinstance(idx, int) and idx < len(mp):
                    moon_numeric = mp[idx]
                else:
                    moon_numeric = mp[0] if mp else None
            else:
                moon_numeric = mp
        # Keep numeric moon_phase for compatibility, rounded
        attrs["moon_phase"] = _round_opt(moon_numeric, 6) if moon_numeric is not None else None
        # Also provide friendly name for user readability
        attrs["moon_phase_name"] = _moon_phase_name(moon_numeric) if moon_numeric is not None else None

        # Top-level current metrics (use formatted_weather produced by scoring — canonical numeric fields)
        formatted = (current.get("forecast_raw") or {}).get("formatted_weather") or {}
        # Temperature (display according to units) — only canonical numeric available here is temperature (C)
        temp_c = formatted.get("temperature")
        if temp_c is not None:
            t_val, t_unit = unit_helpers.temp_c_to_display(temp_c, entry_units)
            attrs["current_temperature"] = _format_with_unit(t_val, t_unit, ndigits=1) if t_val is not None else None
        else:
            attrs["current_temperature"] = None

        # Wind (convert canonical m/s to display) — merged string
        wind_m_s = formatted.get("wind")
        w_val, w_unit = _m_s_to_display(wind_m_s, entry_units)
        attrs["current_wind_speed"] = _format_with_unit(w_val, w_unit, ndigits=2) if w_val is not None else None

        # Gust (if provided) — merged string
        gust_m_s = formatted.get("wind_gust")
        g_val, g_unit = _m_s_to_display(gust_m_s, entry_units)
        attrs["current_wind_gust"] = _format_with_unit(g_val, g_unit, ndigits=2) if g_val is not None else None

        # Pressure (display according to units) — merged string
        p_hpa = formatted.get("pressure_hpa")
        if p_hpa is not None:
            p_val, p_unit = unit_helpers.pressure_hpa_to_display(p_hpa, entry_units)
            attrs["current_pressure"] = _format_with_unit(p_val, p_unit, ndigits=3) if p_val is not None else None
        else:
            attrs["current_pressure"] = None

        # Waves / swell (display according to units) — merged string
        wave_m = formatted.get("wave_height_m")
        if wave_m is not None:
            wave_val, wave_unit = unit_helpers.length_m_to_display(wave_m, entry_units)
            attrs["current_wave_height"] = _format_with_unit(wave_val, wave_unit, ndigits=3) if wave_val is not None else None
        else:
            attrs["current_wave_height"] = None

        attrs["current_swell_period_s"] = _round_opt(formatted.get("swell_period_s"), 1) if formatted.get("swell_period_s") is not None else None

        # Attribution and return
        attrs["attribution"] = ATTRIBUTION
        attrs[ATTR_ATTRIBUTION] = ATTRIBUTION
        return attrs


# --- Platform setup / unload (config entry integration) ---
async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    """
    Set up sensor platform for a config entry.

    Strict behavior: require explicit CONF_NAME in entry.data. No fallbacks allowed.
    """
    _LOGGER.debug("sensor.async_setup_entry called for entry %s", entry.entry_id)
    coordinator = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if coordinator is None:
        _LOGGER.error("Coordinator not found for entry %s; aborting platform setup", entry.entry_id)
        return

    # STRICT: require CONF_NAME present in entry.data
    name = entry.data.get(CONF_NAME)
    if not name:
        _LOGGER.error(
            "Config entry %s missing required '%s' in entry.data (strict); aborting sensor setup",
            entry.entry_id,
            CONF_NAME,
        )
        raise ValueError("Entry data missing 'name' (strict)")

    try:
        sensor = OFASensor(coordinator, name, entry=entry)
        async_add_entities([sensor], True)
        # register options update listener so options changes refresh the coordinator
        try:
            entry.add_update_listener(_async_options_updated)
        except Exception:
            # non-fatal; continue. The worst case is options updates won't trigger a refresh.
            _LOGGER.debug("Failed to attach entry update listener for entry %s", entry.entry_id)
    except Exception as exc:
        _LOGGER.exception("Failed to create OFASensor for entry %s: %s", entry.entry_id, exc)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """
    Optional unload handler for the sensor platform. Home Assistant will call this when
    the integration forwards unloads; nothing special needed here because entities
    are removed by the entity platform. Return True to indicate unload handled.
    """
    _LOGGER.debug("sensor.async_unload_entry called for entry %s", entry.entry_id)
    # nothing to persist/cleanup here for the simplified sensor; return True
    return True