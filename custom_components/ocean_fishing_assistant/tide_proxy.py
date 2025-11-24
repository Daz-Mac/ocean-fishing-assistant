from __future__ import annotations
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import asyncio

from homeassistant.util import dt as dt_util

# Skyfield (loaded lazily)
from skyfield.api import Loader, wgs84  # type: ignore
from skyfield import almanac as _almanac  # type: ignore
from skyfield.framelib import ecliptic_frame
import skyfield

import numpy as np

_LOGGER = logging.getLogger(__name__)

# constants
_DEFAULT_TTL = 15 * 60  # seconds
_TIDE_HALF_DAY_HOURS = 12.42
_SECONDS_PER_HOUR = 3600.0
_ALMANAC_SEARCH_DAYS = 3  # window to search for next transit with skyfield

# Constituent metadata (canonical)
CONSTITUENT_PERIOD_HOURS: Dict[str, float] = {
    "M2": 12.4206,
    "S2": 12.0,
    "N2": 12.6583,
    "K1": 23.9345,
    "O1": 25.8193,
    # a few extra short/overtone constituents for better default waveform
    "P1": 24.0659,
    "Q1": 26.8683,
    "S1": 24.0,
    "M4": 12.4206 / 2.0,
    "M6": 12.4206 / 3.0,
}

# Default relative amplitudes (heuristic; not station-specific)
CONSTITUENT_DEFAULT_RATIOS: Dict[str, float] = {
    "M2": 1.00,
    "S2": 0.25,
    "N2": 0.18,
    "K1": 0.45,
    "O1": 0.25,
    "P1": 0.12,
    "Q1": 0.08,
    "S1": 0.06,
    "M4": 0.06,
    "M6": 0.02,
}


class TideProxy:
    """
    TideProxy with NO persistence. Uses in-memory deterministic coefficients (A_cos, B_sin)
    derived from CONSTITUENT_DEFAULT_RATIOS when no explicit coef_vec is supplied.

    Public methods used by the integration:
      - get_tide_for_timestamps(timestamps)
      - compute_period_indices_for_timestamps(timestamps, mode=..., dawn_window_hours=...)
      - set_coefficients(coef_vec, bias=None)  <-- optional: you can seed real coefficients
    """

    def __init__(
        self,
        hass,
        latitude: float,
        longitude: float,
        ttl: int = _DEFAULT_TTL,
        *,
        coef_vec: Optional[Sequence[float]] = None,
        default_m2_amp: float = 1.0,
        bias: float = 0.0,
        auto_clamp_enabled: bool = False,
        min_height_floor: Optional[float] = None,
        max_amplitude_m: Optional[float] = None,
    ):
        self.hass = hass
        self.latitude = float(latitude or 0.0)
        self.longitude = float(longitude or 0.0)
        self._ttl = int(ttl)
        self._last_calc: Optional[datetime] = None
        self._cache: Optional[Dict[str, Any]] = None

        self._constituents = list(CONSTITUENT_PERIOD_HOURS.keys())
        self._bias = float(bias)
        self._auto_clamp_enabled = bool(auto_clamp_enabled)
        self._min_height_floor = None if min_height_floor is None else float(min_height_floor)
        self._max_amplitude_m = None if max_amplitude_m is None else float(max_amplitude_m)

        # Skyfield loader (lazy)
        try:
            data_dir = hass.config.path("custom_components", "ocean_fishing_assistant", "data")
        except Exception:
            from homeassistant.const import CONFIG_DIR  # type: ignore
            data_dir = os.path.join(CONFIG_DIR, "custom_components", "ocean_fishing_assistant", "data")
        os.makedirs(data_dir, exist_ok=True)
        self._loader = Loader(data_dir)
        self._sf_ts = None
        self._sf_eph = None
        self._sf_wgs = None
        self._sf_almanac = None
        self._load_lock = asyncio.Lock()

        # coefficient vector (A0,B0,A1,B1,...)
        if coef_vec is not None:
            arr = np.asarray(coef_vec, dtype=float)
            if arr.size == 2 * len(self._constituents):
                self._coef_vec = arr.copy()
            else:
                _LOGGER.warning("coef_vec length mismatch; using default built-ins")
                self._coef_vec = self._build_default_coef_vec(default_m2_amp)
        else:
            self._coef_vec = self._build_default_coef_vec(default_m2_amp)

        _LOGGER.debug(
            "TideProxy initialized lat=%s lon=%s coef_len=%d bias=%.3f clamp=%s",
            self.latitude,
            self.longitude,
            self._coef_vec.size,
            self._bias,
            self._auto_clamp_enabled,
        )

    def _build_default_coef_vec(self, m2_amp: float) -> np.ndarray:
        vals: List[float] = []
        for c in self._constituents:
            ratio = CONSTITUENT_DEFAULT_RATIOS.get(c, 0.0)
            a = float(m2_amp * ratio)
            b = 0.0
            vals.extend([a, b])
        return np.asarray(vals, dtype=float)

    def set_coefficients(self, coef_vec: Sequence[float], bias: Optional[float] = None) -> bool:
        try:
            arr = np.asarray(coef_vec, dtype=float)
            if arr.size != 2 * len(self._constituents):
                _LOGGER.error(
                    "set_coefficients: coef_vec length %d != expected %d", arr.size, 2 * len(self._constituents)
                )
                return False
            self._coef_vec = arr.copy()
            if bias is not None:
                self._bias = float(bias)
            self._cache = None
            _LOGGER.info("set_coefficients applied (len=%d) bias=%.3f", arr.size, self._bias)
            return True
        except Exception:
            _LOGGER.exception("set_coefficients failed")
            return False

    async def _ensure_loaded(self) -> None:
        if self._sf_eph is not None and self._sf_ts is not None:
            return
        async with self._load_lock:
            if self._sf_eph is not None and self._sf_ts is not None:
                return

            def _blocking_load():
                sf_ts = self._loader.timescale()
                sf_eph = self._loader("de421.bsp")
                sf_wgs = wgs84
                sf_almanac = _almanac
                version = getattr(skyfield, "__version__", "unknown")
                return sf_ts, sf_eph, sf_wgs, sf_almanac, version

            try:
                sf_ts, sf_eph, sf_wgs, sf_almanac, version = await self.hass.async_add_executor_job(
                    _blocking_load
                )
                self._sf_ts = sf_ts
                self._sf_eph = sf_eph
                self._sf_wgs = sf_wgs
                self._sf_almanac = sf_almanac
                _LOGGER.info("Skyfield loaded version=%s", version)
            except Exception:
                _LOGGER.exception("Failed to load Skyfield resources")
                raise

    async def get_tide_for_timestamps(self, timestamps: Sequence[str]) -> Dict[str, Any]:
        now = dt_util.now().astimezone(timezone.utc)
        # cache fast-path
        if self._last_calc and self._cache and (now - self._last_calc).total_seconds() < self._ttl:
            if self._cache.get("timestamps") == list(timestamps):
                return self._cache

        # parse timestamps into aware datetimes (UTC)
        dt_objs: List[datetime] = []
        for ts in timestamps:
            parsed = dt_util.parse_datetime(str(ts))
            if parsed is None:
                try:
                    # handle numeric epoch strings
                    v = float(ts)
                    if v > 1e12:  # probably milliseconds
                        v = v / 1000.0
                    parsed = datetime.fromtimestamp(v, tz=timezone.utc)
                except Exception:
                    parsed = None
            if parsed is None:
                _LOGGER.error("Unable to parse timestamp: %s", ts)
                return {
                    "timestamps": list(timestamps),
                    "tide_height_m": [None] * len(timestamps),
                    "confidence": "bad_timestamps",
                    "source": "tide_proxy",
                }
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            dt_objs.append(parsed)

        # ensure skyfield to compute helpers (moon phase/anchor)
        try:
            await self._ensure_loaded()
        except Exception:
            _LOGGER.exception("Skyfield unavailable")
            return {
                "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
                "tide_height_m": [None] * len(dt_objs),
                "confidence": "astronomical_unavailable",
                "source": "astronomical_unavailable",
            }

        sf_ts = self._sf_ts
        sf_eph = self._sf_eph
        sf_wgs = self._sf_wgs
        sf_almanac = self._sf_almanac

        # anchor: try moon transit, else first timestamp with a longitude shift
        try:
            moon_transit_dt = await self._async_find_next_moon_transit(sf_eph, sf_ts, sf_almanac, sf_wgs, now)
        except Exception:
            moon_transit_dt = None

        anchor_dt = moon_transit_dt or (dt_objs[0] if dt_objs else now)
        anchor_epoch = anchor_dt.timestamp() if anchor_dt else now.timestamp()
        period_seconds = _TIDE_HALF_DAY_HOURS * _SECONDS_PER_HOUR
        if moon_transit_dt is None:
            lon_shift = (self.longitude / 360.0) * period_seconds
            t_anchor = anchor_epoch - lon_shift
        else:
            t_anchor = anchor_epoch

        # build omegas
        periods_sec = {k: (CONSTITUENT_PERIOD_HOURS[k] * _SECONDS_PER_HOUR) for k in self._constituents}
        omegas = {k: 2.0 * math.pi / periods_sec[k] for k in periods_sec}

        # evaluate model using in-memory coef_vec
        coef_arr = np.asarray(self._coef_vec, dtype=float)
        A = coef_arr[0::2].astype(float)
        B = coef_arr[1::2].astype(float)
        t_rel = np.array([dt.timestamp() - t_anchor for dt in dt_objs], dtype=float)
        pred = np.zeros_like(t_rel)
        for i, c in enumerate(self._constituents):
            w = omegas[c]
            pred += A[i] * np.cos(w * t_rel) + B[i] * np.sin(w * t_rel)
        if float(self._bias) != 0.0:
            pred = pred + float(self._bias)

        # clamp/scale (optional)
        if self._auto_clamp_enabled:
            try:
                if self._max_amplitude_m is not None:
                    current_amp = float(np.max(pred) - np.min(pred))
                    if current_amp > float(self._max_amplitude_m) and current_amp > 1e-12:
                        mean_val = float(np.mean(pred))
                        scale = float(self._max_amplitude_m) / current_amp
                        pred = mean_val + (pred - mean_val) * scale
                if self._min_height_floor is not None:
                    pred = np.maximum(pred, float(self._min_height_floor))
            except Exception:
                _LOGGER.exception("Error applying clamp/scale")

        tide_heights = [round(float(v), 3) for v in pred.tolist()]

        # --- compute next high/low (timestamps + heights) for sensor compatibility (strict dict shape) ---
        next_high = None
        next_low = None
        next_high_height = None
        next_low_height = None
        try:
            now_ts = now.timestamp()
            # index of first timestamp >= now (fallback to 0)
            first_future_idx = 0
            for i, dt in enumerate(dt_objs):
                if dt.timestamp() >= now_ts:
                    first_future_idx = i
                    break

            # Analytic derivative + root-finding approach to find extrema between samples
            def _height_at_epoch(epoch_ts: float) -> float:
                t_rel = epoch_ts - t_anchor
                s = float(self._bias) if hasattr(self, "_bias") else 0.0
                for j, c in enumerate(self._constituents):
                    w = omegas[c]
                    s += float(A[j]) * math.cos(w * t_rel) + float(B[j]) * math.sin(w * t_rel)
                return s

            def _derivative_at_epoch(epoch_ts: float) -> float:
                t_rel = epoch_ts - t_anchor
                s = 0.0
                for j, c in enumerate(self._constituents):
                    w = omegas[c]
                    s += -float(A[j]) * w * math.sin(w * t_rel) + float(B[j]) * w * math.cos(w * t_rel)
                return s

            def _second_derivative_at_epoch(epoch_ts: float) -> float:
                t_rel = epoch_ts - t_anchor
                s = 0.0
                for j, c in enumerate(self._constituents):
                    w = omegas[c]
                    s += -float(A[j]) * (w ** 2) * math.cos(w * t_rel) - float(B[j]) * (w ** 2) * math.sin(w * t_rel)
                return s

            def _find_root_bisection(f, a: float, b: float, maxiter: int = 60, tol: float = 1e-3) -> Optional[float]:
                fa = f(a)
                fb = f(b)
                if abs(fa) < 1e-12:
                    return a
                if abs(fb) < 1e-12:
                    return b
                if fa * fb > 0:
                    return None
                lo = a
                hi = b
                for _ in range(maxiter):
                    mid = 0.5 * (lo + hi)
                    fm = f(mid)
                    if abs(fm) < 1e-9 or (hi - lo) < tol:
                        return mid
                    if fa * fm <= 0:
                        hi = mid
                        fb = fm
                    else:
                        lo = mid
                        fa = fm
                return 0.5 * (lo + hi)

            candidates: List[float] = []
            # search each interval between consecutive samples for derivative sign change
            for i in range(max(0, first_future_idx - 1), len(dt_objs) - 1):
                t0 = dt_objs[i].timestamp()
                t1 = dt_objs[i + 1].timestamp()
                d0 = _derivative_at_epoch(t0)
                d1 = _derivative_at_epoch(t1)
                # if derivative near zero at sample point, consider that point a candidate
                if abs(d0) < 1e-12 and t0 >= now_ts:
                    candidates.append(t0)
                if d0 * d1 < 0:
                    root = _find_root_bisection(_derivative_at_epoch, t0, t1)
                    if root is not None and root >= now_ts:
                        candidates.append(root)

            # classify candidates into maxima/minima using second derivative
            maxima: List[tuple] = []
            minima: List[tuple] = []
            for rt in candidates:
                h = _height_at_epoch(rt)
                sec = _second_derivative_at_epoch(rt)
                if sec < 0:
                    maxima.append((rt, h))
                else:
                    minima.append((rt, h))

            # pick earliest future max/min
            next_high_tuple = None
            next_low_tuple = None
            if maxima:
                next_high_tuple = min(maxima, key=lambda x: x[0])
            if minima:
                next_low_tuple = min(minima, key=lambda x: x[0])

            # fallback: if none found, use sampled argmax/argmin in future window
            if next_high_tuple is None:
                rel = pred[first_future_idx:]
                if rel.size:
                    idx = int(np.argmax(rel)) + first_future_idx
                    next_high_tuple = (dt_objs[idx].timestamp(), float(pred[idx]))
            if next_low_tuple is None:
                rel = pred[first_future_idx:]
                if rel.size:
                    idx = int(np.argmin(rel)) + first_future_idx
                    next_low_tuple = (dt_objs[idx].timestamp(), float(pred[idx]))

            if next_high_tuple is not None:
                nh_ts, nh_h = next_high_tuple
                next_high = datetime.fromtimestamp(nh_ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                next_high_height = float(round(float(nh_h), 3))
            if next_low_tuple is not None:
                nl_ts, nl_h = next_low_tuple
                next_low = datetime.fromtimestamp(nl_ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                next_low_height = float(round(float(nl_h), 3))
        except Exception:
            _LOGGER.debug("Failed to compute next_high/next_low", exc_info=True)

        # moon phases for payload (best-effort)
        try:
            earth = sf_eph["earth"]
            sun_obj = sf_eph["sun"]
            moon_obj = sf_eph["moon"]
            times_list = [sf_ts.from_datetime(dt) for dt in dt_objs]
            moon_phases: List[Optional[float]] = []
            for t in times_list:
                sun_app = earth.at(t).observe(sun_obj).apparent()
                moon_app = earth.at(t).observe(moon_obj).apparent()
                sun_ecl = sun_app.frame_latlon(ecliptic_frame)
                moon_ecl = moon_app.frame_latlon(ecliptic_frame)
                lon_sun = float(sun_ecl[1].degrees)
                lon_moon = float(moon_ecl[1].degrees)
                diff = (lon_moon - lon_sun) % 360.0
                moon_phases.append(diff / 360.0)
        except Exception:
            moon_phases = [None] * len(dt_objs)

        # Build strict dicts for next_high/next_low (sensor expects dict shape)
        next_high_obj = None
        next_low_obj = None
        if next_high is not None:
            try:
                next_high_obj = {"timestamp": next_high, "height_m": next_high_height}
            except Exception:
                next_high_obj = None
        if next_low is not None:
            try:
                next_low_obj = {"timestamp": next_low, "height_m": next_low_height}
            except Exception:
                next_low_obj = None

        raw_tide: Dict[str, Any] = {
            "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
            "tide_height_m": tide_heights,
            "tide_phase": moon_phases,
            "tide_strength": float(round(_compute_tide_strength(moon_phases[0] if moon_phases else None), 3)),
            "confidence": "in_memory_model",
            "source": "in_memory_harmonic_model",
            "_helpers": {
                "constituents": self._constituents,
                "t_anchor": float(t_anchor),
                "period_seconds": float(period_seconds),
                "coef_vec_len": int(self._coef_vec.size),
            },
            # Strict canonical tide objects expected by sensor.py:
            "next_high": next_high_obj,
            "next_low": next_low_obj,
        }

        self._cache = raw_tide
        self._last_calc = now
        return raw_tide

    async def _async_find_next_moon_transit(self, sf_eph, sf_ts, sf_almanac, sf_wgs, start_dt: datetime) -> Optional[datetime]:
        try:
            start_dt = start_dt.astimezone(timezone.utc)
            t0 = sf_ts.utc(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, start_dt.minute, start_dt.second)
            end_dt = start_dt + timedelta(days=_ALMANAC_SEARCH_DAYS)
            t1 = sf_ts.utc(end_dt.year, end_dt.month, end_dt.day, end_dt.hour, end_dt.minute, end_dt.second)
            topos = sf_wgs.latlon(self.latitude, self.longitude)
            f = sf_almanac.meridian_transits(sf_eph, sf_eph["moon"], topos)
            times, events = sf_almanac.find_discrete(t0, t1, f)
            for t in times:
                try:
                    dt = t.utc_datetime().replace(tzinfo=timezone.utc)
                except Exception:
                    dt = datetime.fromtimestamp(t.tt).replace(tzinfo=timezone.utc)
                if dt and dt > start_dt:
                    return dt
            return None
        except Exception:
            _LOGGER.debug("Moon transit search failed", exc_info=True)
            return None

    async def compute_period_indices_for_timestamps(
        self,
        timestamps: Sequence[str],
        mode: str = "full_day",
        dawn_window_hours: float = 1.0,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Compute period indices mapping, used by the coordinator and formatter.
        """
        dt_objs: List[datetime] = []
        for ts in timestamps:
            parsed = dt_util.parse_datetime(str(ts))
            if parsed is None:
                try:
                    v = float(ts)
                    if v > 1e12:
                        v = v / 1000.0
                    parsed = datetime.fromtimestamp(v, tz=timezone.utc)
                except Exception as exc:
                    raise ValueError(f"Unable to parse timestamp '{ts}': {exc}") from exc
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            dt_objs.append(parsed)
        if not dt_objs:
            return {}

        index_dt_pairs = list(enumerate(dt_objs))
        dates_needed = sorted({dt.date() for dt in dt_objs})

        await self._ensure_loaded()
        sf_ts = self._sf_ts
        sf_eph = self._sf_eph
        sf_wgs = self._sf_wgs
        sf_almanac = self._sf_almanac
        earth = sf_eph["earth"]
        topos = sf_wgs.latlon(self.latitude, self.longitude)

        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        def _iso_z(dt: datetime) -> str:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")

        for d in dates_needed:
            try:
                day_start = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc)
                day_end = day_start + timedelta(days=1)

                if mode == "dawn_dusk":
                    expand = timedelta(hours=12)
                    t0_exp = sf_ts.utc((day_start - expand).year, (day_start - expand).month, (day_start - expand).day, (day_start - expand).hour, (day_start - expand).minute, (day_start - expand).second)
                    t1_exp = sf_ts.utc((day_end + expand).year, (day_end + expand).month, (day_end + expand).day, (day_end + expand).hour, (day_end + expand).minute, (day_end + expand).second)
                    f = sf_almanac.sunrise_sunset(sf_eph, topos)
                    times, events = sf_almanac.find_discrete(t0_exp, t1_exp, f)
                    if not times:
                        raise RuntimeError(f"No sunrise/sunset events found for date {d.isoformat()} at location lat={self.latitude},lon={self.longitude}")
                    sunrise_candidates: List[datetime] = []
                    sunset_candidates: List[datetime] = []
                    for t, ev in zip(times, events):
                        try:
                            evt_dt = t.utc_datetime().replace(tzinfo=timezone.utc)
                        except Exception:
                            evt_dt = datetime.fromtimestamp(t.tt).replace(tzinfo=timezone.utc)
                        if bool(ev):
                            sunrise_candidates.append(evt_dt)
                        else:
                            sunset_candidates.append(evt_dt)
                    if not sunrise_candidates or not sunset_candidates:
                        raise RuntimeError(f"Unable to determine sunrise or sunset for date {d.isoformat()} at lat={self.latitude},lon={self.longitude}")
                    morning_target = day_start + timedelta(hours=6)
                    evening_target = day_start + timedelta(hours=18)
                    sunrise_dt = min(sunrise_candidates, key=lambda e: abs((e - morning_target).total_seconds()))
                    sunset_dt = min(sunset_candidates, key=lambda e: abs((e - evening_target).total_seconds()))
                    dawn_start = sunrise_dt - timedelta(hours=dawn_window_hours)
                    dawn_end = sunrise_dt + timedelta(hours=dawn_window_hours)
                    dusk_start = sunset_dt - timedelta(hours=dawn_window_hours)
                    dusk_end = sunset_dt + timedelta(hours=dawn_window_hours)
                    date_key = d.isoformat()
                    result.setdefault(date_key, {})
                    dawn_indices: List[int] = []
                    dusk_indices: List[int] = []
                    for idx, dt in index_dt_pairs:
                        if dt >= dawn_start and dt < dawn_end:
                            dawn_indices.append(idx)
                        if dt >= dusk_start and dt < dusk_end:
                            dusk_indices.append(idx)
                    result[date_key]["dawn"] = {"indices": dawn_indices, "start": _iso_z(dawn_start), "end": _iso_z(dawn_end)}
                    result[date_key]["dusk"] = {"indices": dusk_indices, "start": _iso_z(dusk_start), "end": _iso_z(dusk_end)}
                else:
                    date_key = d.isoformat()
                    result.setdefault(date_key, {})
                    p00_start = day_start
                    p00_end = day_start + timedelta(hours=6)
                    p06_start = p00_end
                    p06_end = day_start + timedelta(hours=12)
                    p12_start = p06_end
                    p12_end = day_start + timedelta(hours=18)
                    p18_start = p12_end
                    p18_end = day_end
                    periods = [
                        ("period_00_06", p00_start, p00_end),
                        ("period_06_12", p06_start, p06_end),
                        ("period_12_18", p12_start, p12_end),
                        ("period_18_24", p18_start, p18_end),
                    ]
                    for pname, pstart, pend in periods:
                        indices: List[int] = []
                        for idx, dt in index_dt_pairs:
                            if dt >= pstart and dt < pend:
                                indices.append(idx)
                        result[date_key][pname] = {"indices": indices, "start": _iso_z(pstart), "end": _iso_z(pend)}
            except Exception as exc:
                _LOGGER.exception("compute_period_indices_for_timestamps failed for date %s: %s", d.isoformat(), exc)
                raise
        return result


# module helpers
def _compute_tide_strength(phase: Optional[float]) -> float:
    try:
        if phase is None:
            return 0.5
        p = float(phase)
        p = max(0.0, min(1.0, p))
        dist_new = min(abs(p - 0.0), abs(1.0 - p))
        dist_full = abs(p - 0.5)
        dist = min(dist_new, dist_full)
        val = max(0.0, 1.0 - (dist / 0.25))
        return float(max(0.0, min(1.0, val)))
    except Exception:
        return 0.5


def _coerce_phase(phase: Any) -> Optional[float]:
    """
    Normalize a moon phase value to 0..1 float.
    sensor.py expects tide_proxy._coerce_phase to exist.
    """
    if phase is None:
        return None
    try:
        p = float(phase)
        p = p % 1.0
        if p < 0:
            p += 1.0
        return float(p)
    except Exception:
        return None