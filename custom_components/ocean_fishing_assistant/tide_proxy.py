from __future__ import annotations
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
import asyncio
import os

from homeassistant.util import dt as dt_util

# Skyfield is required for astronomical helpers (loaded lazily)
from skyfield.api import Loader, wgs84  # type: ignore
from skyfield import almanac as _almanac  # type: ignore
from skyfield.framelib import ecliptic_frame
import skyfield  # for version reporting

import numpy as np

_LOGGER = logging.getLogger(__name__)

# constants
_DEFAULT_TTL = 15 * 60  # seconds
_TIDE_HALF_DAY_HOURS = 12.42
_SECONDS_PER_HOUR = 3600.0
_ALMANAC_SEARCH_DAYS = 3  # window to search for next transit with skyfield

# ---- Expanded constituent metadata (canonical names: no leading underscore) ----
# Periods are in hours.
CONSTITUENT_PERIOD_HOURS: Dict[str, float] = {
    # primary astronomical constituents
    "M2": 12.4206,   # principal lunar semidiurnal
    "S2": 12.0,      # principal solar semidiurnal
    "N2": 12.6583,   # larger lunar elliptic semidiurnal
    "K1": 23.9345,   # lunisolar diurnal
    "O1": 25.8193,   # lunar diurnal
    # additional diurnal constituents
    "P1": 24.0659,
    "Q1": 26.8683,
    "S1": 24.0,
    # overtides / shallow-water terms
    "M4": 12.4206 / 2.0,
    "M6": 12.4206 / 3.0,
    "2N2": 12.6583 / 2.0,
    # longer-period constituents (small influence on short-range forecasts)
    "Mf": 14.765294,  # lunar fortnightly (hours)
    "Mm": 27.55455,   # lunar monthly (hours)
}

# Default relative amplitudes (heuristic, NOT station-specific).
CONSTITUENT_DEFAULT_RATIOS: Dict[str, float] = {
    "M2": 1.00,    # dominant semidiurnal
    "S2": 0.25,    # solar semidiurnal
    "N2": 0.18,
    "K1": 0.45,    # prominent diurnal
    "O1": 0.25,
    "P1": 0.12,
    "Q1": 0.08,
    "S1": 0.06,
    "M4": 0.06,    # small overtide
    "M6": 0.02,
    "2N2": 0.05,
    "Mf": 0.10,
    "Mm": 0.08,
}


class TideProxy:
    """
    TideProxy with expanded constituent set and deterministic defaults.
    No calibration or persistence of tide_coeffs.json performed here.
    """

    def __init__(
        self,
        hass,
        latitude: float,
        longitude: float,
        ttl: int = _DEFAULT_TTL,
        *,
        # if provided, must be 2 * len(constituents) floats (A0, B0, A1, B1, ...)
        coef_vec: Optional[Sequence[float]] = None,
        # if coef_vec is None: default_m2_amp is the amplitude assigned to M2; others scaled by ratios
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

        # internal harmonic coefficients: keep the order consistent with CONSTITUENT_PERIOD_HOURS keys
        self._constituents = list(CONSTITUENT_PERIOD_HOURS.keys())
        self._coef_vec: Optional[np.ndarray] = None
        self._bias = float(bias)

        # clamp/scale options
        self._auto_clamp_enabled = bool(auto_clamp_enabled)
        self._min_height_floor: Optional[float] = None if min_height_floor is None else float(min_height_floor)
        self._max_amplitude_m: Optional[float] = None if max_amplitude_m is None else float(max_amplitude_m)

        # skyfield loader and resources (lazy)
        try:
            data_dir = hass.config.path("custom_components", "ocean_fishing_assistant", "data")
        except Exception:
            # fallback if hass.config.path is unavailable
            from homeassistant.const import CONFIG_DIR  # type: ignore

            data_dir = os.path.join(CONFIG_DIR, "custom_components", "ocean_fishing_assistant", "data")

        os.makedirs(data_dir, exist_ok=True)
        self._loader = Loader(data_dir)
        self._sf_ts = None
        self._sf_eph = None
        self._sf_wgs = None
        self._sf_almanac = None
        self._load_lock = asyncio.Lock()

        # If user provided a coefficient vector, validate and use it; otherwise build defaults
        if coef_vec is not None:
            arr = np.asarray(coef_vec, dtype=float)
            if arr.size == 2 * len(self._constituents):
                self._coef_vec = arr.copy()
            else:
                _LOGGER.warning("coef_vec length mismatch; ignoring provided coef_vec and using defaults")
                self._coef_vec = self._build_default_coef_vec(default_m2_amp)
        else:
            self._coef_vec = self._build_default_coef_vec(default_m2_amp)

        _LOGGER.debug(
            "TideProxy initialized lat=%s lon=%s coef_len=%d bias=%s clamp=%s min_floor=%s max_amp=%s constituents=%s",
            self.latitude,
            self.longitude,
            self._coef_vec.size if self._coef_vec is not None else 0,
            self._bias,
            self._auto_clamp_enabled,
            self._min_height_floor,
            self._max_amplitude_m,
            self._constituents,
        )

    # ---- coefficient helpers ----

    def _build_default_coef_vec(self, m2_amp: float) -> np.ndarray:
        """
        Build a default coefficient vector from CONSTITUENT_DEFAULT_RATIOS.
        Set cosine (A) coefficients to ratio * m2_amp and sine (B) coefficients to 0.0.
        """
        vals: List[float] = []
        for c in self._constituents:
            ratio = CONSTITUENT_DEFAULT_RATIOS.get(c, 0.0)
            a = float(m2_amp * ratio)
            b = 0.0
            vals.append(a)
            vals.append(b)
        return np.asarray(vals, dtype=float)

    def set_coefficients(self, coef_vec: Sequence[float], bias: Optional[float] = None) -> bool:
        """
        Replace the in-memory coefficient vector. Returns True on success.
        coef_vec must be length 2 * num_constituents.
        Optionally set a small bias (meters).
        """
        try:
            arr = np.asarray(coef_vec, dtype=float)
            if arr.size != 2 * len(self._constituents):
                _LOGGER.error("set_coefficients: coef_vec length %d does not match expected %d", arr.size, 2 * len(self._constituents))
                return False
            self._coef_vec = arr.copy()
            if bias is not None:
                self._bias = float(bias)
            _LOGGER.info("set_coefficients: updated in-memory coefficients (len=%d) bias=%.3f", arr.size, self._bias)
            # clear cache so new coeffs are used immediately
            self._cache = None
            return True
        except Exception:
            _LOGGER.exception("set_coefficients failed")
            return False

    # ---- Skyfield lazy load ----

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
                version_tuple = getattr(skyfield, "VERSION", None)
                return sf_ts, sf_eph, sf_wgs, sf_almanac, version, version_tuple

            try:
                sf_ts, sf_eph, sf_wgs, sf_almanac, version, version_tuple = await self.hass.async_add_executor_job(_blocking_load)
                self._sf_ts = sf_ts
                self._sf_eph = sf_eph
                self._sf_wgs = sf_wgs
                self._sf_almanac = sf_almanac
                _LOGGER.info("Skyfield loaded version=%s", version)
            except Exception:
                _LOGGER.exception("Failed to load Skyfield resources")
                raise

    # ---- Tide computation ----

    async def get_tide_for_timestamps(self, timestamps: Sequence[str]) -> Dict[str, Any]:
        """
        Compute tide payload for given ISO timestamps (UTC expected).
        Uses in-memory coefficients; if none present (shouldn't happen) returns failure payload.
        """
        now = dt_util.now().astimezone(timezone.utc)

        # simple cache by identical timestamp list
        if self._last_calc and self._cache and (now - self._last_calc).total_seconds() < self._ttl:
            if self._cache.get("timestamps") == list(timestamps):
                return self._cache

        # parse timestamps
        try:
            dt_objs: List[datetime] = []
            for ts in timestamps:
                parsed = dt_util.parse_datetime(str(ts))
                if parsed is None:
                    raise ValueError(f"parse_datetime returned None for {ts}")
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                else:
                    parsed = parsed.astimezone(timezone.utc)
                dt_objs.append(parsed)
        except Exception:
            _LOGGER.exception("Failed to parse timestamps")
            return {
                "timestamps": list(timestamps),
                "tide_height_m": [None] * len(timestamps),
                "tide_phase": None,
                "tide_strength": 0.5,
                "confidence": "bad_timestamps",
                "source": "tide_proxy",
            }

        # ensure skyfield loaded for helpers
        try:
            await self._ensure_loaded()
        except Exception:
            _LOGGER.exception("Skyfield unavailable")
            return {
                "timestamps": [dt.isoformat().replace('+00:00', 'Z') for dt in dt_objs],
                "tide_height_m": [None] * len(dt_objs),
                "tide_phase": None,
                "tide_strength": 0.5,
                "confidence": "astronomical_unavailable",
                "source": "astronomical_unavailable",
            }

        sf_ts = self._sf_ts
        sf_eph = self._sf_eph
        sf_wgs = self._sf_wgs
        sf_almanac = self._sf_almanac

        # attempt moon transit for anchor
        try:
            moon_transit_dt = await self._async_find_next_moon_transit(sf_eph, sf_ts, sf_almanac, sf_wgs, now)
        except Exception:
            _LOGGER.debug("Moon transit search failed")
            moon_transit_dt = None

        # build skyfield time objects
        times_list = [sf_ts.from_datetime(dt) for dt in dt_objs]

        earth = sf_eph["earth"]
        sun_obj = sf_eph["sun"]
        moon_obj = sf_eph["moon"]
        topos = sf_wgs.latlon(self.latitude, self.longitude)

        # moon altitudes
        moon_altitudes: List[Optional[float]] = []
        for t in times_list:
            try:
                astrom = (earth + topos).at(t).observe(moon_obj).apparent()
                alt, az, dist = astrom.altaz()
                moon_altitudes.append(float(getattr(alt, "degrees", 0.0)))
            except Exception:
                moon_altitudes.append(None)

        # moon phases
        moon_phases: List[Optional[float]] = []
        for t in times_list:
            try:
                sun_app = earth.at(t).observe(sun_obj).apparent()
                moon_app = earth.at(t).observe(moon_obj).apparent()
                sun_ecl = sun_app.frame_latlon(ecliptic_frame)
                moon_ecl = moon_app.frame_latlon(ecliptic_frame)
                lon_sun = float(sun_ecl[1].degrees)
                lon_moon = float(moon_ecl[1].degrees)
                diff = (lon_moon - lon_sun) % 360.0
                moon_phases.append(diff / 360.0)
            except Exception:
                moon_phases.append(None)

        # anchor and tide strength
        moon_phase_scalar = None
        try:
            if moon_transit_dt is not None:
                t_anchor_sf = sf_ts.from_datetime(moon_transit_dt)
                sun_app = earth.at(t_anchor_sf).observe(sun_obj).apparent()
                moon_app = earth.at(t_anchor_sf).observe(moon_obj).apparent()
                sun_ecl = sun_app.frame_latlon(ecliptic_frame)
                moon_ecl = moon_app.frame_latlon(ecliptic_frame)
                lon_sun = float(sun_ecl[1].degrees)
                lon_moon = float(moon_ecl[1].degrees)
                moon_phase_scalar = (((lon_moon - lon_sun) % 360.0) / 360.0)
            else:
                moon_phase_scalar = moon_phases[0] if moon_phases else None
        except Exception:
            moon_phase_scalar = moon_phases[0] if moon_phases else None

        tide_strength = _compute_tide_strength(moon_phase_scalar)
        anchor_dt = moon_transit_dt or (dt_objs[0] if dt_objs else now)
        anchor_epoch = anchor_dt.timestamp() if anchor_dt else now.timestamp()
        period_seconds = _TIDE_HALF_DAY_HOURS * _SECONDS_PER_HOUR

        # simple lon shift if no transit
        if moon_transit_dt is None:
            lon_shift = (self.longitude / 360.0) * period_seconds
            t_anchor = anchor_epoch - lon_shift
        else:
            t_anchor = anchor_epoch

        # omegas per constituent
        periods_sec = {k: (CONSTITUENT_PERIOD_HOURS[k] * _SECONDS_PER_HOUR) for k in self._constituents}
        omegas = {k: 2.0 * math.pi / periods_sec[k] for k in periods_sec}

        # ensure coefficients present
        if self._coef_vec is None:
            _LOGGER.error("No harmonic coefficients available (internal).")
            return {
                "timestamps": [dt.isoformat().replace('+00:00', 'Z') for dt in dt_objs],
                "tide_height_m": [None] * len(dt_objs),
                "tide_phase": moon_phases,
                "tide_strength": float(round(tide_strength, 3)),
                "confidence": "no_coefficients",
                "source": "no_coefficients",
                "_helpers": {
                    "constituents": self._constituents,
                    "t_anchor": float(t_anchor),
                    "period_seconds": float(period_seconds),
                    "coef_vec_len": 2 * len(self._constituents),
                },
            }

        # evaluate model
        coef_arr = np.asarray(self._coef_vec, dtype=float)
        A = coef_arr[0::2].astype(float)
        B = coef_arr[1::2].astype(float)
        t_rel = np.array([dt.timestamp() - t_anchor for dt in dt_objs], dtype=float)
        pred = np.zeros_like(t_rel)
        for i, c in enumerate(self._constituents):
            w = omegas[c]
            pred += A[i] * np.cos(w * t_rel) + B[i] * np.sin(w * t_rel)
        # apply bias
        if float(self._bias) != 0.0:
            pred = pred + float(self._bias)

        # preserve original for diagnostics
        orig_pred = pred.copy()

        # clamp/scale logic (optional)
        clamp_applied = False
        clamp_details = {
            "auto_clamp_enabled": bool(self._auto_clamp_enabled),
            "min_height_floor": self._min_height_floor,
            "max_amplitude_m": self._max_amplitude_m,
            "was_scaled": False,
            "was_clamped_floor": False,
            "scale_factor": 1.0,
        }

        if self._auto_clamp_enabled:
            try:
                if self._max_amplitude_m is not None:
                    current_amp = float(np.max(pred) - np.min(pred))
                    if current_amp > float(self._max_amplitude_m) and current_amp > 1e-12:
                        mean_val = float(np.mean(pred))
                        scale = float(self._max_amplitude_m) / current_amp
                        pred = mean_val + (pred - mean_val) * scale
                        clamp_applied = True
                        clamp_details["was_scaled"] = True
                        clamp_details["scale_factor"] = float(scale)
                        _LOGGER.debug("Amplitude scaling applied: current_amp=%.3f max_allowed=%.3f scale=%.3f", current_amp, self._max_amplitude_m, scale)

                if self._min_height_floor is not None:
                    below = pred < float(self._min_height_floor)
                    if np.any(below):
                        pred = np.maximum(pred, float(self._min_height_floor))
                        clamp_applied = True
                        clamp_details["was_clamped_floor"] = True
                        _LOGGER.debug("Floor clamping applied: min_floor=%.3f (some predictions raised)", self._min_height_floor)
            except Exception:
                _LOGGER.exception("Error applying clamp/scale")

        tide_heights = [round(float(v), 3) for v in pred.tolist()]

        # scalar evaluators (used for refining extrema)
        def _tide_val(epoch_sec: float) -> float:
            rel = epoch_sec - t_anchor
            v = 0.0
            for i, c in enumerate(self._constituents):
                w = omegas[c]
                a = float(A[i])
                b = float(B[i])
                v += a * math.cos(w * rel) + b * math.sin(w * rel)
            v += float(self._bias)
            # approximate application of scaling/clamping to scalar eval to keep extrema consistent
            if self._auto_clamp_enabled:
                try:
                    if self._max_amplitude_m is not None:
                        orig_amp = float(np.max(orig_pred) - np.min(orig_pred))
                        if orig_amp > 1e-12 and orig_amp > float(self._max_amplitude_m):
                            mean_val = float(np.mean(orig_pred))
                            raw_v = v
                            scale = float(self._max_amplitude_m) / orig_amp
                            v = mean_val + (raw_v - mean_val) * scale
                    if self._min_height_floor is not None and v < float(self._min_height_floor):
                        v = float(self._min_height_floor)
                except Exception:
                    pass
            return v

        def _tide_derivative(epoch_sec: float) -> float:
            rel = epoch_sec - t_anchor
            s = 0.0
            for i, c in enumerate(self._constituents):
                w = omegas[c]
                a = float(A[i])
                b = float(B[i])
                s += (-a * w * math.sin(w * rel) + b * w * math.cos(w * rel))
            # scale derivative when amplitude was scaled
            if self._auto_clamp_enabled and self._max_amplitude_m is not None:
                try:
                    orig_amp = float(np.max(orig_pred) - np.min(orig_pred))
                    if orig_amp > 1e-12 and orig_amp > float(self._max_amplitude_m):
                        scale = float(self._max_amplitude_m) / orig_amp
                        s = s * scale
                except Exception:
                    pass
            return s

        # extremum finder
        def _find_extremum_near(t_guess: float, half_window: float = 10800.0) -> Tuple[float, float]:
            a = t_guess - half_window
            b = t_guess + half_window
            steps = 24
            xs = [a + i * (b - a) / steps for i in range(steps + 1)]
            dvals = []
            for x in xs:
                try:
                    dvals.append(_tide_derivative(x))
                except Exception:
                    dvals.append(float("nan"))

            bracket = None
            for i in range(len(xs) - 1):
                fa = dvals[i]
                fb = dvals[i + 1]
                if math.isnan(fa) or math.isnan(fb):
                    continue
                if fa == 0.0:
                    bracket = (xs[i], xs[i])
                    break
                if fa * fb < 0.0:
                    bracket = (xs[i], xs[i + 1])
                    break

            if bracket is None:
                return t_guess, _tide_val(t_guess)

            aa, bb = bracket
            fa = _tide_derivative(aa)
            fb = _tide_derivative(bb)

            tol = 1e-6
            max_iter = 60
            for _ in range(max_iter):
                m = 0.5 * (aa + bb)
                fm = _tide_derivative(m)
                if abs(fm) < tol:
                    return m, _tide_val(m)
                if fa * fm <= 0:
                    bb = m
                    fb = fm
                else:
                    aa = m
                    fa = fm
            m = 0.5 * (aa + bb)
            return m, _tide_val(m)

        # compute next high/low by sampling
        try:
            now_epoch = now.timestamp()
            horizon_seconds = 48 * 3600.0
            sample_dt = 5 * 60.0
            n_samples = int(horizon_seconds // sample_dt) + 1
            sample_epochs = [now_epoch + i * sample_dt for i in range(n_samples)]

            se = np.array(sample_epochs, dtype=float) - t_anchor
            sval = np.zeros_like(se)
            for i, c in enumerate(self._constituents):
                w = omegas[c]
                sval += float(A[i]) * np.cos(w * se) + float(B[i]) * np.sin(w * se)
            if float(self._bias) != 0.0:
                sval = sval + float(self._bias)

            # apply clamp/scale to sampled values
            if self._auto_clamp_enabled:
                try:
                    if self._max_amplitude_m is not None:
                        current_amp = float(np.max(sval) - np.min(sval))
                        if current_amp > float(self._max_amplitude_m) and current_amp > 1e-12:
                            mean_val = float(np.mean(sval))
                            scale = float(self._max_amplitude_m) / current_amp
                            sval = mean_val + (sval - mean_val) * scale
                    if self._min_height_floor is not None:
                        sval = np.maximum(sval, float(self._min_height_floor))
                except Exception:
                    _LOGGER.exception("Failed to apply clamp/scale to sampled values")

            max_idx = int(float(np.argmax(sval)))
            min_idx = int(float(np.argmin(sval)))
            max_epoch = sample_epochs[max_idx]
            min_epoch = sample_epochs[min_idx]
            refined_high_epoch, refined_high_val = _find_extremum_near(max_epoch, half_window=period_seconds / 4.0)
            refined_low_epoch, refined_low_val = _find_extremum_near(min_epoch, half_window=period_seconds / 4.0)

            next_high_dt = datetime.fromtimestamp(refined_high_epoch, tz=timezone.utc)
            next_low_dt = datetime.fromtimestamp(refined_low_epoch, tz=timezone.utc)
            next_high_height = round(float(refined_high_val), 3)
            next_low_height = round(float(refined_low_val), 3)
        except Exception:
            _LOGGER.exception("Failed to compute next_high/next_low")
            next_high_dt = None
            next_low_dt = None
            next_high_height = None
            next_low_height = None

        raw_tide: Dict[str, Any] = {
            "timestamps": [dt.isoformat().replace("+00:00", "Z") for dt in dt_objs],
            "tide_height_m": tide_heights,
            "tide_phase": moon_phases,
            "tide_strength": float(round(tide_strength, 3)),
            "next_high": next_high_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if next_high_dt else "",
            "next_low": next_low_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if next_low_dt else "",
            "next_high_height_m": next_high_height,
            "next_low_height_m": next_low_height,
            "confidence": "in_memory_model",
            "source": "in_memory_harmonic_model",
            "_helpers": {
                "constituents": self._constituents,
                "t_anchor": float(t_anchor),
                "period_seconds": float(period_seconds),
                "coef_vec_len": int(self._coef_vec.size),
            },
            "_clamp": clamp_details if clamp_applied else {"auto_clamp_enabled": self._auto_clamp_enabled, "applied": False},
        }

        self._cache = raw_tide
        self._last_calc = now
        return raw_tide

    # ---- Skyfield helper: next moon transit ----

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


# ---- module-level helpers ----

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